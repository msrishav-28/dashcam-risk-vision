from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dashcam_risk.dataset import ClipFrameDataset
from dashcam_risk.metrics import classification_report
from dashcam_risk.models import build_model
from dashcam_risk.splits import clip_level_split, load_manifest, split_counts


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        logits = model(batch["frames"].to(device))
        prob = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        ys.append(batch["y"].numpy())
        ps.append(prob)
    import numpy as np

    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    return classification_report(y, p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/processed/example_manifest.csv")
    p.add_argument("--model", default="gru")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="experiments/last_run.json")
    args = p.parse_args()

    df = load_manifest(args.manifest)
    parts = clip_level_split(df, seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_ds = ClipFrameDataset(parts["train"], n_frames=args.n_frames, image_size=args.image_size)
    val_ds = ClipFrameDataset(parts["val"], n_frames=args.n_frames, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch["frames"].to(device))
            loss = loss_fn(logits, batch["y"].to(device))
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(batch["y"])
            n += len(batch["y"])
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": running / max(n, 1), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(row)

    payload = {
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "splits": split_counts(parts),
        "history": history,
        "note": "Placeholder tensors if frame files are missing. Do not cite as a result.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
