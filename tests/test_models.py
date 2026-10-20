import pytest
import torch

from dashcam_risk.models import (
    FrameGRUClassifier,
    LastFrameClassifier,
    MajorityClassClassifier,
    build_model,
)


def test_majority_emits_fitted_prior():
    model = MajorityClassClassifier()
    labels = torch.tensor([0, 0, 1])
    model.fit_prior(labels)
    out = model(torch.zeros(4, 8, 3, 224, 224))
    assert out.shape == (4, 2)
    probs = torch.softmax(out, dim=-1)[0]
    assert probs[1].item() == pytest.approx(1 / 3, abs=1e-5)


def test_majority_single_class_train_does_not_crash():
    model = MajorityClassClassifier()
    model.fit_prior(torch.zeros(3, dtype=torch.long))
    out = model(torch.zeros(2, 8, 3, 224, 224))
    assert torch.isfinite(out).all()


def test_build_model_dispatch():
    assert isinstance(build_model("majority"), MajorityClassClassifier)
    assert isinstance(build_model("last_frame"), LastFrameClassifier)
    assert isinstance(build_model("gru"), FrameGRUClassifier)
    with pytest.raises(ValueError):
        build_model("transformer")
