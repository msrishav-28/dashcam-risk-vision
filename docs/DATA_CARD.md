# Data card

## Intended public source

[Nexar Dashcam Collision Prediction](https://arxiv.org/abs/2503.03848) (~1,500 clips,
~40 s, event type, weather, scene, event time, first-alert time).

Swap the adapter in `src/dashcam_risk/schema.py` if you use another set.
Do not commit the videos.

## Manifest schema (`data/processed/manifest.csv`)

| column | meaning |
|---|---|
| clip_id | primary split key |
| trip_id | optional group key |
| path | relative path to video or frame folder |
| label | `risk` or `normal` |
| event_type | collision / near-miss / normal |
| weather | rain / clear / unknown |
| time_of_day | day / night / unknown |
| scene | urban / highway / rural / unknown |
| duration_s | clip length |
| fps | frames per second if known |
| event_time_s | time of event, null on normal |
| alert_time_s | first time the event is considered predictable |
| license | dataset licence string |

## Licence

Follow the dataset licence. Nexar challenge materials are typically CC BY;
confirm on the official page before redistributing frames.
