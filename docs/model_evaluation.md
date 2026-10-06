# Model Evaluation Report

Generated: 2026-10-05 20:38
Model: `C:\Users\Admin\Desktop\CampusHub\models\helmet_model.pt`

## Speed

Source: `samples/demo.mp4`, frames measured: 175

- Average latency per frame (whole pipeline): **23.6 ms**
- 95th percentile latency: 25.0 ms
- Throughput: **42.3 FPS** (target 20-30 FPS)

## Notes

- Precision for no_helmet matters most for avoiding false alerts; recall for no_helmet for not missing violations.
- Record the camera position, lighting and hardware used, and any known failure cases, below.
