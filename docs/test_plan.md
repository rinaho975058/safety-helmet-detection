# Test Plan and Results

**Automated tests:** run `pytest`. They cover association, alerts, statistics, temporal logic, configuration and the pipeline.

**Manual tests:** these need a camera, the real model and real scenes. Fill in the Result column (Pass / Fail plus notes) and the conditions used.

Hardware: ____________________  Model: ____________________  Date: __________

| # | Test | How | Expected result | Result |
|---|---|---|---|---|
| 1 | Camera | `python main.py --start` | Webcam opens and boxes appear | |
| 2 | Video | `--source videos/test.mp4` | Whole video processes and the summary is printed at the end | |
| 3 | Image | `--image photo.jpg` | `photo_result.jpg` is saved with boxes and statuses | |
| 4 | Person detection | Walk through the view | Each person gets a box and a stable `#id` | |
| 5 | Helmet | Wear a helmet facing the camera | Green **Helmet** box | |
| 6 | No helmet | Remove the helmet | After about 1 s, red **No Helmet** and one alert | |
| 7 | Occlusion | Cover the head with a hand or paper | **Unknown**, or keeps the previous status. No new alert | |
| 8 | Distance | Stand near, medium and far | Note the distance where detection stops working | |
| 9 | Lighting | Daylight, indoor, low light, backlight | Note the accuracy in each condition | |
| 10 | Multiple people | 2-4 people, some with helmets | Each person gets their own correct status | |
| 11 | Association | Two people close together, one with a helmet | The helmet is assigned to the right person, or both are Unknown. Never the wrong person | |
| 12 | Temporal | Stay without a helmet for 30 s | Exactly **one** alert | |
| 13 | Cooldown | Put the helmet on and take it off within 10 s | No second alert within the cooldown | |
| 14 | Alerts off | Press `A`, then remove the helmet | No alert, and "ALERTS OFF" is shown | |
| 15 | Statistics | Compare the panel with the real scene | Counts and violation rate match | |
| 16 | Event log | Check `logs/events_*.csv` | One row per alert, with no personal data | |
| 17 | Snapshot | `snapshots_enabled: true`, trigger an alert, press `S` | JPEGs saved, and "SNAPSHOTS ON" is shown | |
| 18 | Camera unplugged | Unplug the webcam while monitoring | Reconnect attempts, then a clear error. No crash | |
| 19 | Missing model | Rename `models/helmet_model.pt` | Warning shown, people still detected, all statuses Unknown | |
| 20 | Bad source | `--source missing.mp4` | Clear "Video file not found" message | |
| 21 | Performance | `python scripts/evaluate.py --source 0 --speed-only` | FPS is recorded (target 20-30) | |

## Known failure cases found

| Situation | What happened | Mitigation |
|---|---|---|
| | | |
