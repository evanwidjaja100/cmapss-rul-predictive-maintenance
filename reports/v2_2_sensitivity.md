# V2.2 sensor sensitivity (final model: xgb_w90_d6)

Method: per-sensor occlusion (sensor values replaced by the engine's own
mean over its observed history) on the fixed outer pseudo-test manifests
of the 85 development engines (425 checkpoints, 5 fractions).
Descriptive post-freeze analysis; NOT SHAP values; not used for selection.

| Sensor | mean |delta| | RMSE delta | overprediction-worsened share |
|---|---|---|---|
| sensor_4 | 15.069 | +15.597 | 0.918 |
| sensor_11 | 13.124 | +13.074 | 0.882 |
| sensor_12 | 10.263 | +9.470 | 0.899 |
| sensor_9 | 10.011 | +9.303 | 0.927 |
| sensor_3 | 9.533 | +8.303 | 0.951 |
| sensor_20 | 8.504 | +6.634 | 0.981 |
| sensor_7 | 7.449 | +6.379 | 0.899 |
| sensor_21 | 6.383 | +5.675 | 0.859 |
| sensor_15 | 5.859 | +5.528 | 0.873 |
| sensor_2 | 4.580 | +4.618 | 0.692 |
| sensor_14 | 4.058 | +3.618 | 0.668 |
| sensor_17 | 3.699 | +3.814 | 0.826 |
| sensor_13 | 3.168 | +1.981 | 0.769 |
| sensor_8 | 3.057 | +2.019 | 0.918 |
| sensor_6 | 0.507 | +0.074 | 0.031 |
| sensor_5 | 0.000 | +0.000 | 0.000 |
| sensor_1 | 0.000 | +0.000 | 0.000 |
| sensor_10 | 0.000 | +0.000 | 0.000 |
| sensor_16 | 0.000 | +0.000 | 0.000 |
| sensor_19 | 0.000 | +0.000 | 0.000 |
| sensor_18 | 0.000 | +0.000 | 0.000 |

## Region (fraction) sensitivity — top-3 sensors per region

fraction 0.25: sensor_20 (11.417), sensor_9 (11.098), sensor_11 (8.246)
fraction 0.45: sensor_4 (12.414), sensor_20 (11.118), sensor_11 (11.049)
fraction 0.65: sensor_4 (12.204), sensor_11 (11.996), sensor_12 (10.918)
fraction 0.80: sensor_4 (16.165), sensor_11 (13.498), sensor_9 (10.824)
fraction 0.95: sensor_4 (26.739), sensor_11 (20.831), sensor_12 (11.916)
