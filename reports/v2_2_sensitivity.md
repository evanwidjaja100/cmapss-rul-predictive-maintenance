# V2.2 sensor sensitivity (final model: xgb_w90_d6)

Method: per-sensor occlusion on the fixed outer pseudo-test manifests
of the 85 development engines (425 checkpoints, 5 fractions).
The replacement value is the engine's PREFIX-ONLY observed mean (cycles
<= cutoff; never future rows). Rows are aligned by (engine_id,
cutoff_cycle); RMSE deltas use exactly aligned rows.
Descriptive post-freeze analysis; NOT SHAP values; not used for selection.

| Sensor | mean |delta| | RMSE delta | overprediction-worsened share |
|---|---|---|---|
| sensor_4 | 16.977 | +16.486 | 0.972 |
| sensor_11 | 14.179 | +13.406 | 0.922 |
| sensor_3 | 10.459 | +9.183 | 0.958 |
| sensor_9 | 10.350 | +10.066 | 0.953 |
| sensor_12 | 10.290 | +9.452 | 0.908 |
| sensor_7 | 9.234 | +8.164 | 0.944 |
| sensor_20 | 7.487 | +6.122 | 0.981 |
| sensor_21 | 6.632 | +5.751 | 0.887 |
| sensor_15 | 5.873 | +5.589 | 0.913 |
| sensor_2 | 4.631 | +4.306 | 0.713 |
| sensor_14 | 4.055 | +3.904 | 0.715 |
| sensor_17 | 3.734 | +3.777 | 0.866 |
| sensor_8 | 2.924 | +2.128 | 0.899 |
| sensor_13 | 2.522 | +1.827 | 0.753 |
| sensor_6 | 0.250 | +0.050 | 0.014 |
| sensor_5 | 0.000 | +0.000 | 0.000 |
| sensor_1 | 0.000 | +0.000 | 0.000 |
| sensor_10 | 0.000 | +0.000 | 0.000 |
| sensor_16 | 0.000 | +0.000 | 0.000 |
| sensor_19 | 0.000 | +0.000 | 0.000 |
| sensor_18 | 0.000 | +0.000 | 0.000 |

## Region (fraction) sensitivity — top-3 sensors per region

fraction 0.25: sensor_7 (12.091), sensor_11 (11.532), sensor_9 (11.000)
fraction 0.45: sensor_4 (16.121), sensor_7 (12.270), sensor_11 (12.086)
fraction 0.65: sensor_4 (13.800), sensor_11 (12.570), sensor_12 (11.317)
fraction 0.80: sensor_4 (17.704), sensor_11 (13.908), sensor_3 (11.477)
fraction 0.95: sensor_4 (27.412), sensor_11 (20.800), sensor_12 (12.094)
