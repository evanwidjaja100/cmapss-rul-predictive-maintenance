# V2.2 error analysis (frozen model, POST-HOC official FD001)

The official FD001 labels are permanently post-hoc (inspected in the V2-0
audit). This analysis is descriptive; it never selects or tunes the model.

Overall: mean signed error +19.63 cycles; overprediction share 0.91.

| observed bucket | n engines | mean signed error | overprediction share | mean abs error |
|---|---|---|---|---|
| [0,45) | 4 | +49.410 | 1.000 | 49.410 |
| [45,90) | 22 | +29.019 | 0.955 | 29.452 |
| [90,128) | 18 | +14.981 | 0.944 | 16.331 |
| [128,200) | 48 | +15.286 | 0.896 | 16.883 |
| [200,10000) | 8 | +15.426 | 0.750 | 21.692 |

Descriptive observation: among observed-history buckets below 200 cycles, signed errors and absolute errors are larger. This is a post-hoc descriptive finding; it does NOT drive serving behavior (the app exposes only objective padding/history facts).
