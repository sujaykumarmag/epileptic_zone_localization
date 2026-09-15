

## Overall Results


### Model Results

Metrics are reported as mean ± standard deviation across five seeds. Bold
values mark the best value in each metric column; `gcn_multimodal_identity` is
the strongest overall model on the test set.

#### Train

| Model | AUPRC | AUROC | DICE |
|---|---:|---:|---:|
| baseline_multimodal | 0.3858 ± 0.0762 | **0.8545 ± 0.0255** | 0.4172 ± 0.0424 |
| baseline_only_a | 0.2957 ± 0.0540 | 0.8227 ± 0.0237 | 0.3456 ± 0.0446 |
| baseline_only_b | 0.3149 ± 0.0812 | 0.8037 ± 0.0359 | 0.3439 ± 0.0602 |
| **gcn_multimodal_identity** | **0.4295 ± 0.0127** | 0.8492 ± 0.0067 | **0.4485 ± 0.0149** |
| gcn_multimodal_real | 0.2001 ± 0.0121 | 0.7305 ± 0.0101 | 0.2454 ± 0.0261 |
| gcn_multimodal_shuffled | 0.1981 ± 0.0122 | 0.7293 ± 0.0104 | 0.2404 ± 0.0248 |
| gcn_only_a | 0.1734 ± 0.0026 | 0.7003 ± 0.0032 | 0.1942 ± 0.0021 |
| gcn_only_b | 0.1215 ± 0.0233 | 0.6118 ± 0.0338 | 0.1465 ± 0.0472 |

#### Validation

| Model | AUPRC | AUROC | DICE |
|---|---:|---:|---:|
| baseline_multimodal | 0.2558 ± 0.0637 | 0.7657 ± 0.0158 | 0.3053 ± 0.0530 |
| baseline_only_a | 0.1907 ± 0.0145 | 0.7334 ± 0.0150 | 0.2384 ± 0.0467 |
| baseline_only_b | 0.1926 ± 0.0436 | 0.6893 ± 0.0099 | 0.2302 ± 0.0427 |
| **gcn_multimodal_identity** | **0.3698 ± 0.0115** | **0.8092 ± 0.0043** | **0.4011 ± 0.0132** |
| gcn_multimodal_real | 0.1729 ± 0.0073 | 0.6607 ± 0.0103 | 0.2035 ± 0.0156 |
| gcn_multimodal_shuffled | 0.1714 ± 0.0086 | 0.6599 ± 0.0119 | 0.2021 ± 0.0134 |
| gcn_only_a | 0.1574 ± 0.0051 | 0.6427 ± 0.0051 | 0.1721 ± 0.0039 |
| gcn_only_b | 0.1040 ± 0.0196 | 0.5630 ± 0.0279 | 0.1260 ± 0.0306 |

#### Test

| Model | AUPRC | AUROC | DICE |
|---|---:|---:|---:|
| baseline_multimodal | 0.2773 ± 0.0499 | 0.7777 ± 0.0115 | 0.3347 ± 0.0391 |
| baseline_only_a | 0.2327 ± 0.0456 | 0.7577 ± 0.0218 | 0.2967 ± 0.0478 |
| baseline_only_b | 0.1966 ± 0.0244 | 0.6866 ± 0.0231 | 0.2382 ± 0.0398 |
| **gcn_multimodal_identity** | **0.4353 ± 0.0125** | **0.8598 ± 0.0027** | **0.4563 ± 0.0094** |
| gcn_multimodal_real | 0.1506 ± 0.0104 | 0.7044 ± 0.0089 | 0.2031 ± 0.0169 |
| gcn_multimodal_shuffled | 0.1488 ± 0.0104 | 0.7032 ± 0.0090 | 0.2009 ± 0.0190 |
| gcn_only_a | 0.1306 ± 0.0025 | 0.6789 ± 0.0031 | 0.1834 ± 0.0073 |
| gcn_only_b | 0.1101 ± 0.0192 | 0.5994 ± 0.0333 | 0.1067 ± 0.0234 |




### Simulation-Based Results (Given)

| Split | AUPRC | AUROC | Top-k DICE |
|---|---:|---:|---:|
| Test | 0.4048 | 0.7726 | 0.4563 |
| Train | 0.4147 | 0.7593 | 0.4970 |
| Validation | 0.4371 | 0.7722 | 0.5548 |





### Pearson & Spearman Correlation

Correlation between `model_score` and `sim_score`:

| Model | Split | Rows | Pearson | Spearman |
|---|---|---:|---:|---:|
| baseline_multimodal | Test | 1,904 | 0.2242 | 0.1531 |
| baseline_multimodal | Train | 5,712 | **0.3121** | 0.1721 |
| baseline_multimodal | Validation | 1,904 | **0.2580** | **0.1708** |
| baseline_only_a | Test | 1,904 | 0.1799 | 0.1323 |
| baseline_only_a | Train | 5,712 | 0.2205 | 0.1406 |
| baseline_only_a | Validation | 1,904 | 0.1457 | 0.1183 |
| baseline_only_b | Test | 1,904 | 0.1362 | 0.1488 |
| baseline_only_b | Train | 5,712 | 0.2618 | 0.1665 |
| baseline_only_b | Validation | 1,904 | 0.1853 | 0.1692 |
| gcn_multimodal_identity | Test | 1,904 | **0.2719** | 0.1255 |
| gcn_multimodal_identity | Train | 5,712 | 0.2714 | 0.1412 |
| gcn_multimodal_identity | Validation | 1,904 | 0.2538 | 0.1143 |
| gcn_multimodal_real | Test | 1,904 | 0.1762 | **0.1669** |
| gcn_multimodal_real | Train | 5,712 | 0.1960 | **0.1857** |
| gcn_multimodal_real | Validation | 1,904 | 0.1209 | -0.0005 |
| gcn_multimodal_shuffled | Test | 1,904 | 0.1754 | 0.1665 |
| gcn_multimodal_shuffled | Train | 5,712 | 0.1930 | 0.1815 |
| gcn_multimodal_shuffled | Validation | 1,904 | 0.1207 | -0.0006 |
| gcn_only_a | Test | 1,904 | 0.1348 | 0.1004 |
| gcn_only_a | Train | 5,712 | 0.1617 | 0.1400 |
| gcn_only_a | Validation | 1,904 | 0.0956 | -0.0110 |
| gcn_only_b | Test | 1,904 | 0.1445 | 0.1523 |
| gcn_only_b | Train | 5,712 | 0.1081 | 0.0712 |
| gcn_only_b | Validation | 1,904 | 0.1071 | 0.0207 |



### Discordant Pairs

Pairs are counted when `0.8 ≤ |model_score - sim_score| ≤ 1.0`.

| Model | Split | Discordant pairs | Total pairs | Percentage |
|---|---|---:|---:|---:|
| baseline_multimodal | Test | 59 | 1,904 | 3.0987% |
| baseline_multimodal | Train | 175 | 5,712 | 3.0637% |
| baseline_multimodal | Validation | 65 | 1,904 | 3.4139% |
| baseline_only_a | Test | 50 | 1,904 | 2.6261% |
| baseline_only_a | Train | 161 | 5,712 | 2.8186% |
| baseline_only_a | Validation | 59 | 1,904 | 3.0987% |
| baseline_only_b | Test | 66 | 1,904 | 3.4664% |
| baseline_only_b | Train | 178 | 5,712 | 3.1162% |
| baseline_only_b | Validation | 64 | 1,904 | 3.3613% |
| gcn_multimodal_identity | Test | 46 | 1,904 | **2.4160%** |
| gcn_multimodal_identity | Train | 158 | 5,712 | **2.7661%** |
| gcn_multimodal_identity | Validation | 53 | 1,904 | **2.7836%** |
| gcn_multimodal_real | Test | 44 | 1,904 | **2.3109%** |
| gcn_multimodal_real | Train | 169 | 5,712 | 2.9587% |
| gcn_multimodal_real | Validation | 68 | 1,904 | 3.5714% |
| gcn_multimodal_shuffled | Test | 46 | 1,904 | 2.4160% |
| gcn_multimodal_shuffled | Train | 174 | 5,712 | 3.0462% |
| gcn_multimodal_shuffled | Validation | 68 | 1,904 | 3.5714% |
| gcn_only_a | Test | 48 | 1,904 | 2.5210% |
| gcn_only_a | Train | 177 | 5,712 | 3.0987% |
| gcn_only_a | Validation | 65 | 1,904 | 3.4139% |
| gcn_only_b | Test | 62 | 1,904 | 3.2563% |
| gcn_only_b | Train | 214 | 5,712 | 3.7465% |
| gcn_only_b | Validation | 73 | 1,904 | 3.8340% |



### Naive Average Results

Naive average score: `(model_score + sim_score) / 2`.

| Model | Split | AUPRC | AUROC | Top-k DICE | Prevalence |
|---|---|---:|---:|---:|---:|
| baseline_multimodal | Test | 0.6041 | 0.8763 | 0.5548 | 0.0720 |
| **baseline_multimodal** | **Train** | **0.6778** | **0.9197** | **0.6202** | 0.0734 |
| **baseline_multimodal** | **Validation** | 0.6071 | 0.8564 | **0.6248** | 0.0714 |
| baseline_only_a | Test | 0.5682 | 0.8631 | 0.5106 | 0.0720 |
| baseline_only_a | Train | 0.6184 | 0.9020 | 0.5880 | 0.0734 |
| baseline_only_a | Validation | 0.5806 | 0.8540 | 0.6021 | 0.0714 |
| baseline_only_b | Test | 0.5242 | 0.8118 | 0.5343 | 0.0720 |
| baseline_only_b | Train | 0.6477 | 0.9129 | 0.6127 | 0.0734 |
| baseline_only_b | Validation | 0.5434 | 0.8164 | 0.5736 | 0.0714 |
| gcn_multimodal_identity | Test | **0.6502** | **0.9108** | **0.5572** | 0.0720 |
| gcn_multimodal_identity | Train | 0.6640 | 0.9055 | 0.6189 | 0.0734 |
| gcn_multimodal_identity | Validation | **0.6300** | **0.8821** | 0.6207 | 0.0714 |
| gcn_multimodal_real | Test | 0.4843 | 0.8131 | 0.4634 | 0.0720 |
| gcn_multimodal_real | Train | 0.5497 | 0.8458 | 0.5300 | 0.0734 |
| gcn_multimodal_real | Validation | 0.5002 | 0.8290 | 0.5607 | 0.0714 |
| gcn_multimodal_shuffled | Test | 0.4835 | 0.8125 | 0.4634 | 0.0720 |
| gcn_multimodal_shuffled | Train | 0.5484 | 0.8451 | 0.5320 | 0.0734 |
| gcn_multimodal_shuffled | Validation | 0.4998 | 0.8280 | 0.5607 | 0.0714 |
| gcn_only_a | Test | 0.4696 | 0.8082 | 0.4554 | 0.0720 |
| gcn_only_a | Train | 0.5262 | 0.8347 | 0.5193 | 0.0734 |
| gcn_only_a | Validation | 0.4916 | 0.8194 | 0.5503 | 0.0714 |
| gcn_only_b | Test | 0.4729 | 0.7886 | 0.4583 | 0.0720 |
| gcn_only_b | Train | 0.5006 | 0.8046 | 0.5162 | 0.0734 |
| gcn_only_b | Validation | 0.4495 | 0.7784 | 0.5225 | 0.0714 |



## Limitations
