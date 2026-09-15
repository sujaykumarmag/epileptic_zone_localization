# Analysis

## 1. Data Handling

### Q. Missing cell data in Modality_A

**A.** In Modality_A, I have use imputation method (0 as <NA>).

**Reason:** The choice of removing data because of 6% is not valid, as we have
less data and also choice of average imputation is however valid, but
approximately the avg is 0, because of z-score.

### Q. Missing Patient data in Modality_B

**A.** In Modality_B, I have sampled from Gaussian Distribution (multivariate ~7)

**Reason:** In GNNs, when the node features are absent, we train it with random
features, so I have also used this method and for parameters of Gaussian,
Assuming each subject is iid the average of each node features and the std_dev
is used.

## 2. Baseline Model

### Q. choice of MLP

**A.** For a simple baseline (2 layers with 1 GELU unit)

**Reason:** want to use pytorch across the pipeline

## 3. Graph Model

### Q. choice of GCN

**A.** Because of weighted graphs acc to [1] and simplistic nature.

**Reason:** An simpler version of the model is used, due to the time-sensitive
nature.

### Q. question of normalization

**A.** Acc to the GCN paper [2] I have used normalization w.r.t the degree
matrices, which includes self edges.

### Q. Is the Graph Structure really contributing.

**A.** Not able to find it (but the reported results suggest that, the trained
GCN was not able to surpass the baseline)

**Outcome:** There is still room for improvement.

## 4. Multimodal fusion

### Q. modality_A v/s modality_B v/s multi_modality

**A.** From the test results, multimodality > modality_A > modality_B

**Outcome:** Yes, both modalities are better predictors

## References

[1]. https://pytorch-geometric.readthedocs.io/en/2.6.1/cheatsheet/gnn_cheatsheet.html

[2]. https://arxiv.org/abs/1609.02907
