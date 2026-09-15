# Simulation-Informed AI


This project predicts abnormal nodes in subject-level graphs. Each subject is a
weighted graph with 68 aligned nodes. The model performs node-level binary
classification and writes one probability for every node.

## Requirements

- Python 3.12 (developed with Python 3.12.14)
- PyTorch and PyTorch Geometric
- Dependencies listed in `requirements.txt`

Create an environment and install the dependencies from the project root:
Add the data/ folder in the root itself 

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```



## Run the experiments

Run the complete experiment suite from the project root:

```bash
cd src/
bash run.bash
```


To run each experiment manually:

```bash
cd src/
python main.py --config args/args_gcn_multimodal_real.yml
```

Results are saved under `src/results/<experiment-name>/`:

- `predictions.csv`: node-level predictions
- `evaluation_by_seed.csv`: metrics for each random seed
- `evaluation_summary.csv`: mean and standard deviation across seeds
- `best_seed_predictions.csv`: predictions from the best validation seed
- `best_model.pt`: saved model checkpoint

## Repository layout

```text
candidate_package/
├── data/                    
├── src/
│   ├── args/
│   │   ├── args_baseline_multi.yml                 # baseline model with both modalities
│   │   ├── args_baseline_only_a.yml                # baseline model with modality A
│   │   ├── args_baseline_only_b.yml                # baseline model with modality B
│   │   ├── args_gcn_multimodal_ident.yml           # GCN model with both modalities (identity)
│   │   ├── args_gcn_multimodal_real.yml            # GCN model with both modalities (Real)
│   │   ├── args_gcn_multimodal_shuffled.yml        # GCN model with both modalities (shuffled)
│   │   ├── args_gcn_only_a.yml                     # GCN model with modality A
│   │   └── args_gcn_only_b.yml                     # GCN model with modality B
│   │ 
│   │ 
│   ├── baseline.py                                 # non-graph baseline
│   ├── datahandler.py                              # graph and non-graph datasets with Dataloader
│   ├── evaluation.py                               # evaluation metrics
│   ├── gnn.py                                      # weighted GCN node classifier
│   ├── main.py                                     # training and prediction pipeline
│   └── run.bash                                    # complete experiment runner
│
│
├── pipeline_check.ipynb                            # data and graph construction checks
├── post_process.ipynb                              # simulation-score post-processing
├── REPORT.md                                       # quantitative results
├── README.md
├── ANALYSIS.md                                     # analysis and findings
└── requirements.txt
```



## Notebooks

- `pipeline_check.ipynb` checks data alignment, missing values, adjacency
	matrices, and PyG graph construction.
- `post_process.ipynb` evaluates and compares the independent simulation
	scores. This analysis is separate from the training pipeline.

See `data/README_DATA.md` for complete column definitions, missing-data
details, and data-use restrictions.


See the [analysis](ANALYSIS.md), [report](REPORT.md)


