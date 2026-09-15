# entry point of the program
import argparse
import warnings
import argparse
import copy
import os
import random
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from baseline import LinearBaseline
from datahandler import DataLoader
from evaluation import evaluate
from gnn import GCNNodeClassifier


def merge_args_and_yaml(args, config_dict):
    for key, value in config_dict.items():
        if key in args.__dict__:
            warnings.warn(f"Command line argument '{key}' overwritten by config.")
        args.__dict__[key] = Namespace(**value) if isinstance(value, dict) else value
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingPipeline:
    def __init__(self, args):
        self.lr = args.lr
        self.epochs = args.numepochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seeds = list(args.seeds)
        self.datatype = args.datatype
        results_name = getattr(args, "name", "default")
        self.results_dir = Path(getattr(args, "results_dir", "results")) / results_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.train_loader, self.val_loader, self.test_loader = DataLoader(args).get_loaders()
        first_batch = next(iter(self.train_loader))
        
        if self.datatype == "graph":
            self.input_dim = int(first_batch.x.shape[1])
        else:
            self.input_dim = int(first_batch[0].flatten(start_dim=1).shape[1])

    def _predict(self, model, loader, split, seed):
        model.eval()
        rows = []
        with torch.no_grad():
            for batch in loader:
                if self.datatype == "graph":
                    batch = batch.to(self.device)
                    scores = model(batch).cpu().reshape(-1)
                    labels = batch.y.cpu().reshape(-1)
                    graph_ids = batch.batch.cpu().tolist()
                    subject_ids = [batch.subject_id[graph_id] for graph_id in graph_ids]
                    node_ids = batch.node_id.cpu().reshape(-1).tolist()
                else:
                    x_batch, y_batch, metadata = batch
                    scores = model(x_batch.to(self.device)).cpu().reshape(-1)
                    labels = y_batch.reshape(-1).cpu()
                    subject_ids = metadata["subject_id"]
                    node_ids = metadata["node_id"].reshape(-1).tolist()
                for subject_id, node_id, score, label in zip(
                    subject_ids,
                    node_ids,
                    scores.tolist(),
                    labels.tolist(),
                ):
                    rows.append({
                        "seed": seed,
                        "split": split,
                        "subject_id": subject_id,
                        "node_id": int(node_id),
                        "model_score": float(score),
                        "node_labels": int(label),
                    })
        return pd.DataFrame(rows)

    def _train_seed(self, seed):
        set_seed(seed)
        if self.datatype == "graph":
            model = GCNNodeClassifier(self.input_dim).to(self.device)
        else:   
            model = LinearBaseline(self.input_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        best_state = None
        best_val_loss = float("inf")
        print(f"Training with seed {seed}...")

        for e in range(self.epochs):
            model.train()
            for batch in self.train_loader:
                optimizer.zero_grad()
                if self.datatype == "graph":
                    batch = batch.to(self.device)
                    predictions = model(batch).reshape(-1)
                    targets = batch.y.float().reshape(-1)
                else:
                    x_batch, y_batch, _ = batch
                    predictions = model(x_batch.to(self.device)).reshape(-1)
                    targets = y_batch.float().to(self.device).reshape(-1)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    if self.datatype == "graph":
                        batch = batch.to(self.device)
                        predictions = model(batch).reshape(-1)
                        targets = batch.y.float().reshape(-1)
                        sample_count = batch.num_nodes
                    else:
                        x_batch, y_batch, _ = batch
                        predictions = model(x_batch.to(self.device)).reshape(-1)
                        targets = y_batch.float().to(self.device).reshape(-1)
                        sample_count = len(y_batch)
                    val_loss += criterion(
                        predictions, targets
                    ).item() * sample_count
            val_loss /= len(self.val_loader.dataset)
            print(f"Validation loss for epoch {e+1}: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        return model

    def train(self):
        prediction_frames = []
        metric_rows = []
        best_seed = None
        best_val_auprc = float("-inf")
        best_model_state = None
        best_seed_predictions = None

        for seed in self.seeds:
            model = self._train_seed(seed)
            seed_prediction_frames = []
            for split, loader in (
                ("train", self.train_loader),
                ("val", self.val_loader),
                ("test", self.test_loader),
            ):
                predictions = self._predict(model, loader, split, seed)
                prediction_frames.append(predictions)
                seed_prediction_frames.append(predictions)
                metric_rows.append({"seed": seed, "split": split, **evaluate(predictions)})

            seed_metrics = [row for row in metric_rows if row["seed"] == seed]
            val_metrics = next(row for row in seed_metrics if row["split"] == "val")
            if val_metrics["auprc"] > best_val_auprc:
                best_seed = seed
                best_val_auprc = val_metrics["auprc"]
                best_model_state = copy.deepcopy(model.state_dict())
                best_seed_predictions = pd.concat(seed_prediction_frames, ignore_index=True)

        predictions = pd.concat(prediction_frames, ignore_index=True)
        per_seed = pd.DataFrame(metric_rows)
        aggregate = per_seed.groupby("split")[
            ["auprc", "auroc", "top_k_dice", "prevalence"]
        ].agg(["mean", "std"])
        aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
        aggregate = aggregate.reset_index()
        predictions.to_csv(self.results_dir / "predictions.csv", index=False)
        per_seed.to_csv(self.results_dir / "evaluation_by_seed.csv", index=False)
        aggregate.to_csv(self.results_dir / "evaluation_summary.csv", index=False)
        best_seed_predictions.to_csv(
            self.results_dir / "best_seed_predictions.csv", index=False
        )
        torch.save(
            {
                "seed": best_seed,
                "input_dim": self.input_dim,
                "model_state_dict": best_model_state,
                "selection_metric": "val_auprc",
                "selection_score": best_val_auprc,
            },
            self.results_dir / "best_model.pt",
        )
        print(per_seed.to_string(index=False))
        print(aggregate.to_string(index=False))
        print(f"Best seed: {best_seed} (validation AUPRC={best_val_auprc:.6f})")
        return predictions, per_seed, aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open() as config_file:
        args = merge_args_and_yaml(args, yaml.safe_load(config_file))
    if not os.path.isabs(args.data):
        args.data = str((config_path.parent / args.data).resolve())
    TrainingPipeline(args).train()


if __name__ == "__main__":
    main()