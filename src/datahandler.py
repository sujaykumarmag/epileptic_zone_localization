import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Dataset as PyGDataset
from torch_geometric.loader import DataLoader as PyGDataLoader




class GraphDataset(PyGDataset):
    """
    Custom PyTorch Geometric Dataset loading all data frames 
    with index-level row alignment per subject.
    """
    def __init__(
        self,
        args,
        subjects_df,
        mod_a_df,
        mod_b_df,
        labels_df,
        sims_score_df,
    ):
        super().__init__()
    
        # args
        self.mod_a = args.modality.include_mod_a
        self.mod_b = args.modality.include_mod_b

        self.mod_a_df = mod_a_df
        self.mod_b_df = mod_b_df
        self.adj_path = os.path.join(args.data, "adjacency")
        self.adjacency_mode = getattr(args, "adjacency_mode", "real")
    
    
        # dataframes
        self.subjects_df = subjects_df.reset_index(drop=True)
        self.labels_df = labels_df.drop("resected",axis=1) # not to include while training
        self.labels_df["uid"] = (
            self.labels_df["subject_id"].astype(str)
                + "_"
                + self.labels_df["node_id"].astype(str)
            )
        self.mod_a_df["uid"] = (
                        self.mod_a_df["subject_id"].astype(str)
                        + "_"
                        + self.mod_a_df["node_id"].astype(str)
                    )
        self.mod_b_df["uid"] = (
                        self.mod_b_df["subject_id"].astype(str)
                        + "_"
                        + self.mod_b_df["node_id"].astype(str)
                    )
        self.sims_score_df = sims_score_df # for post processing 
    
        self.mod_a_features = ["ct_z","gmv_z","sa_z","curv_z","t1t2_z","lgi_z"] 
        self.mod_b_features = ["rel_delta",	"rel_theta", "rel_alpha", "rel_beta", "rel_gamma", "spike_rate", "hfo_rate"]
            
            
    
        self.node_level_features = self.labels_df.copy()
        if self.mod_a:
            self.node_level_features = self.node_level_features.merge(
                    self.mod_a_df[["uid", *self.mod_a_features]], on="uid", how="inner"
                )

        if self.mod_b:
            self.node_level_features = self.node_level_features.merge(
                    self.mod_b_df[["uid", *self.mod_b_features]], on="uid", how="inner"
                )

        self.feature_columns = []
        if self.mod_a:
            self.feature_columns.extend(self.mod_a_features)
        if self.mod_b:
            self.feature_columns.extend(self.mod_b_features)
        if not self.feature_columns:
            raise ValueError("At least one modality must be enabled for GraphDataset.")

    def len(self):
        return len(self.subjects_df)

    def get(self, idx):
        subject_id = self.subjects_df.iloc[idx]["subject_id"]
        adjacency = np.asarray(
            np.load(os.path.join(self.adj_path, f"{subject_id}_adj.npy")),
            dtype=np.float32,
        )
        if adjacency.shape != (68, 68):
            raise ValueError(f"Expected a 68x68 adjacency matrix for {subject_id}.")

        subject_rows = self.node_level_features[
            self.node_level_features["subject_id"] == subject_id
        ].sort_values("node_id")
        if len(subject_rows) != 68 or not np.array_equal(
            subject_rows["node_id"].to_numpy(), np.arange(68)
        ):
            raise ValueError(f"Expected 68 aligned nodes for {subject_id}.")

        if self.adjacency_mode == "identity":
            adjacency = np.eye(68, dtype=np.float32)
            node_order = np.arange(68)
        elif self.adjacency_mode == "shuffled":
            rng = np.random.default_rng(100 + idx)
            node_order = rng.permutation(68)
            adjacency = adjacency[np.ix_(node_order, node_order)]
        elif self.adjacency_mode == "real":
            node_order = np.arange(68)
        else:
            raise ValueError(f"Unknown adjacency mode: {self.adjacency_mode}")

        subject_rows = subject_rows.iloc[node_order].reset_index(drop=True)
        node_features = subject_rows[self.feature_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(node_features).all():
            raise ValueError(f"Missing or invalid node feature for {subject_id}.")

        adjacency = np.maximum(adjacency, adjacency.T)
        upper_edges = np.argwhere(np.triu(adjacency, k=1) > 0)
        reverse_edges = upper_edges[:, [1, 0]]
        edge_pairs = np.vstack((upper_edges, reverse_edges))
        edge_index = torch.tensor(edge_pairs.T, dtype=torch.long)
        edge_weight = torch.tensor(
            adjacency[edge_pairs[:, 0], edge_pairs[:, 1]], dtype=torch.float32
        )
        labels = torch.tensor(
            subject_rows["is_abnormal"].to_numpy(), dtype=torch.long
        )

        data = Data(
            x=torch.tensor(node_features),
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=labels,
            node_id=torch.tensor(node_order, dtype=torch.long),
        )

        data.subject_id = subject_id
        return data
    



class NonGraphDataset(TorchDataset):
    """
    Standard PyTorch Dataset for non-graph models.
    All scalar components are bypassed, and all tables are natively bound.
    """
    def __init__(self, args, subjects_df, mod_a_df, mod_b_df, labels_df, sims_score_df):
        super().__init__()

        # args
        self.mod_a = args.modality.include_mod_a
        self.mod_b = args.modality.include_mod_b

        self.mod_a_df = mod_a_df
        self.mod_b_df = mod_b_df
        self.adj_path = os.path.join(args.data, "adjacency")


        # dataframes
        self.subjects_df = subjects_df.reset_index(drop=True)
        self.labels_df = labels_df.drop("resected",axis=1) # not to include while training
        self.labels_df["uid"] = (
            self.labels_df["subject_id"].astype(str)
            + "_"
            + self.labels_df["node_id"].astype(str)
        )
        self.mod_a_df["uid"] = (
                    self.mod_a_df["subject_id"].astype(str)
                    + "_"
                    + self.mod_a_df["node_id"].astype(str)
                )
        self.mod_b_df["uid"] = (
                    self.mod_b_df["subject_id"].astype(str)
                    + "_"
                    + self.mod_b_df["node_id"].astype(str)
                )
        self.sims_score_df = sims_score_df # for post processing 

        self.mod_a_features = ["ct_z","gmv_z","sa_z","curv_z","t1t2_z","lgi_z"] 
        self.mod_b_features = ["rel_delta",	"rel_theta", "rel_alpha", "rel_beta", "rel_gamma", "spike_rate", "hfo_rate"]

        
        

        self.node_level_features = self.labels_df.copy()
        if self.mod_a:
            self.node_level_features = self.node_level_features.merge(
                self.mod_a_df[["uid", *self.mod_a_features]], on="uid", how="inner"
            )
        if self.mod_b:
            self.node_level_features = self.node_level_features.merge(
                self.mod_b_df[["uid", *self.mod_b_features]], on="uid", how="inner"
            )

        self.prepare_data()



    def prepare_data(self):
        """ Prepares the dataset by loading adjacency matrices and aligning node-level features with labels."""
        self.data = []
        for sub_id in self.subjects_df["subject_id"]:
            matrix = np.load(os.path.join(self.adj_path, f"{sub_id}_adj.npy"))
            if matrix.shape != (68, 68):
                raise ValueError(f"Expected a 68x68 adjacency matrix for {sub_id}.")

            subject_rows = self.node_level_features[
                self.node_level_features["subject_id"] == sub_id
            ].sort_values("node_id")
            if len(subject_rows) != 68:
                raise ValueError(f"Expected 68 aligned nodes for {sub_id}.")

            for _, row in subject_rows.iterrows():
                node_id = int(row["node_id"])
                feature_values = [matrix[node_id]]
                if self.mod_a:
                    feature_values.append(row[self.mod_a_features].to_numpy(dtype=float))
                if self.mod_b:
                    feature_values.append(row[self.mod_b_features].to_numpy(dtype=float))

                x = np.concatenate(feature_values).astype(np.float32)
                if not np.isfinite(x).all():
                    raise ValueError(f"Missing or invalid feature value for {sub_id}, node {node_id}.")
                self.data.append({
                    "x": x,
                    "y": int(row["is_abnormal"]),
                    "subject_id": sub_id,
                    "node_id": node_id,
                })

        self.X = np.stack([item["x"] for item in self.data])
        self.y = np.asarray([item["y"] for item in self.data], dtype=np.int64)




    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        item = self.data[idx]
        return item["x"], item["y"], {"subject_id": item["subject_id"], "node_id": item["node_id"]}








class DataLoader():
    """
    Main Data Loader class responsible for orchestrating files, splitting, 
    and outputting operational PyTorch loaders.
    """
    def __init__(self, args):
        self.args = args
        self.data_dir = args.data
        self.batch_size = args.bz    
        self.datatype = args.datatype 

    
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None



    def null_handler_b(self, modality, subject_ids):
        """
        Imputes absent modality B subjects with node-wise Gaussian samples.
        """
        # preprocessing (dropping region)
        modality = modality.drop("region", axis=1)
        features = list(modality.columns[2:])

        rng = np.random.default_rng(1) # fixed seed for reproducibe data 
        sampled_rows = []
        present_subjects = set(modality["subject_id"].unique())
        missing_subjects = sorted(set(subject_ids) - present_subjects)

        for node_id, node_group in modality.groupby("node_id"):
            observed = node_group[features].dropna()
            
            mean = observed.mean().to_numpy(dtype=float)
            covariance = np.atleast_2d(observed.cov().to_numpy(dtype=float))
            # optional suggested by copilot
            covariance = np.nan_to_num(covariance, nan=0.0) 
            covariance += np.eye(len(features)) * 1e-8

            for subject_id in missing_subjects:
                sampled_vector = rng.multivariate_normal(mean, covariance)
                sampled_rows.append(
                    [subject_id, node_id, *sampled_vector]
                )

        imputed = pd.DataFrame(sampled_rows, columns=modality.columns)
        return pd.concat([modality, imputed], ignore_index=True)
        
        
    def null_handler_a(self, modality):
        """
        Handles missing values in the modality A dataframes.
        """
        modality = modality.drop("region", axis=1)
        modality = modality.fillna(0)
        return modality



    def load(self):
        """
        Loads all .csvs 
        """
        subjects_df = pd.read_csv(os.path.join(self.data_dir, "subjects.csv"))
        labels_df = pd.read_csv(os.path.join(self.data_dir, "node_labels.csv"))
        sims_score_df = pd.read_csv(os.path.join(self.data_dir, "simulation_scores.csv")) # Parsed correctly from file structure

        mod_a_df = pd.read_csv(os.path.join(self.data_dir, "modality_A.csv"))
        mod_a_df = self.null_handler_a(mod_a_df)

        mod_b_df = pd.read_csv(os.path.join(self.data_dir, "modality_B.csv"))
        mod_b_df = self.null_handler_b(mod_b_df, subjects_df["subject_id"])


        # Splitting
        train_subs = subjects_df[subjects_df['split'] == 'train']
        val_subs   = subjects_df[subjects_df['split'] == 'val']
        test_subs  = subjects_df[subjects_df['split'] == 'test']

        if self.datatype == "nongraph":
            # For baseline
            self.train_dataset = NonGraphDataset(self.args, train_subs, mod_a_df, mod_b_df, labels_df, sims_score_df)
            self.val_dataset = NonGraphDataset(self.args, val_subs, mod_a_df, mod_b_df, labels_df, sims_score_df)
            self.test_dataset = NonGraphDataset(self.args, test_subs, mod_a_df, mod_b_df, labels_df, sims_score_df)
        else:
            # For graphs
            self.train_dataset = GraphDataset(
                self.args, train_subs, mod_a_df, mod_b_df, labels_df, sims_score_df,
            )
            self.val_dataset = GraphDataset(
                self.args, val_subs, mod_a_df, mod_b_df, labels_df, sims_score_df,
               
            )
            self.test_dataset = GraphDataset(
                self.args, test_subs, mod_a_df, mod_b_df, labels_df, sims_score_df,)
        
        print(f"Loaded Splits -> Train Subjects: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")

    def get_loaders(self):
        """
        Generates functional training, validation, and test loaders.
        """
        if self.train_dataset is None:
            self.load()

        if self.datatype == "nongraph":
            train_loader = TorchDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader   = TorchDataLoader(self.val_dataset, batch_size=1, shuffle=False)
            test_loader  = TorchDataLoader(self.test_dataset, batch_size=1, shuffle=False)
        else:
            train_loader = PyGDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader   = PyGDataLoader(self.val_dataset, batch_size=1, shuffle=False)
            test_loader  = PyGDataLoader(self.test_dataset, batch_size=1, shuffle=False)
        
        return train_loader, val_loader, test_loader


