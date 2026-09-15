import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate(df):
    """
    Evaluates node classification performance from a 2D matrix structure.
    
    Expected Columns / Layout:
    Index 0: subject_id     
    Index 1: node_id    
    Index 2: model_score 
    Index 3: node_labels 
        # seed  split subject_id  node_id  predict_score  real_label
    """


    df["model_score"] = pd.to_numeric(df["model_score"])
    df["node_labels"] = pd.to_numeric(df["node_labels"]).astype(int)

    # Micro-Averaged Pooled Metrics (AUPRC & AUROC)
    pooled_probs = df["model_score"].values
    pooled_labels = df["node_labels"].values
    
    final_auprc = average_precision_score(pooled_labels, pooled_probs)
    final_auroc = roc_auc_score(pooled_labels, pooled_probs)

    positive_baseline = np.mean(pooled_labels) # > 0.073
    
    # Top-k DICE (Macro Avg)
    subject_dice_scores = []

    for pat_id, group in df.groupby("subject_id"):
        
        subject_probs = group["model_score"].values
        subject_labels = group["node_labels"].values
        
        # k = total actual abnormal nodes for this subject
        k = int(np.sum(subject_labels == 1))
        if k == 0:
            print(f"Skipping subject {pat_id} for top-k DICE because k=0")
            continue

        # get indices of the top-k highest probability scores for this subject
        top_k_indices = set(np.argsort(subject_probs)[-k:])
        true_indices = set(np.where(subject_labels == 1)[0])
        
        # intersect top-k selections with actual targets: |pred (intersection) true|
        correct_guesses = len(top_k_indices & true_indices)

        # dice score
        patient_dice = correct_guesses / k
        subject_dice_scores.append(patient_dice)
        
    mean_top_k_dice = np.mean(subject_dice_scores)
    
    
    return {
        "auprc": final_auprc,
        "auroc": final_auroc,
        "top_k_dice": mean_top_k_dice,
        "prevalence": positive_baseline
    }
