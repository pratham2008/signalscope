import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve

os.makedirs("report/figures", exist_ok=True)

def compute_metrics(y_true, y_probs, operating_threshold=None, target_fpr=0.05):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    if operating_threshold is None:
        idx_fpr = np.argmin(np.abs(fpr - target_fpr))
        operating_threshold = float(thresholds[idx_fpr])
        derived_fpr = float(fpr[idx_fpr])
    else:
        derived_fpr = float(np.mean((y_probs >= operating_threshold) & (y_true == 0)))

    auc = roc_auc_score(y_true, y_probs)
    y_pred = (y_probs >= operating_threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)

    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for _ in range(1000):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        bootstrapped_scores.append(roc_auc_score(y_true[indices], y_probs[indices]))
    
    ci_lower = float(np.percentile(bootstrapped_scores, 2.5))
    ci_upper = float(np.percentile(bootstrapped_scores, 97.5))

    return {
        "auc": auc,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "operating_threshold": operating_threshold,
        "fpr_at_threshold": derived_fpr,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_probs": y_probs
    }

def generate_plots(results, prefix=""):
    cm = results["confusion_matrix"]
    threshold = results["operating_threshold"]
    y_true = results["y_true"]
    y_probs = results["y_probs"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Synthetic'], 
                yticklabels=['Real', 'Synthetic'])
    plt.title(f"Confusion Matrix @ Threshold {threshold:.2f}")
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"report/figures/{prefix}confusion_matrix.png", dpi=300)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Detector')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.title("Reliability / Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"report/figures/{prefix}calibration_curve.png", dpi=300)
    plt.close()
    print("Figures exported to report/figures/")

def run_dummy_verification():
    print("Generating dummy validation and unseen prediction sets...")
    np.random.seed(42)

    val_y = np.random.binomial(1, 0.5, 1000)
    val_probs = np.clip(val_y * 0.7 + np.random.normal(0.2, 0.15, 1000), 0.0, 1.0)
    val_results = compute_metrics(val_y, val_probs, target_fpr=0.05)
    tuned_threshold = val_results["operating_threshold"]

    test_ids = [f"img_{i:04d}" for i in range(3000)]
    test_y = np.random.binomial(1, 0.5, 3000)
    test_probs = np.clip(test_y * 0.6 + np.random.normal(0.25, 0.18, 3000), 0.0, 1.0)
    
    df_test = pd.DataFrame({
        "image_id": test_ids,
        "true_label": test_y,
        "predicted_prob": test_probs
    })

    test_results = compute_metrics(df_test["true_label"], df_test["predicted_prob"], operating_threshold=tuned_threshold)

    print("\n=== DUMMY EVALUATION PIPELINE VERIFICATION ===")
    print(f"Tuned 5% FPR Val Threshold: {tuned_threshold:.4f}")
    print(f"Validation AUC:             {val_results['auc']:.4f}")
    print(f"Unseen Test AUC:            {test_results['auc']:.4f} (95% CI: [{test_results['ci_lower']:.4f}, {test_results['ci_upper']:.4f}])")
    print(f"Unseen Test Macro-F1:       {test_results['macro_f1']:.4f}")
    print(f"Unseen Test Confusion Matrix:\n{test_results['confusion_matrix']}")

    generate_plots(test_results, prefix="dummy_")

if __name__ == "__main__":
    run_dummy_verification()