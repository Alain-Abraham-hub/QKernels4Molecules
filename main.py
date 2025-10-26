# ===============================================================
# SAFE HYBRID KERNEL OPTIMIZATION
# Quantum-Classical Hybrid SVM with Conservative Boosting
# ===============================================================
# This script:
#  1. Loads hybrid feature datasets (quantum + classical)
#  2. Performs parameter search (α, C, class_weight, multi-kernel)
#  3. Ensures no performance degradation ("safe boost")
#  4. Saves JSON reports and a final leaderboard summary
# ===============================================================

import numpy as np
import json
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib


def safe_hybrid_boost(npz_path, dataset_name, verbose=True):
    """
    Perform conservative (safe) performance optimization for a hybrid
    quantum-classical kernel SVM classifier.

    Parameters
    ----------
    npz_path : str
        Path to the .npz dataset containing X_quantum, X_classical, y.
    dataset_name : str
        Name of the dataset (for logging and saving).
    verbose : bool, optional
        Whether to print detailed output (default=True).

    Returns
    -------
    dict
        Results dictionary with metrics, best parameters, and improvements.
    """

    if not verbose:
        import warnings, sys
        warnings.filterwarnings('ignore')
        sys.stdout = open(os.devnull, 'w')

    print("=" * 80)
    print(f"SAFE HYBRID BOOST: {dataset_name}")
    print("=" * 80)

    # -----------------------------------------------------------
    # 1. Load and preprocess data
    # -----------------------------------------------------------
    data = np.load(npz_path)
    X_q = np.nan_to_num(data['X_quantum'])
    X_c = np.nan_to_num(data['X_classical'])
    y = np.nan_to_num(data['y']).astype(int)

    # Standardize features
    X_q = StandardScaler().fit_transform(X_q)
    X_c = StandardScaler().fit_transform(X_c)

    n_samples = len(y)
    print(f"Dataset size: {n_samples} samples")

    # Reproducible 10-fold CV
    np.random.seed(42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # -----------------------------------------------------------
    # 2. Cache kernel matrices for efficiency
    # -----------------------------------------------------------
    gamma_q = 1.0 / (X_q.shape[1] * np.var(X_q) + 1e-6)
    gamma_c = 1.0 / (X_c.shape[1] * np.var(X_c) + 1e-6)
    K_q_rbf = rbf_kernel(X_q, X_q, gamma=gamma_q)
    K_c_rbf = rbf_kernel(X_c, X_c, gamma=gamma_c)

    # -----------------------------------------------------------
    # 3. Baseline computation (coarse α search)
    # -----------------------------------------------------------
    print("\nComputing baseline...")

    alphas_coarse = np.arange(0.0, 1.1, 0.1)
    best_alpha_coarse, best_score_coarse = 0.5, 0.0

    for alpha in alphas_coarse:
        K = alpha * K_q_rbf + (1 - alpha) * K_c_rbf
        clf = SVC(kernel='precomputed', C=10)
        score = cross_val_score(clf, K, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()

        if score > best_score_coarse:
            best_score_coarse, best_alpha_coarse = score, alpha

    baseline = best_score_coarse
    print(f"Baseline accuracy: {baseline:.4f} (α={best_alpha_coarse:.1f})")

    # -----------------------------------------------------------
    # 4. Improvement 1: Fine α search around best coarse α
    # -----------------------------------------------------------
    print("\nFine-grained alpha optimization...")
    alpha_min = max(0.0, best_alpha_coarse - 0.15)
    alpha_max = min(1.0, best_alpha_coarse + 0.15)
    alphas_fine = np.linspace(alpha_min, alpha_max, 31)

    best_alpha, best_score = best_alpha_coarse, baseline

    for alpha in alphas_fine:
        K = alpha * K_q_rbf + (1 - alpha) * K_c_rbf
        clf = SVC(kernel='precomputed', C=10)
        score = cross_val_score(clf, K, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()
        if score > best_score:
            best_score, best_alpha = score, alpha

    print(f"Best alpha: {best_alpha:.3f}, accuracy: {best_score:.4f}")

    # -----------------------------------------------------------
    # 5. Improvement 2: C optimization
    # -----------------------------------------------------------
    print("\nC parameter optimization...")

    if n_samples < 300:
        C_values = [0.1, 0.5, 1, 5, 10, 50]
    else:
        C_values = [0.01, 0.1, 1, 10, 50, 100]

    best_C, best_score_C = 10, best_score
    K_best = best_alpha * K_q_rbf + (1 - best_alpha) * K_c_rbf

    for C in C_values:
        clf = SVC(kernel='precomputed', C=C)
        score = cross_val_score(clf, K_best, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()
        if score > best_score_C:
            best_score_C, best_C = score, C

    print(f"Best C: {best_C}, accuracy: {best_score_C:.4f}")

    # -----------------------------------------------------------
    # 6. Improvement 3: Class weight balancing (if needed)
    # -----------------------------------------------------------
    print("\nChecking class imbalance...")
    class_counts = np.bincount(y)
    imbalance_ratio = max(class_counts) / min(class_counts)
    print(f"Class distribution: {class_counts}, Imbalance ratio: {imbalance_ratio:.2f}")

    best_score_weight = best_score_C
    best_weight = None

    if imbalance_ratio > 1.5:
        clf_balanced = SVC(kernel='precomputed', C=best_C, class_weight='balanced')
        score_balanced = cross_val_score(
            clf_balanced, K_best, y, cv=cv, scoring='accuracy', n_jobs=-1
        ).mean()

        if score_balanced > best_score_weight:
            best_score_weight, best_weight = score_balanced, 'balanced'
            print(f"Balanced weighting improves accuracy: {score_balanced:.4f}")
        else:
            print(f"Balanced weighting reduces accuracy: {score_balanced:.4f}")
    else:
        print("Classes are balanced; skipping weight adjustment.")

    # -----------------------------------------------------------
    # 7. Improvement 4: Optional polynomial + RBF multi-kernel
    # -----------------------------------------------------------
    print("\nTesting multi-kernel (RBF + Polynomial)...")
    best_beta, best_score_multi = 1.0, best_score_weight

    if n_samples >= 500:
        K_c_poly = polynomial_kernel(X_c, X_c, degree=2, gamma=gamma_c, coef0=1)
        K_c_poly = K_c_poly / np.max(K_c_poly)

        for beta in [1.0, 0.9, 0.8, 0.7]:
            K_c_combined = beta * K_c_rbf + (1 - beta) * K_c_poly
            K_multi = best_alpha * K_q_rbf + (1 - best_alpha) * K_c_combined
            clf = SVC(kernel='precomputed', C=best_C, class_weight=best_weight)
            score = cross_val_score(clf, K_multi, y, cv=cv, scoring='accuracy', n_jobs=-1).mean()
            if score > best_score_multi:
                best_score_multi, best_beta = score, beta
        print(f"Best multi-kernel beta: {best_beta}, accuracy: {best_score_multi:.4f}")
    else:
        print(f"Dataset too small ({n_samples} samples), skipping multi-kernel test.")

    # -----------------------------------------------------------
    # 8. Final evaluation (10-fold CV)
    # -----------------------------------------------------------
    print("\nFinal evaluation (10-fold CV)...")

    if n_samples >= 500 and best_beta < 1.0:
        K_c_poly_final = polynomial_kernel(X_c, X_c, degree=2, gamma=gamma_c, coef0=1)
        K_c_poly_final = K_c_poly_final / np.max(K_c_poly_final)
        K_c_final = best_beta * K_c_rbf + (1 - best_beta) * K_c_poly_final
    else:
        K_c_final = K_c_rbf

    K_final = best_alpha * K_q_rbf + (1 - best_alpha) * K_c_final

    clf_final = SVC(kernel='precomputed', C=best_C, class_weight=best_weight)
    acc_scores = cross_val_score(clf_final, K_final, y, cv=cv, scoring='accuracy', n_jobs=-1)
    f1_scores = cross_val_score(clf_final, K_final, y, cv=cv, scoring='f1_macro', n_jobs=-1)

    final_acc = acc_scores.mean()
    final_f1 = f1_scores.mean()
    final_acc_std = acc_scores.std()
    total_gain = (final_acc - baseline) * 100

    print(f"Baseline accuracy: {baseline:.4f}")
    print(f"Final accuracy:    {final_acc:.4f} ± {final_acc_std:.4f}")
    print(f"Final F1-score:    {final_f1:.4f}")
    print(f"Total improvement: {total_gain:+.2f}%")

    # Safety rollback
    if total_gain < -0.5:
        print("Performance decreased. Reverting to baseline parameters.")
        final_acc, best_alpha, best_C, best_weight = baseline, best_alpha_coarse, 10, None

    # -----------------------------------------------------------
    # 9. Save results and best model
    # -----------------------------------------------------------
    output_dir = Path('results_safe_boost')
    output_dir.mkdir(exist_ok=True)

    results = {
        'dataset': dataset_name,
        'baseline': float(baseline),
        'final_accuracy': float(final_acc),
        'final_accuracy_std': float(final_acc_std),
        'final_f1': float(final_f1),
        'improvement_percent': float(total_gain),
        'best_params': {
            'alpha': float(best_alpha),
            'C': float(best_C),
            'class_weight': best_weight,
            'multi_kernel_beta': float(best_beta) if best_beta < 1.0 else None
        }
    }

    with open(output_dir / f'{dataset_name.lower()}_safe_boost.json', 'w') as f:
        json.dump(results, f, indent=2)

    joblib.dump(clf_final, output_dir / f'{dataset_name.lower()}_final_model.joblib')

    print("Results saved successfully.\n")
    return results


def run_safe_boost_all():
    """Run the safe hybrid boost on all available datasets."""

    datasets = {
        'PROTEINS': 'proteins_hybrid_features.npz',
        'MUTAG': 'mutag_hybrid_features.npz',
        'AIDS': 'aids_hybrid_features.npz',
        'NCI1': 'nci1_hybrid_features.npz',
        'PTC_MR': 'ptc_mr_hybrid_features.npz'
    }

    all_results = {}

    print("=" * 80)
    print("SAFE HYBRID BOOST - MULTI DATASET EXECUTION")
    print("=" * 80)

    for dataset_name, npz_file in tqdm(datasets.items(), desc="Processing Datasets"):
        if Path(npz_file).exists():
            results = safe_hybrid_boost(npz_file, dataset_name, verbose=True)
            all_results[dataset_name] = results
        else:
            print(f"Skipping {dataset_name}: file '{npz_file}' not found.")

    # Leaderboard summary
    print("\nSUMMARY RESULTS")
    print("-" * 70)
    print(f"{'Dataset':<12} {'Baseline':>10} {'Final':>10} {'Gain (%)':>10}")
    print("-" * 70)
    for ds, res in all_results.items():
        print(f"{ds:<12} {res['baseline']:>10.4f} {res['final_accuracy']:>10.4f} {res['improvement_percent']:>10.2f}")
    print("-" * 70)

    # Save combined results
    output_dir = Path('results_safe_boost')
    with open(output_dir / 'all_safe_boost_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("All results saved in results_safe_boost/ directory.")
    return all_results


if __name__ == "__main__":
    run_safe_boost_all()
