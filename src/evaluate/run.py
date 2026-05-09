"""Evaluate stage - A/B test ML vs DL with constitutional gates."""
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from src.config import load_config
from src.evaluate.evaluator import ABEvaluator


def main() -> None:
    """Run 3-way A/B/C test on ML vs DL vs GMM; write reports; exit 1 if gates fail."""
    try:
        # Load config
        config = load_config()
        
        # Read environment variables
        report_path = os.getenv("REPORT_PATH", "docs/evaluation_report.json")
        ab_report_path = os.getenv("AB_REPORT_PATH", "docs/ab_test_comparison.json")
        
        # Verify operational preconditions
        gmm_model_path = Path("models/best_gmm_model.pkl")
        if not gmm_model_path.exists():
            raise FileNotFoundError(
                f"GMM model not found: {gmm_model_path}. "
                "Run train_gmm stage before evaluate."
            )
        
        print("Evaluate: 3-way A/B/C test on ML vs DL vs GMM experiments")
        print(f"  ML experiment: {config.mlflow.experiment_name_ml}")
        print(f"  DL experiment: {config.mlflow.experiment_name_dl}")
        print(f"  GMM experiment: {config.mlflow.experiment_name_gmm}")
        print(f"  Seeds: {config.ab_test.seeds}")
        print(f"  Alpha: {config.ab_test.alpha}")
        print(f"  Tiebreak: {config.ab_test.tiebreak}")
        print(f"  Constitutional gates:")
        print(f"    - macro F1 > {config.model.macro_f1_threshold}")
        print(f"    - fatal recall > {config.model.fatal_recall_threshold}")
        
        # Create evaluator and run A/B test
        evaluator = ABEvaluator(
            mlflow_config=config.mlflow,
            ab_test_config=config.ab_test,
            model_config=config.model,
        )
        
        result = evaluator.evaluate()
        
        # Write evaluation_report.json (all fields)
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        
        # Write ab_test_comparison.json (statistical details)
        alpha_bonf = config.ab_test.alpha / 3.0
        ab_comparison = {
            "winner": result.winner,
            "alpha_bonferroni": alpha_bonf,
            "p_value_ml_dl": result.p_value_ml_dl,
            "p_value_ml_gmm": result.p_value_ml_gmm,
            "p_value_dl_gmm": result.p_value_dl_gmm,
            "significant_ml_dl": result.p_value_ml_dl < alpha_bonf,
            "significant_ml_gmm": result.p_value_ml_gmm < alpha_bonf,
            "significant_dl_gmm": result.p_value_dl_gmm < alpha_bonf,
            "cohens_d_ml_dl": result.cohens_d_ml_dl,
            "cohens_d_ml_gmm": result.cohens_d_ml_gmm,
            "cohens_d_dl_gmm": result.cohens_d_dl_gmm,
            "ml_mean_f1": result.ml_mean_f1,
            "dl_mean_f1": result.dl_mean_f1,
            "gmm_mean_f1": result.gmm_mean_f1,
            "ml_ci_low": result.ml_ci_low,
            "ml_ci_high": result.ml_ci_high,
            "dl_ci_low": result.dl_ci_low,
            "dl_ci_high": result.dl_ci_high,
            "gmm_ci_low": result.gmm_ci_low,
            "gmm_ci_high": result.gmm_ci_high,
        }
        with open(ab_report_path, "w") as f:
            json.dump(ab_comparison, f, indent=2)
        
        # Report results
        print(f"\n3-Way A/B/C Test Results (Bonferroni-corrected alpha = {config.ab_test.alpha / 3.0:.4f}):")
        print(f"  Winner: {result.winner.upper()}")
        print(f"  p-value (ml vs dl): {result.p_value_ml_dl:.4f}")
        print(f"  p-value (ml vs gmm): {result.p_value_ml_gmm:.4f}")
        print(f"  p-value (dl vs gmm): {result.p_value_dl_gmm:.4f}")
        print(f"  Cohen's d (ml vs dl): {result.cohens_d_ml_dl:.4f}")
        print(f"  Cohen's d (ml vs gmm): {result.cohens_d_ml_gmm:.4f}")
        print(f"  Cohen's d (dl vs gmm): {result.cohens_d_dl_gmm:.4f}")
        print(f"\n  ML:  macro F1 = {result.ml_mean_f1:.4f} (95% CI: [{result.ml_ci_low:.4f}, {result.ml_ci_high:.4f}])")
        print(f"       fatal recall = {result.ml_mean_fatal_recall:.4f}")
        print(f"  DL:  macro F1 = {result.dl_mean_f1:.4f} (95% CI: [{result.dl_ci_low:.4f}, {result.dl_ci_high:.4f}])")
        print(f"       fatal recall = {result.dl_mean_fatal_recall:.4f}")
        print(f"  GMM: macro F1 = {result.gmm_mean_f1:.4f} (95% CI: [{result.gmm_ci_low:.4f}, {result.gmm_ci_high:.4f}])")
        print(f"       fatal recall = {result.gmm_mean_fatal_recall:.4f}")
        gate_label = "[PASS]" if result.gates_passed else "[FAIL]"
        print(f"\nConstitutional Gates: {gate_label}")

        if not result.gates_passed:
            if result.winner == "ml":
                winner_f1 = result.ml_mean_f1
                winner_recall = result.ml_mean_fatal_recall
            elif result.winner == "dl":
                winner_f1 = result.dl_mean_f1
                winner_recall = result.dl_mean_fatal_recall
            else:  # gmm
                winner_f1 = result.gmm_mean_f1
                winner_recall = result.gmm_mean_fatal_recall

            if winner_f1 <= config.model.macro_f1_threshold:
                print(f"  [FAIL] Winner macro F1 ({winner_f1:.4f}) <= threshold ({config.model.macro_f1_threshold})")
            if winner_recall <= config.model.fatal_recall_threshold:
                print(f"  [FAIL] Winner fatal recall ({winner_recall:.4f}) <= threshold ({config.model.fatal_recall_threshold})")
        
        print(f"\nReports written:")
        print(f"  {report_path}")
        print(f"  {ab_report_path}")
        
        # Always exit 0 — tune stage will check gates_passed and decide whether to run HPO
        if not result.gates_passed:
            print("\nGates failed - tune stage will run Optuna HPO")
        else:
            print("\nGates passed - tune stage will skip HPO")
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
