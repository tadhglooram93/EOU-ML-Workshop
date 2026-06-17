"""
Find the best feature set for a linear regression irrigation model using PyCaret.

Run from the repo root:

    python scripts/find_best_features_pycaret.py

What it does:
    1. Loads data/model_dataset.csv and data/model_metadata.json.
    2. Runs an exhaustive best-subset search: it scores a linear regression
       on every possible combination of the candidate features using
       cross-validated R2 / RMSE, then keeps the combination that wins.
    3. Re-fits a PyCaret regression experiment on the winning feature set so
       you get a full, reproducible PyCaret model + tuning + saved artifact.

Outputs:
    data/feature_search_results.csv   ranked list of every feature combo tried
    models/best_irrigation_lr.pkl     finalized PyCaret linear-regression model

Notes:
    - With ~17 candidate features an exhaustive search is 2^17 - 1 = 131,071
      combinations. That is tractable for plain linear regression but can be
      slow. Use --max-features to cap the subset size and speed things up.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score


DATA_DIR = Path("data")
MODELS_DIR = Path("models")

DATASET_PATH = DATA_DIR / "model_dataset.csv"
METADATA_PATH = DATA_DIR / "model_metadata.json"

RESULTS_PATH = DATA_DIR / "feature_search_results.csv"
MODEL_PATH = MODELS_DIR / "best_irrigation_lr"  # PyCaret appends .pkl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help=(
            "Maximum number of features allowed in a combination. "
            "Defaults to all candidate features (full exhaustive search)."
        ),
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help="Minimum number of features allowed in a combination. Default: 1.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for the search. Default: 5.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many of the best combinations to print to the console.",
    )
    parser.add_argument(
        "--skip-pycaret",
        action="store_true",
        help="Only run the combinatorial search; skip the final PyCaret fit.",
    )
    return parser.parse_args()


def load_dataset() -> tuple[pd.DataFrame, str, list[str]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATASET_PATH}. "
            "Run scripts/build_model_dataset.py first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {METADATA_PATH}.")

    df = pd.read_csv(DATASET_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    target_col = metadata["target_col"]
    feature_cols = metadata["feature_cols"]

    # Keep only features that actually exist and are numeric.
    feature_cols = [c for c in feature_cols if c in df.columns]
    feature_cols = (
        df[feature_cols].select_dtypes(include="number").columns.tolist()
    )

    model_df = df[[target_col] + feature_cols].dropna().reset_index(drop=True)
    return model_df, target_col, feature_cols


def exhaustive_feature_search(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    min_features: int,
    max_features: int,
    folds: int,
) -> pd.DataFrame:
    """
    Score a linear regression on every feature combination using
    cross-validated R2 and RMSE. Returns a ranked results DataFrame.
    """

    X_all = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    feature_index = {name: i for i, name in enumerate(feature_cols)}

    cv = KFold(n_splits=folds, shuffle=True, random_state=42)

    total_combos = sum(
        _n_choose_k(len(feature_cols), k)
        for k in range(min_features, max_features + 1)
    )
    print(
        f"Searching {total_combos:,} feature combinations "
        f"(sizes {min_features}-{max_features}, {folds}-fold CV)...\n"
    )

    records: list[dict] = []
    evaluated = 0

    for k in range(min_features, max_features + 1):
        for combo in combinations(feature_cols, k):
            cols = [feature_index[name] for name in combo]
            X = X_all[:, cols]

            model = LinearRegression()
            r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            rmse_scores = -cross_val_score(
                model, X, y, cv=cv, scoring="neg_root_mean_squared_error"
            )

            records.append(
                {
                    "n_features": k,
                    "features": ", ".join(combo),
                    "cv_r2_mean": r2_scores.mean(),
                    "cv_r2_std": r2_scores.std(),
                    "cv_rmse_mean": rmse_scores.mean(),
                    "cv_rmse_std": rmse_scores.std(),
                }
            )

            evaluated += 1
            if evaluated % 5000 == 0:
                print(f"  ...evaluated {evaluated:,} / {total_combos:,}")

    results = pd.DataFrame.from_records(records)
    results = results.sort_values(
        ["cv_r2_mean", "cv_rmse_mean"], ascending=[False, True]
    ).reset_index(drop=True)
    return results


def _n_choose_k(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


def run_pycaret_on_best(
    df: pd.DataFrame,
    target_col: str,
    best_features: list[str],
) -> None:
    """
    Re-run a full PyCaret regression experiment on the winning feature set.
    """

    try:
        from pycaret.regression import (
            setup,
            create_model,
            tune_model,
            finalize_model,
            pull,
            save_model,
        )
    except ImportError:
        print(
            "\nPyCaret is not installed, so the final modeling step was skipped.\n"
            "Install it with:  pip install pycaret\n"
            "The exhaustive feature-search results were still saved."
        )
        return

    print("\n" + "=" * 70)
    print("Running PyCaret linear regression on the best feature set")
    print("=" * 70)

    model_df = df[[target_col] + best_features].copy()

    setup(
        data=model_df,
        target=target_col,
        session_id=42,
        normalize=True,
        fold=5,
        verbose=False,
    )

    lr = create_model("lr", verbose=False)
    print("\nBaseline linear regression (cross-validated):")
    print(pull())

    tuned_lr = tune_model(lr, verbose=False)
    print("\nTuned linear regression (cross-validated):")
    print(pull())

    final_lr = finalize_model(tuned_lr)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(final_lr, str(MODEL_PATH))
    print(f"\nSaved finalized model to: {MODEL_PATH}.pkl")


def main() -> None:
    args = parse_args()

    df, target_col, feature_cols = load_dataset()

    n_features = len(feature_cols)
    max_features = args.max_features or n_features
    max_features = min(max_features, n_features)
    min_features = max(1, args.min_features)

    print(f"Target column : {target_col}")
    print(f"Candidate features ({n_features}): {', '.join(feature_cols)}")
    print(f"Rows          : {len(df)}\n")

    results = exhaustive_feature_search(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols,
        min_features=min_features,
        max_features=max_features,
        folds=args.folds,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved full ranked results to: {RESULTS_PATH}")

    print(f"\nTop {args.top} feature combinations by cross-validated R2:")
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(
            results.head(args.top)[
                ["n_features", "cv_r2_mean", "cv_rmse_mean", "features"]
            ].to_string(index=False)
        )

    best_row = results.iloc[0]
    best_features = [f.strip() for f in best_row["features"].split(",")]

    print("\n" + "-" * 70)
    print("Best feature set found:")
    print(f"  features   : {', '.join(best_features)}")
    print(f"  cv R2 mean : {best_row['cv_r2_mean']:.4f}")
    print(f"  cv RMSE    : {best_row['cv_rmse_mean']:.4f}")
    print("-" * 70)

    if not args.skip_pycaret:
        run_pycaret_on_best(df, target_col, best_features)


if __name__ == "__main__":
    main()
