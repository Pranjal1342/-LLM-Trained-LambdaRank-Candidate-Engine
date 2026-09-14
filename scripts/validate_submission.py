from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd


def validate_submission(submission_path: str) -> bool:
    """
    Run all format validation checks on submission.csv.

    Returns True if all checks pass, False if any fail.
    Prints detailed output for each check.
    """
    errors = []
    warnings = []

    print("=" * 60)
    print("SUBMISSION VALIDATOR")
    print(f"File: {submission_path}")
    print("=" * 60)

    #  file existence
    if not os.path.isfile(submission_path):
        print(f"\n[FAIL] File not found: {submission_path}")
        return False

    try:
        df = pd.read_csv(submission_path, dtype={"candidate_id": str, "reasoning": str})
    except Exception as e:
        print(f"\n[FAIL] Cannot parse CSV: {e}")
        return False

    print(f"\nParsed: {len(df)} rows × {len(df.columns)} columns")


    required_cols = ["candidate_id", "rank", "score", "reasoning"]
    if list(df.columns) != required_cols:
        missing = set(required_cols) - set(df.columns)
        extra = set(df.columns) - set(required_cols)
        wrong_order = set(df.columns) == set(required_cols) and list(df.columns) != required_cols

        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        if extra:
            errors.append(f"Extra columns (not allowed): {sorted(extra)}")
        if wrong_order:
            errors.append(
                f"Column order wrong. Expected: {required_cols}, "
                f"Got: {list(df.columns)}"
            )

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return False


    if len(df) != 100:
        errors.append(f"Expected exactly 100 rows, got {len(df)}")

    try:
        ranks = df["rank"].tolist()
        rank_set = set(int(r) for r in ranks)
        if rank_set != set(range(1, 101)):
            missing_ranks = set(range(1, 101)) - rank_set
            extra_ranks = rank_set - set(range(1, 101))
            if missing_ranks:
                errors.append(f"Missing ranks: {sorted(missing_ranks)[:10]}")
            if extra_ranks:
                errors.append(f"Invalid ranks (out of 1–100): {sorted(extra_ranks)[:10]}")
        if len(ranks) != len(set(ranks)):
            errors.append("Duplicate ranks found")
    except (TypeError, ValueError) as e:
        errors.append(f"Rank column contains non-integer values: {e}")

    try:
        scores = pd.to_numeric(df["score"], errors="raise")
        if scores.isna().any():
            errors.append("Score column contains NaN values")
        else:
            if scores.min() < 0:
                errors.append(f"Score below 0: min={scores.min():.6f}")
            if scores.max() > 1.0001:
                errors.append(f"Score above 1: max={scores.max():.6f}")
    except ValueError as e:
        errors.append(f"Score column contains non-numeric values: {e}")

    try:
        df_sorted = df.copy()
        df_sorted["rank_int"] = pd.to_numeric(df_sorted["rank"], errors="coerce")
        df_sorted = df_sorted.sort_values("rank_int")
        score_vals = pd.to_numeric(df_sorted["score"], errors="coerce").values

        violations = []
        for i in range(1, len(score_vals)):
            if score_vals[i] > score_vals[i - 1] + 1e-9:
                violations.append(
                    f"rank {i} → {i+1}: {score_vals[i-1]:.6f} → {score_vals[i]:.6f}"
                )

        if violations:
            errors.append(
                f"Monotonicity violated at {len(violations)} positions: "
                f"{violations[:3]}"
            )
    except Exception as e:
        errors.append(f"Could not check monotonicity: {e}")

    if df["candidate_id"].isna().any():
        errors.append("candidate_id column contains NaN values")
    else:
        if df["candidate_id"].duplicated().any():
            dups = df[df["candidate_id"].duplicated()]["candidate_id"].tolist()
            errors.append(f"Duplicate candidate_ids: {dups[:5]}")

     
        bad_format = [
            cid for cid in df["candidate_id"]
            if not re.match(r'^(CAND_\d{7}|SYNTH_[A-Z_]+)$', str(cid))
        ]
        if bad_format:
            warnings.append(
                f"{len(bad_format)} candidate_ids don't match CAND_XXXXXXX format: "
                f"{bad_format[:3]}"
            )

    if df["reasoning"].isna().any():
        errors.append(f"{df['reasoning'].isna().sum()} reasoning fields are null")

    empty_reasoning = df["reasoning"].fillna("").str.strip() == ""
    if empty_reasoning.any():
        errors.append(f"{empty_reasoning.sum()} reasoning fields are empty")

    # check reasonable length (warn if very short)
    short_reasoning = df["reasoning"].fillna("").str.len() < 20
    if short_reasoning.any():
        warnings.append(
            f"{short_reasoning.sum()} reasoning fields are very short (<20 chars)"
        )

    stripped = df["candidate_id"].str.strip()
    if (stripped != df["candidate_id"]).any():
        errors.append("Some candidate_ids have leading/trailing whitespace")


    print()
    if errors:
        print(f"RESULT: FAIL ({len(errors)} error(s), {len(warnings)} warning(s))\n")
        for e in errors:
            print(f"  [FAIL] {e}")
        for w in warnings:
            print(f"  [WARN] {w}")
        return False
    else:
        print(f"RESULT: PASS (0 errors, {len(warnings)} warning(s))\n")

        df_sorted = df.sort_values("rank")
        scores = pd.to_numeric(df_sorted["score"])
        print(f"  Rows: {len(df)}")
        print(f"  Ranks: 1–{int(df['rank'].max())}")
        print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
        print(f"  Avg reasoning length: {df['reasoning'].str.len().mean():.0f} chars")
        print(f"  Distinct candidate_ids: {df['candidate_id'].nunique()}")

        for w in warnings:
            print(f"\n  [WARN] {w}")

        print("\nSAFE TO SUBMIT [PASS]")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate submission.csv against the spec checklist"
    )
    parser.add_argument(
        "--submission",
        default="./ranked_candidates.csv",
        help="Path to ranked_candidates.csv to validate",
    )
    args = parser.parse_args()

    passed = validate_submission(os.path.abspath(args.submission))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
