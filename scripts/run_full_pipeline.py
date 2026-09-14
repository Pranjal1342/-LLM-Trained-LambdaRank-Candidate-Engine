#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import argparse


_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)

def check_artifacts_up_to_date(precomputed_dir: str, candidates_path: str) -> bool:
    """Check if precomputed artifacts exist and are newer than candidates.jsonl."""
    required_files = [
        "bm25_index.pkl",
        "candidate_ids.pkl",
        "lgbm_model.pkl",
        "static_features.pkl",
        "vocab.pkl",
        "bm25_matrix.npz",
        "candidate_offsets.pkl",
        "lgbm_model.txt"
    ]
    for f in required_files:
        fpath = os.path.join(precomputed_dir, f)
        if not os.path.isfile(fpath):
            return False
        
        #  mtime vs candidates.jsonl
        if os.path.isfile(candidates_path):
            if os.path.getmtime(fpath) < os.path.getmtime(candidates_path):
                return False
    return True

def run_step(command_list, step_label, step_num):
    print(f"\n[{step_num}/3] Running {step_label}...")
    t0 = time.time()
    
    # process
    result = subprocess.run(command_list, capture_output=True, text=True)
    
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num}/3 ({step_label}) FAILED (Exit Code: {result.returncode})")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        sys.exit(result.returncode)
    
    print(result.stdout.strip())
    print(f"[{step_num}/3] {step_label.capitalize()} complete ({elapsed:.2f}s)")
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="Ranking Pipeline Runner")
    parser.add_argument("--candidates", default="./candidates.jsonl", help="Path to candidates JSONL")
    parser.add_argument("--out", default="./CTRL_COFFEE_REPEAT.csv", help="Path to output CSV")
    parser.add_argument("--force-precompute", action="store_true", help="Force rebuild precompute artifacts")
    args = parser.parse_args()

    candidates_path = os.path.abspath(args.candidates)
    out_path = os.path.abspath(args.out)
    precomputed_dir = os.path.join(_PROJECT_ROOT, "precomputed")
    
    t_start = time.time()
    
    artifacts_ready = check_artifacts_up_to_date(precomputed_dir, candidates_path)
    
    python_exe = sys.executable

    t_precompute = 0.0
    if not artifacts_ready or args.force_precompute:
        cmd = [python_exe, "scripts/precompute.py", "--candidates", candidates_path, "--base-dir", _PROJECT_ROOT]
        t_precompute = run_step(cmd, "precompute", 1)
    else:
        print("\n[1/3] Precompute skipped (artifacts up to date)")

    # rank
    cmd = [python_exe, "src/rank.py", "--candidates", candidates_path, "--out", out_path, "--base-dir", _PROJECT_ROOT]
    t_rank = run_step(cmd, "rank", 2)

    # validate
    cmd = [python_exe, "scripts/validate_submission.py", "--submission", out_path]
    t_validate = run_step(cmd, "validate_submission", 3)

    total_wall = time.time() - t_start
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Total Clock Time: {total_wall:.2f} seconds")
    print(f"  Step 1 (Precompute):   {t_precompute:.2f}s" if t_precompute > 0 else "  Step 1 (Precompute):   Skipped (up to date)")
    print(f"  Step 2 (Ranking):      {t_rank:.2f}s")
    print(f"  Step 3 (Validation):   {t_validate:.2f}s")
    
    
    if os.path.isfile(out_path):
        try:
            import pandas as pd
            df = pd.read_csv(out_path)
            if len(df) == 100:
                print("  CONFIRMED:             submission.csv exists with exactly 100 rows.")
            else:
                print(f"  [ERROR]                submission.csv has {len(df)} rows, expected exactly 100!")
                sys.exit(1)
        except Exception as e:
            print(f"  [ERROR]                Error reading CSV: {e}")
            sys.exit(1)
    else:
        print("  [ERROR]                Missing output file submission.csv!")
        sys.exit(1)
    
    log_dir = os.path.join(_PROJECT_ROOT, "logs")
    if os.path.isdir(log_dir):
        logs = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("rank_")]
        if logs:
            latest_log = max(logs, key=os.path.getmtime)
            print(f"  Latest Log File:       {latest_log}")
            
    print("=" * 60)

if __name__ == "__main__":
    main()
