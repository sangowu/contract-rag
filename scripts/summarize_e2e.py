# scripts/summarize_vanilla.py
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from loguru import logger
import pandas as pd
from src.utils.plot import plot_category_hits

VANILLA_E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_vanilla.csv"
HYBRID_E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_hybrid.csv"
REANKED_E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_reranked.csv"
REANKED_PARENT_CHILD_E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_reranked_parent_child.csv"

VANILLA_PLOT_LOC = "vanilla_e2e"
HYBRID_PLOT_LOC = "hybrid_e2e"
REANKED_PLOT_LOC = "reranked_e2e"
BATCH_REANKED_PLOT_LOC = "reranked_e2e_batch"
BATCH_REANKED_PARENT_CHILD_PLOT_LOC = "reranked_e2e_batch_parent_child"
TOP_N_CATEGORIES = 0

def summarize_metrics(df: pd.DataFrame) -> None:
    logger.success("=== Overall E2E Metrics (all answer types) ===")
    overall = {
        "hit@k": df["hit@k"].mean(),
        "rr@k": df["rr@k"].mean(),
        "recall@k": df["recall@k"].mean(),
        "f1_mean": df["f1"].mean(skipna=True),
        "em_mean": df["em"].mean(skipna=True),
        "acc_mean": df["acc"].mean(skipna=True),
    }
    for k, v in overall.items():
        if pd.isna(v):
            continue
        logger.info(f"{k:10s}: {v:.4f}")
    logger.info("="*50)

    logger.info("=== Metrics by answer_type ===")
    group = df.groupby("answer_type").agg(
        hit_k=("hit@k", "mean"),
        rr_k=("rr@k", "mean"),
        recall_k=("recall@k", "mean"),
        f1=("f1", "mean"),
        em=("em", "mean"),
        acc=("acc", "mean"),
        count=("answer_type", "count"),
    )

    logger.info(
        "answer_type | #samples | hit@k  | rr@k  | recall@k |   f1   |   em   |  acc"
    )
    logger.info("-" * 80)
    for atype, row in group.iterrows():
        print(
            f"{atype:11s} | "
            f"{int(row['count']):8d} | "
            f"{row['hit_k']:.4f} | "
            f"{row['rr_k']:.4f} | "
            f"{row['recall_k']:.4f} | "
            f"{(row['f1'] if pd.notna(row['f1']) else 0):6.4f} | "
            f"{(row['em'] if pd.notna(row['em']) else 0):6.4f} | "
            f"{(row['acc'] if pd.notna(row['acc']) else 0):5.4f}"
        )
    logger.info("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=REANKED_PARENT_CHILD_E2E_RESULTS_PATH,
        help="Path to e2e results csv file",
    )
    parser.add_argument(
        "--top_n_categories",
        type=int,
        default=TOP_N_CATEGORIES,
        help="If >0, only plot the lowest-N categories by hit@k",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    logger.info(f"Loaded e2e results from {args.csv}")
    summarize_metrics(df)
    plot_category_hits(
        df,
        loc=BATCH_REANKED_PARENT_CHILD_PLOT_LOC,
        max_categories=args.top_n_categories if args.top_n_categories > 0 else None,
    )

if __name__ == "__main__":
    main()
