# scripts/analyze_errors.py
import argparse
import os
import math
import pandas as pd
from loguru import logger
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
VANILLA_INPUT_CSV = "/root/autodl-tmp/results/csv/cuad_v1_e2e_vanilla.csv"
VANILLA_OUTPUT_MD = "/root/autodl-tmp/results/md/vanilla_e2e/vanilla_error_cases.md"
HYBRID_INPUT_CSV = "/root/autodl-tmp/results/csv/cuad_v1_e2e_hybrid.csv"
HYBRID_OUTPUT_MD = "/root/autodl-tmp/results/md/hybrid_e2e/hybrid_error_cases.md"
REANKER_INPUT_CSV = "/root/autodl-tmp/results/csv/cuad_v1_e2e_reranked.csv"
REANKER_PARENT_CHILD_INPUT_CSV = "/root/autodl-tmp/results/csv/cuad_v1_e2e_reranked_parent_child.csv"
REANKER_OUTPUT_MD = "/root/autodl-tmp/results/md/reranked_e2e/reranked_error_cases.md"
BATCH_REANKER_OUTPUT_MD = "/root/autodl-tmp/results/md/reranked_e2e_batch/reranked_error_cases_batch.md"
BATCH_REANKER_PARENT_CHILD_OUTPUT_MD = "/root/autodl-tmp/results/md/reranked_e2e_batch_parent_child/reranked_error_cases_batch_parent_child.md"
NUM_CATEGORIES = 5
EXAMPLES_PER_CATEGORY = 5

def sanitize(text):
    if isinstance(text, float) and math.isnan(text):
        return ""
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\n", " ").replace("|", "\\|")
    return " ".join(s.split())


def generate_error_md(
    df: pd.DataFrame,
    num_categories: int,
    examples_per_category: int,
    out_md: str,
):
    grouped = df.groupby("category").agg(
        hit_k=("hit@k", "mean"),
        rr_k=("rr@k", "mean"),
        n=("category", "count"),
    )

    worst = grouped.sort_values("hit_k").head(num_categories)

    lines = []
    lines.append("# Vanilla RAG Error Cases\n")
    lines.append("Auto-generated examples from CUAD v1 E2E evaluation.\n")

    for cat, row in worst.iterrows():
        lines.append(f"## Category: {cat}")
        lines.append("")
        lines.append(f"- #samples: {int(row['n'])}")
        lines.append(f"- hit@k (mean): {row['hit_k']:.4f}")
        lines.append(f"- MRR@k (mean): {row['rr_k']:.4f}")
        lines.append("")

        df_cat = df[df["category"] == cat]

        errors = df_cat[(df_cat["hit@k"] == 0) | (df_cat["rr@k"] == 0)]
        if errors.empty:
            errors = df_cat.sort_values("rr@k").head(examples_per_category * 2)
        if len(errors) > examples_per_category:
            errors = errors.sample(n=examples_per_category, random_state=42)

        for i, row_ex in enumerate(errors.to_dict("records"), start=1):
            lines.append(f"### Example {i}")
            lines.append("")
            lines.append(f"- **query**: {sanitize(row_ex.get('query'))}")
            lines.append(f"- **gold_answer**: {sanitize(row_ex.get('gold_answer'))}")
            lines.append(f"- **model_answer**: {sanitize(row_ex.get('model_answer'))}")
            lines.append(
                f"- **hit@k**: {row_ex.get('hit@k'):.4f}, "
                f"**rr@k**: {row_ex.get('rr@k'):.4f}, "
                f"**recall@k**: {row_ex.get('recall@k'):.4f}"
            )

            f1 = row_ex.get("f1")
            em = row_ex.get("em")
            acc = row_ex.get("acc")
            extra = []
            if not (isinstance(f1, float) and math.isnan(f1)):
                extra.append(f"f1={f1:.4f}")
            if not (isinstance(em, float) and math.isnan(em)):
                extra.append(f"em={em:.4f}")
            if not (isinstance(acc, float) and math.isnan(acc)):
                extra.append(f"acc={acc:.4f}")
            if extra:
                lines.append(f"- **extra**: " + ", ".join(extra))

            lines.append("")

        lines.append("---")
        lines.append("")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[saved] error cases -> {out_md}")
    print("\nWorst categories by hit@k:")
    print(worst[["hit_k", "rr_k", "n"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=REANKER_PARENT_CHILD_INPUT_CSV,
        help="Path to cuad_v1_e2e_reranked_parent_child.csv",
    )
    parser.add_argument(
        "--out_md",
        type=str,
        default=BATCH_REANKER_PARENT_CHILD_OUTPUT_MD,
        help="Where to save the markdown error report.",
    )
    parser.add_argument(
        "--num_categories",
        type=int,
        default=NUM_CATEGORIES,
        help="How many worst categories to inspect.",
    )
    parser.add_argument(
        "--examples_per_category",
        type=int,
        default=EXAMPLES_PER_CATEGORY,
        help="How many error examples to show per category.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    logger.info(f"Loaded e2e results from {args.csv}")
    generate_error_md(
        df=df,
        num_categories=args.num_categories,
        examples_per_category=args.examples_per_category,
        out_md=args.out_md,
    )


if __name__ == "__main__":
    main()
