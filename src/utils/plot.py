import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional
import os
save_path = Path('results/plots/')
save_path.parent.mkdir(parents=True, exist_ok=True)

def plot_hits(hits: pd.DataFrame, loc: str):
    plt.figure(figsize=(8, 5))
    counts = hits['hits'].value_counts().sort_index()
    labels = counts.index.astype(str)
    values = counts.values

    bars =  plt.bar(labels, values)
    plt.xlabel('Hits', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Hits Distribution - {loc}', fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(values) * 0.01, f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    img_path = Path(save_path) / loc / f"{loc}_hits.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    logger.info(f"Saved hits plot to {img_path}")

def plot_rrs(rrs: pd.DataFrame,  loc: str, top_n: int = 10):
    plt.figure(figsize=(8, 5))
    counts = rrs['rrs'].value_counts().sort_index(ascending=False)
    labels = [f'{idx:.2f}' if isinstance(idx, float) else idx for idx in counts.index]
    values = counts.values
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    bars = plt.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
    plt.xlabel('RRs Values', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'RRs Distribution - {loc}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
    max_val = values.max()
    plt.ylim(0, max_val * 1.2)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(values) * 0.01, f'{float(height):.2f}', ha='center', va='bottom')
        
    plt.tight_layout()
    img_path = Path(save_path) / loc / f"{loc}_rrs.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    logger.info(f"Saved rrs plot to {img_path}")    

def plot_recalls(recalls: pd.DataFrame,  loc: str, top_n: int = 10):
    plt.figure(figsize=(8, 5))
    counts = recalls['recalls'].value_counts().sort_index(ascending=False)
    labels = [f'{idx:.2f}' if isinstance(idx, float) else idx for idx in counts.index]
    values = counts.values
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    bars = plt.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
    plt.xlabel('Recalls Values', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Recalls Distribution - {loc}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=10)
    max_val = values.max()
    plt.ylim(0, max_val * 1.2)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + max(values) * 0.01, f'{float(height):.2f}', ha='center', va='bottom')
        
    plt.tight_layout()
    img_path = Path(save_path) / loc / f"{loc}_recalls.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    logger.info(f"Saved recalls plot to {img_path}")    

def plot_category_hits(
    df: pd.DataFrame,
    loc: str,
    top_n: Optional[int] = None,
) -> None:

    grouped = (
        df.groupby("category")
        .agg(hit_k=("hit@k", "mean"), count=("category", "count"))
        .sort_values("hit_k")
    )

    if top_n is not None and top_n > 0:
        grouped = grouped.head(top_n)

    plt.figure(figsize=(10, max(4, len(grouped) * 0.3)))
    plt.barh(grouped.index, grouped["hit_k"])
    plt.xlabel("hit@k")
    plt.title(f"E2E hit@k by category - {loc}")
    plt.tight_layout()
    img_path = Path(save_path) / loc / f"{loc}_hit_at_k_by_category.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)
    plt.close()
    logger.success(f"Saved hit@k by category plot to {img_path}")


if __name__ == "__main__":
    hits = pd.DataFrame([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], columns=['hits'])
    rrs = pd.DataFrame([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], columns=['rrs'])
    plot_hits(hits, "test")
    plot_rrs(rrs, "test")
    logger.info("Done!")
