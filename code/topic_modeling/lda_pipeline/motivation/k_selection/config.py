from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from config import PROJECT_ROOT


@dataclass
class KSweepConfig:
    k_values: list[int] = field(default_factory=lambda: [15, 20, 30, 40, 50])
    n_seeds: int = 5
    alpha: float = 0.05
    eta: float = 0.01
    min_df: int = 20
    max_df: float = 0.18
    min_bigram_count: int = 20
    iterations: int = 1500
    burn_in: int = 0
    top_n_words: int = 15
    heatmap_k_values: list[int] = field(default_factory=lambda: [30, 40])
    qualitative_k_values: list[int] = field(default_factory=lambda: [30, 40])
    top_split_examples: int = 8
    output_dir: Path = PROJECT_ROOT / "outputs" / "lda" / "k_sweep"
