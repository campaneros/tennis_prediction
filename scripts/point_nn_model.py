"""
Neural network architecture for point-level win probability.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn


def _embedding_dim(cardinality: int) -> int:
    """Rule-of-thumb embedding dimension."""
    return int(min(32, max(4, np.ceil(cardinality ** 0.5 + 1))))


class PointOutcomeNet(nn.Module):
    """
    Tabular model with categorical embeddings and gated residual blocks.
    """

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: Dict[str, int],
        hidden_sizes: Iterable[int] = (256, 128, 64),
        dropout: float = 0.15,
    ):
        super().__init__()

        self.cat_names = list(cat_cardinalities.keys())
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinality, _embedding_dim(cardinality))
                for name, cardinality in cat_cardinalities.items()
            }
        )

        emb_total = sum(self.embeddings[name].embedding_dim for name in self.cat_names)
        input_dim = num_numeric + emb_total

        layers = []
        prev = input_dim
        for hs in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev, hs),
                    nn.BatchNorm1d(hs),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hs
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        emb_out = []
        if self.cat_names:
            for idx, name in enumerate(self.cat_names):
                emb_layer = self.embeddings[name]
                emb_out.append(emb_layer(x_cat[:, idx]))
        if emb_out:
            cat_vec = torch.cat(emb_out, dim=1)
            x = torch.cat([x_num, cat_vec], dim=1)
        else:
            x = x_num
        x = self.backbone(x)
        return self.head(x).squeeze(-1)


def build_model(num_numeric: int, cat_maps: Dict[str, dict], dropout: float = 0.15) -> PointOutcomeNet:
    cardinalities = {name: len(mapping) for name, mapping in cat_maps.items()}
    return PointOutcomeNet(num_numeric=num_numeric, cat_cardinalities=cardinalities, dropout=dropout)
