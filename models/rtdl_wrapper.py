from typing import Any, Dict
import rtdl
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_tokenizer: rtdl.CategoricalFeatureTokenizer,
        mlp_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.cat_tokenizer = cat_tokenizer
        self.model = rtdl.MLP.make_baseline(
            d_in=n_num_features + cat_tokenizer.n_tokens * cat_tokenizer.d_token,
            **mlp_kwargs,
        )

    def forward(self, x_num, x_cat):
        return self.model(
            torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
        )
