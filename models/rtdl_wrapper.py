from typing import Any, Dict
import rtdl
import torch
import torch.nn as nn


# class Model(nn.Module):
#     def __init__(
#         self,
#         n_num_features: int,
#         cat_tokenizer: rtdl.CategoricalFeatureTokenizer,
#         mlp_kwargs: Dict[str, Any],
#     ):
#         super().__init__()
#         self.cat_tokenizer = cat_tokenizer
#         self.model = rtdl.MLP.make_baseline(
#             d_in=n_num_features + cat_tokenizer.n_tokens * cat_tokenizer.d_token,
#             **mlp_kwargs,
#         )
#
#     def forward(self, x_num, x_cat):
#         return self.model(
#             torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
#         )
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_features, num_features)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class Model(nn.Module):
    def __init__(self, n_num_features, cat_tokenizer, mlp_kwargs):
        super().__init__()
        self.cat_tokenizer = cat_tokenizer
        d_layers = mlp_kwargs.get('d_layers', [256, 256, 256])
        dropout = mlp_kwargs.get('dropout', 0.1)
        d_out = mlp_kwargs.get('d_out', 1)

        self.fc1 = nn.Linear(n_num_features + cat_tokenizer.n_tokens * cat_tokenizer.d_token, d_layers[0])
        self.res_blocks = nn.Sequential(*[ResidualBlock(size) for size in d_layers])
        self.fc2 = nn.Linear(d_layers[-1], d_out)

    def forward(self, x_num, x_cat):
        x = torch.cat([x_num, self.cat_tokenizer(x_cat).flatten(1, -1)], dim=1)
        x = self.fc1(x)
        x = self.res_blocks(x)
        return self.fc2(x)
