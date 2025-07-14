import pandas as pd
import numpy as np
import torch
import math
from src.constants import POLLUTANT_COLUMNS

def calculate_baseline(dataset_df : pd.DataFrame):
    nlls = []
    for _, group in dataset_df.groupby('city_name'):
        group = group.copy()
        y = group.drop(columns=['city_name'])[POLLUTANT_COLUMNS]
        y = torch.tensor(y.values, dtype=torch.float32)
        mu = y.mean()
        sigma = y.std()

        eps = 1e-6
        sigma = torch.clamp(sigma, min=eps)
        z = (y - mu) / sigma

        log_prob = -0.5 * 7 * math.log(2 * math.pi) - torch.sum(torch.log(sigma)) - 0.5 * torch.sum(z ** 2, dim=1)
        nll = -log_prob.mean()

        nlls.append(nll.item())
        return np.average(nlls)
