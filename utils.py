import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from junk import run_with_spinner

def preprocess_data(adata, counts_str, log_str, n_genes):
    adata.var_names_make_unique()

    adata.layers[counts_str] = adata.X.copy()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", layer="counts")

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()

    x_counts = adata_hvg.layers[counts_str].copy()
    x_inputs = adata_hvg.layers[log_str].copy()

    if sparse.issparse(x_counts):
        x_counts = x_counts.toarray()

    if sparse.issparse(x_inputs):
        x_inputs = x_inputs.toarray()

    x_counts = x_counts.astype("float32")
    x_inputs = x_inputs.astype("float32")

    return x_counts, x_inputs, adata_hvg

def check_dummy_case(vae_nn, x_counts):
    x_counts = torch.tensor(x_counts, dtype=torch.float32)
    library_size = x_counts.sum(dim=1, keepdim=True).clamp(min=1.0)

    global_r = x_counts.sum(dim=0, keepdim=True)
    global_r = global_r / global_r.sum()

    mu_baseline = library_size * global_r

    theta = torch.ones(1, x_counts.shape[1])

    baseline_nb = vae_nn.nb_loss(x_counts, mu_baseline, theta)

    print(baseline_nb.item())

def write_latent_h5ad(adata_hvg, csv_str = "Data/5xad_vae_latent.csv", h5ad_str = "Data/5xad_hvg_with_vae.h5ad"):
    Z_df = pd.DataFrame(adata_hvg.obsm["X_vae"], index=adata_hvg.obs_names, columns=[f"VAE_{j + 1}" for j in range(adata_hvg.obsm["X_vae"].shape[1])])

    Z_df.to_csv(csv_str)
    adata_hvg.write_h5ad(h5ad_str)





