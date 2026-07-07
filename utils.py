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
    Z_df = pd.DataFrame(
        adata_hvg.obsm["X_vae"],
        index=adata_hvg.obs_names,
        columns=[f"VAE_{j + 1}" for j in range(adata_hvg.obsm["X_vae"].shape[1])]
    )

    # Z_df.to_csv(csv_str)
    run_with_spinner(
        f"Saving latent CSV to {csv_str}",
        Z_df.to_csv,
        csv_str,
    )

    run_with_spinner(
        f"Saving AnnData h5ad to {h5ad_str}",
        adata_hvg.write_h5ad,
        h5ad_str,
    )
    print("Finished saving VAE outputs.")
    # adata_hvg.write_h5ad(h5ad_str)

def neighbor_label_agreement(Z, labels, k=15):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(Z)
    _, idx = nbrs.kneighbors(Z)

    idx = idx[:, 1:]

    labels = np.asarray(labels)
    same = labels[idx] == labels[:, None]

    return same.mean()

def pseudotime_comp(adata):
    # print("Mean/median pseudotime by condition:")
    # print(
    #     adata.obs.groupby("condition", observed=True).agg(
    #         n_cells=("condition", "size"),
    #         mean_dpt_vae=("dpt_vae", "mean"),
    #         median_dpt_vae=("dpt_vae", "median"),
    #         mean_dpt_pca=("dpt_pca", "mean"),
    #         median_dpt_pca=("dpt_pca", "median"),
    #     )
    # )
    # print()

    rho = spearmanr(
        adata.obs["dpt_vae"],
        adata.obs["dpt_pca"],
        nan_policy="omit",
    ).correlation

    print("Spearman correlation between VAE-DPT and PCA-DPT:")
    print(rho)
    print()

    opc = adata.obs["ident"] == "OPCs"
    oligo = adata.obs["ident"] == "Oligodendrocytes"

    sep_vae = (
            adata.obs.loc[oligo, "dpt_vae"].mean()
            - adata.obs.loc[opc, "dpt_vae"].mean()
    )

    sep_pca = (
            adata.obs.loc[oligo, "dpt_pca"].mean()
            - adata.obs.loc[opc, "dpt_pca"].mean()
    )

    print("Mean pseudotime separation, Oligodendrocytes - OPCs:")
    print("VAE:", sep_vae)
    print("PCA:", sep_pca)

    sc.pl.violin(
        adata,
        keys=["dpt_vae", "dpt_pca"],
        groupby="ident",
        rotation=45,
    )

    y = (adata.obs["ident"] == "Oligodendrocytes").astype(int)

    auc_vae = roc_auc_score(y, adata.obs["dpt_vae"])
    auc_pca = roc_auc_score(y, adata.obs["dpt_pca"])

    print("AUC VAE:", auc_vae)
    print("AUC PCA:", auc_pca)

def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    pooled = np.sqrt(
        ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
        / (nx + ny - 2)
    )

    return (x.mean() - y.mean()) / pooled

def get_gene_expression(adata, gene, layer="logcounts"):
    X = adata.layers[layer]

    gene_idx = adata.var_names.get_loc(gene)

    x = X[:, gene_idx]

    if sparse.issparse(x):
        x = x.toarray().ravel()
    else:
        x = np.asarray(x).ravel()

    return x


def plot_gene_vs_pseudotime(adata, gene, pseudotime_key, layer="logcounts"):
    tau = adata.obs[pseudotime_key].to_numpy()
    expr = get_gene_expression(adata, gene, layer=layer)

    order = np.argsort(tau)
    tau_sorted = tau[order]
    expr_sorted = expr[order]

    # Smooth with rolling average
    window = max(25, len(expr_sorted) // 50)
    kernel = np.ones(window) / window
    expr_smooth = np.convolve(expr_sorted, kernel, mode="same")

    plt.figure(figsize=(6, 4))
    plt.scatter(tau, expr, s=3, alpha=0.15)
    plt.plot(tau_sorted, expr_smooth, linewidth=2)
    plt.xlabel(pseudotime_key)
    plt.ylabel(f"{gene} expression")
    plt.title(f"{gene} vs {pseudotime_key}")
    plt.show()


