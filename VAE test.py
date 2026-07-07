import torch
import numpy as np
import pandas as pd
import scanpy as sc
import os


from vae import vae
from brain import dataset, train_vae, get_latent_mu, test_data, validate_vae
from utils import *


train = False
validate = False
verbose = True

os.makedirs("Data", exist_ok=True)

# adata = sc.read_h5ad("Data/5xad.h5ad")
#
# x_counts, x_inputs, adata_hvg  = preprocess_data(adata, 'counts', 'logcounts', 2000)
#
# np.savez('Data/5xad_hvg.npz',
#          x_counts = x_counts, x_inputs = x_inputs,
#          cell_names=adata_hvg.obs_names.to_numpy(),
#          gene_names=adata_hvg.var_names.to_numpy())
# adata_hvg.write_h5ad("Data/5xad_hvg.h5ad")

adata_ = np.load('Data/5xad_hvg.npz')
x_counts, x_inputs= adata_['x_counts'], adata_['x_inputs']
adata_hvg = sc.read_h5ad("Data/5xad_hvg.h5ad")

if verbose:
    print("x_counts id:", id(x_counts))
    print("x_inputs id:", id(x_inputs))

    print("x_counts:", x_counts.shape, x_counts.dtype)
    print("x_inputs:", x_inputs.shape, x_inputs.dtype)

    print("x_counts min/max/mean:", x_counts.min(), x_counts.max(), x_counts.mean())
    print("x_inputs min/max/mean:", x_inputs.min(), x_inputs.max(), x_inputs.mean())

data = dataset(x_counts, x_inputs)

n_c = x_counts.shape[0]
n_g = x_counts.shape[1]

vae_nn = vae(input_dim = n_g, latent_dim = 15, hidden_dim = 512)

"""***************  Train and Save VAE  ***************"""
if train:
    results = train_vae(model = vae_nn, dataset = data, epochs=100, batch_size = 256, save = True)

    if verbose:
        print(results)

vae_nn.load_state_dict(torch.load('State_Dicts/vae_nn.pth'))
vae_nn.eval()

validate_vae(vae_nn, data)

Z_data = get_latent_mu(vae_nn, data)
sc.pp.pca(adata_hvg, n_comps=15)
adata_hvg.obsm["X_vae"] = Z_data

write_latent_h5ad(adata_hvg, csv_str = "Data/5xad_vae_latent.csv", h5ad_str = "Data/5xad_hvg_with_vae.h5ad")
