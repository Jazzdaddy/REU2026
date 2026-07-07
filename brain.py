import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os

class dataset(Dataset):
    def __init__(self, x_counts, x_input):
        self.counts = torch.tensor(x_counts, dtype=torch.float32)
        self.x_input = torch.tensor(x_input, dtype=torch.float32)

        gene_counts = self.counts.sum(dim=1, keepdim=True)
        self.gene_counts = torch.clamp(gene_counts, min=1.0)

    def __len__(self):
        return self.counts.shape[0]

    def __getitem__(self, idx):
        item = {
            "x_counts": self.counts[idx],
            "x_input": self.x_input[idx],
            "gene_counts": self.gene_counts[idx],
        }

        return item

def train_vae(model,dataset,epochs=100,batch_size=256,lr=1e-3,beta=0.5, save = False, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    print(f"device: {device}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_history = []

    progress_bar = tqdm(total=epochs, desc='Epochs', position=0, leave=True)

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_nb = 0.0
        total_kl = 0.0
        n_seen = 0

        for batch in loader:
            x_counts = batch["x_counts"].to(device)
            x_input = batch["x_input"].to(device)
            gene_counts = batch["gene_counts"].to(device)

            out = model(x_input, gene_counts)

            # check = out["mu_counts"]

            L = model.nb_loss(x=x_counts,mu_x=out["mu_counts"],theta=out["theta"])

            kl = model.dkl_loss(mu=out["mu"],lvar=out["lvar"],)

            loss = L + beta * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            bs = x_counts.shape[0]
            total_loss += loss.item() * bs
            total_nb += L.item() * bs
            total_kl += kl.item() * bs
            n_seen += bs

        epoch_stats = {"epoch": epoch,"loss": total_loss / n_seen,"nb_loss": total_nb / n_seen,"kl_loss": total_kl / n_seen}

        progress_bar.set_description(
            f'ELBO Loss: {total_loss / n_seen:.5f} NB Loss: {total_nb / n_seen:.5f} KL : {total_kl / n_seen:.5f} Epoch: {epoch}')
        progress_bar.update(1)

        train_history.append(epoch_stats)

    if save:
        os.makedirs("State_Dicts", exist_ok=True)
        torch.save(model.state_dict(),f'State_Dicts/vae_nn.pth')

    return train_history

@torch.no_grad()
def get_latent_mu(model,dataset, batch_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu",):
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_z = []

    for batch in loader:
        x_input = batch["x_input"].to(device)
        mu_z, logvar_z = model.encode(x_input)
        all_z.append(mu_z.cpu().numpy())

    return np.vstack(all_z)

def test_data(gene_counts, x_inputs):

    idx = np.arange(gene_counts.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=0, shuffle=True)

    train_dataset = dataset(gene_counts[train_idx], x_inputs[train_idx])
    val_dataset = dataset(gene_counts[val_idx], x_inputs[val_idx])

    return {"train_idx": train_idx, "val_idx": val_idx, "train_dataset": train_dataset, "val_dataset": val_dataset}

@torch.no_grad()
def eval_vae(model, dataset, batch_size=256, beta=0.5, device="cuda" if torch.cuda.is_available() else "cpu"):

    vae_nn = model.to(device)
    vae_nn.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss, total_nb, total_kl, n_seen = 0.0, 0.0, 0.0, 0

    for batch in loader:
        x_counts = batch["x_counts"].to(device)
        x_input = batch["x_input"].to(device)
        gene_counts = batch["gene_counts"].to(device)

        out = vae_nn(x_input, gene_counts)

        nb_loss = vae_nn.nb_loss(x=x_counts,mu_x=out["mu_counts"],theta=out["theta"])
        kl_loss = vae_nn.dkl_loss(mu=out["mu"],lvar=out["lvar"],)
        loss = nb_loss + beta * kl_loss

        bs = x_counts.shape[0]
        total_loss += loss.item() * bs
        total_nb += nb_loss.item() * bs
        total_kl += kl_loss.item() * bs
        n_seen += bs
    return {
        "loss": total_loss / n_seen,
        "nb_loss": total_nb / n_seen,
        "kl_loss": total_kl / n_seen,
        "nb_loss_per_gene": (total_nb / n_seen) / dataset.counts.shape[1],
    }

def validate_vae(vae_nn, data):
    x_counts = data.counts
    x_inputs = data.x_input

    counts_for_eval = x_counts.clone()
    inputs_for_eval = x_inputs.clone()

    full_dataset = dataset(counts_for_eval, inputs_for_eval)

    split = test_data(counts_for_eval, inputs_for_eval)

    full_metrics = eval_vae(vae_nn, full_dataset, beta=0.5)
    train_metrics = eval_vae(vae_nn, split["train_dataset"], beta=0.5)
    val_metrics = eval_vae(vae_nn, split["val_dataset"], beta=0.5)

    weighted = (len(split["train_dataset"]) * train_metrics["loss"] + len(split["val_dataset"]) * val_metrics["loss"]) / len(full_dataset)

    print("full:", full_metrics)
    print("train:", train_metrics)
    print("val:", val_metrics)
    print("weighted:", weighted)
