import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class vae(nn.Module):
    def __init__(self, input_dim, latent_dim = 15, hidden_dim = 512):
        super(vae, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim,
        self.hidden_dim = hidden_dim


        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim//2, latent_dim)
        self.lvar = nn.Linear(hidden_dim//2, latent_dim)



        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, input_dim)
        )

        self.log_theta = nn.Parameter(torch.zeros(input_dim))


    def encode(self, x):
        u = self.encoder(x)
        mu = self.mu(u)
        lvar = self.lvar(u)

        return mu, lvar

    def lat_z_param(self, mu, lvar):
        if self.training:
            sigma = torch.exp(0.5 * lvar)
            eps = torch.randn_like(sigma)
            return mu + sigma * eps
        else:
            return mu

    def decode(self, z):

        y = self.decoder(z)

        r = F.softmax(y, dim=-1)

        return r

    def forward(self, x, t_count):
        mu, lvar = self.encode(x)

        z = self.lat_z_param(mu, lvar)

        r = self.decode(z)

        mu_x = t_count * r

        theta = torch.exp(self.log_theta).unsqueeze(0)

        return {"z": z, "mu": mu, "lvar": lvar, "r": r, "mu_counts": mu_x, "theta": theta}

    def nb_log_likelihood(self, x, mu, theta, tol = 1e-9):

        x = x.float()
        theta = torch.clamp(theta, min=tol)

        ex_log = (torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        + theta*(torch.log(theta + tol) - torch.log(theta + mu + tol)) + x*(torch.log(mu + tol)
                                                                            - torch.log(theta + mu + tol)))

        return ex_log.sum(dim = -1)

    def nb_loss(self, x, mu_x, theta):

        L = self.nb_log_likelihood(x, mu_x, theta)

        return -L.mean()

    def dkl_loss(self, mu, lvar):

        dkl = 1/2*torch.sum(mu**2 + torch.exp(lvar) - 1 - lvar, dim=-1)

        return dkl.mean()

