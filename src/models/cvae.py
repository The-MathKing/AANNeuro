"""
Conditional Variational Autoencoder (cVAE) vs Residualized Linear Representation Benchmark.
Used for exploratory analysis E10 to test whether non-linear age conditioning provides genuine orthogonalization over residualized PCA.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim=1, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar


def train_infold_cvae(X_train, age_train, latent_dim=16, epochs=80, lr=1e-3, seed=42):
    """
    Train CVAE inside CV fold.
    X_train: (n_samples, n_features) numpy array
    age_train: (n_samples, 1) numpy array
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler_x = StandardScaler()
    scaler_c = StandardScaler()

    x_norm = scaler_x.fit_transform(X_train)
    c_norm = scaler_c.fit_transform(age_train.reshape(-1, 1))

    x_t = torch.tensor(x_norm, dtype=torch.float32)
    c_t = torch.tensor(c_norm, dtype=torch.float32)

    model = CVAE(input_dim=X_train.shape[1], cond_dim=1, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x_t, c_t)
        recon_loss = nn.functional.mse_loss(recon_x, x_t, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.1 * kld_loss
        loss.backward()
        optimizer.step()

    model.eval()
    return model, scaler_x, scaler_c
