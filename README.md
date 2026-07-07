# REU2026

## Instructions:
- Download the Seurat/scRNA raw count file to a 5fad data object or alternative data object
- Run the requirements.txt file in your terminal
  <code> pip install -r requirements.txt </code>
- Create a subdirectory called "Data". This will be used to save and load data files for training and analysis.
- Train VAE following the outline VAE test.py
- After training and validation, export latent data to Monocle3 to perform Pseudotime analysis

## vae.py
### This file contains the class for the Variational Autoencoder along with its functions
#### Input parameters:
- **Input dim ($n$)**: Number of Highly Variable Genes selected. 
- **Hidden dim**: Dimension of the first hidden layer, default set to 512. Each layer hidden layer in the Encoder reduces the dimension by 2, then to the Latent dimension.
- **Latent dim ($rR)**: The dimension of the reduced data, default set to 15. > $r << n$

#### Encoder: 
Given a sample of log normalized count from a cell, $x_i$, outputs the parameters for the latent space.


