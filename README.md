# Piano-VAE
A Pytorch Implementation of VAE-based musical model to generate and interpolate piano'notes using Nottingham dataset.



![](./media/piano-VAE.svg) 

### Table of Contents

- **[Introduction](#Introduction)**
- **[Setup](#Setup)**
- [**Run the code**](#Run-the-code)
- **[Training](#Training)**
- **[Inference](#Inference)**
- **[Play with the model](#Play-with-the-model)**
- **[Connect with me](#Connect-with-me)**
- **[License](#License)** 



## Introduction

This is a Pytorch implementation of a musical model that capable to generate piano'notes and interpolate between them as the model latent space is continuous. The model is a variational autoencoder where the encoder and the decoder are LSTM networks. The model is trained on Nottingham dataset, you can download it from [here](http://www-ens.iro.umontreal.ca/~boulanni/icml2012).

### Setup

The code is using `pipenv` as a virtual environment and package manager. To run the code, all you need is to install the necessary dependencies. open the terminal and type:

- `git clone https://github.com/Khamies/Piano-VAE.git` 
- `cd Piano-VAE`
- `pipenv install`

And you should be ready to go to play with code and build upon it!

### Run the code

- To train the model, run: `python main.py`
- To train the model with specific arguments, run: `python main.py --batch_size=64`. The following command-line arguments are available:
  - Batch size: `--batch_size`
  - Learning rate: `--lr`
  - Embedding size: `--embed_size`
  - Hidden size: `--hidden_size`
  - Latent size: `--latent_size`

### Training

The model is trained on 30 epochs using Adam as an optimizer with a learning rate 0.001. Here are the results from training the LSTM-VAE model:

- **KL Loss**

  <img src="./media/KL.jpg" align="center" height="300" width="500" >

- **Reconstruction loss**

  <img src="./media/reco.jpg" align="center" height="300" width="500" >

- **KL loss vs Reconstruction loss**

  <img src="./media/kl_reco.jpg" align="center" height="300" width="500" >

- **ELBO loss**

  <img src="./media/elbo.jpg" align="center" height="300" width="500" >

### Inference

#### 1. Sample Generation

Here are generated samples from the model. We randomly sampled two latent codes z from standard Gaussian distributions.The following are the generated notes:

<p align="center">
 <a href="https://www.youtube.com/watch?v=fPu3hMfQC-A">  <img src="http://img.youtube.com/vi/fPu3hMfQC-A/0.jpg?raw=true" alt="Sublime's custom image"/> </a>
</p>

#### 2. Interpolation

The "President" word has been used as the start of the sentences. We randomly generated two sentences and interpolated between them.

- first audio:

  <p align="center">
   <a href="https://www.youtube.com/watch?v=fPu3hMfQC-A">  <img src="http://img.youtube.com/vi/fPu3hMfQC-A/0.jpg?raw=true" alt="Sublime's custom image"/> </a>
  </p>

- second audio:

  <p align="center">
   <a href="https://www.youtube.com/watch?v=fPu3hMfQC-A">  <img src="http://img.youtube.com/vi/fPu3hMfQC-A/0.jpg?raw=true" alt="Sublime's custom image"/> </a>
  </p>

interpolation 5

Interpolation 6

interpolation 16
