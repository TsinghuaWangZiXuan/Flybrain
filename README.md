# Fly's Brain Drivers Design System
## Background
<p>Inducing a specific protein into a single neuron in Drosophila relies on utilizing an upstream enhancer to drive downstream gene expression. Nowadays, many drivers in Drosophila have been developed in recent years. These fly strains made a great contribution to researches in neuroscience. Typically, they contained an upstream enhancer and a common promoter to initiate the transcription. In this way, these drivers can express proteins in targeted neurons. However, these strains of Drosophila, such as Flylight or VDRC, are not perfect. They hardly label a single neuron and are usually dirty. Thus, whether we could design specific drivers that are clean enough remains unclear.</p>
  
<p>To understand the corresponding relationship between enhancer sequences and report genes' expression in the brain and design our specific drivers, we tried several models and algorithms using deep learning to predict the expression pattern of reporter genes based on enhancer sequence features. Ideally, our system will learn how drivers work in fly's brain and can help us design specifc drivers.</p>

## Our Work
### Autoencoder
<p>Firstly, We implemented an autoencoder to encode DNA sequences. We aimed to investigate the distribution of DNA segments and to examine whether there are certain rules in distribution. The result of this model may lay a foundation for further analysis. The following figure is the structure of our autoencoder.</p>

<div align="left">
  
<img src="https://github.com/TsinghuaWangZiXuan/Flybrain/blob/main/Images/Autoencoder.png" height="300" width="400" >
  
</div>

### Variational Autoencoder(VAE)

<p>Next, we also implemented a similar architecture to process images. To boost the performance, we adopted a variational encoder in our model. The following figure is the structure of our VAE. </p>

<div align="left">
  
<img src="https://github.com/TsinghuaWangZiXuan/Flybrain/blob/main/Images/VAE.png" height="110" width="400" >
  
</div>

### Machine Learning
<p>After revealing the distribution of latent space, we then tried several traditional algorithms to predict the expression patterns from different enhances.</p>

<div align="left">
  
<img src="https://github.com/TsinghuaWangZiXuan/Flybrain/blob/main/Images/Traditional%20Machine%20Learnig.png" height="360" width="400" >
  
</div>

## Future Plan
## References
