# Collection of Generative Adversarial Networks

## 🚀 Installation
```
git clone git@github.com:chen-yuxuan/gans.git
cd gans
pip install -e .
```

## 💡 Usage

To run the default experiment setting (i.e. GAN on MNIST), run:
```bash
python main.py
```

To run your own setting, an example:
```bash
python main.py dataset=cifar10 model=wgan batch_size=64 hidden_size=512 num_epochs=50
```

To show the available options and the default config, do:
```bash
python main.py --help
```
which results in something like this:

```
== Configuration groups ==
Compose your configuration from those groups (group=option)

dataset: mnist


== Config ==
Override anything in the config (foo.bar=value)

dataset:
  _target_: gans.data.mnist
  root: ./datasets/
  train: true
  download: false

model: GAN
seed: 1234
cuda_device: 0

batch_size: 100
hidden_size: 256
latent_size: 16
num_epochs: 100
g_lr: 0.0002
d_lr: 0.0001
weight_decay: 0
```

## List of GANs
### Vanilla GAN (GAN)

- Paper: [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- Citation
```
@inproceedings{NIPS2014_5ca3e9b1,
 author = {Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {Z. Ghahramani and M. Welling and C. Cortes and N. Lawrence and K. Q. Weinberger},
 pages = {},
 publisher = {Curran Associates, Inc.},
 title = {Generative Adversarial Nets},
 url = {https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf},
 volume = {27},
 year = {2014}
}
```
- Results

![GAN for MNIST](docs/GAN_mnist.png)


### Conditional GAN (CGAN)

- Paper: [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)
- Citation
```
@misc{mirza2014conditional,
      title={Conditional Generative Adversarial Nets}, 
      author={Mehdi Mirza and Simon Osindero},
      year={2014},
      eprint={1411.1784},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
- Results

![CGAN for MNIST](docs/CGAN_mnist.png)


### Wasserstein GAN (WGAN)

- Paper: [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
- Citation
```
@InProceedings{pmlr-v70-arjovsky17a,
  title = 	 {{W}asserstein Generative Adversarial Networks},
  author =       {Martin Arjovsky and Soumith Chintala and L{\'e}on Bottou},
  booktitle = 	 {Proceedings of the 34th International Conference on Machine Learning},
  pages = 	 {214--223},
  year = 	 {2017},
  editor = 	 {Precup, Doina and Teh, Yee Whye},
  volume = 	 {70},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {06--11 Aug},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf},
  url = 	 {https://proceedings.mlr.press/v70/arjovsky17a.html},
  abstract = 	 {We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to different distances between distributions.}
}
```

- Results

![WGAN for MNIST](docs/WGAN_mnist.png)