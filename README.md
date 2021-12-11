# Collection of Generative Adversarial Networks

### 🚀&nbsp; Installation
```
git clone git@github.com:chen-yuxuan/gans.git
cd gans
pip install -e .
```
### 
#### Vanilla Generative Adversarial Network (GAN)

- [Paper]()
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


#### Wasserstein Generative Adversarial Network (WGAN)

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
