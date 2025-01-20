# CSCoefficientsLearning

This is the code and data for:
Chaoqing Tang, et al. 《Training-free Ultra Small Model for Universal Sparse Reconstruction in Compressed Sensing》

, which makes universal compressed sensing reconstruction fast and accurate in neural network architecture, and more importantly, this coefficients learning method has the same generality and is fully interpretable, it requires no training. We hope this method can fully substitute traditional iterative methods like OMP and IHT, especially for spase reconstruction of image and large-scale data in under-determined linear systems.


Both PaddlePaddle and Pytorch version are given, our paper uses the PaddlePaddle version.

We implemented three structures, i.e. CSRec1, CSRec2 and CSRec3, they work both for 1D and 2D signals. CSRec2 work best in our test.


CLOMP1D is for one-dimensional signal testing;
CLOMP2D is for two-dimensional signal testing;



