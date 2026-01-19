# TopicModelingWithReplicatedSoftmax
Thesis project for master. Direct integration with the OCTIS package for topic modeling.

The aim was to implement and optimize the [RSM (Hinton and Salakhutdinov 2009)](https://proceedings.neurips.cc/paper_files/paper/2009/file/31839b036f63806cba3f47b93af8ccb5-Paper.pdf) and the [over-RSM (Hinton, Salakhutdinov and Strivastava 2013)](https://arxiv.org/pdf/1309.6865) topic models.

These are Topic Models based on the concept of the Restricted Boltzmann Machines. 

The over-RSM is really a fancy modification of the RSM that gives it the structure of a deep Boltzmann Machine.

This implementation is the first to use Mean Field Contrastive Duvergence, Persistent Contrastive Divergence, L1 penalization and gradient descent optimization (momentum, RMSprop, Adam)
for this topic model. Inside OCTIS is also possible to tweak the hyperparameters using bayesian optimization.

The validation metric used in this repo is the perplexity (a mean field approximation of it), while inside OCTIS is also possible to compute other common metrics like classification and coherence. 


The code is inspired from these repositories:

| Purpose                     | Repository                                                                 |
|----------------------------:|----------------------------------------------------------------------------|
| standard RSM                | https://github.com/wtang0512/Replicated-Softmax-Model                      |
| standard RSM                | https://github.com/fylance/rsm                                             |
| over-RSM                    | https://github.com/dongwookim-ml/RSM                                       |
| standard RBM                | https://github.com/mr-easy/Restricted-Boltzmann-Machine                    |
| topic modeling framework    | https://github.com/MIND-Lab/OCTIS                                          |


