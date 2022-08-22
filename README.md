# TPP (Trainability Preserving Pruning)

### [ArXiv](https://arxiv.org/abs/2207.12534) | [PDF](https://arxiv.org/pdf/2207.12534.pdf) 

<div align="center">
    <a><img src="figs/smile.png"  height="90px" ></a>
    &nbsp &nbsp
    <a><img src="figs/neu.png"  height="90px" ></a>
</div>

This repository is for a new structured network pruning method (`Trainability Preserving Pruning, TPP`) for efficient deep learning, introduced in our paper:
> **Trainability Preserving Neural Structured Pruning** \
> [Huan Wang](http://huanwang.tech/), [Yun Fu](http://www1.ece.neu.edu/~yunfu/) \
> Northeastern University, Boston, MA, USA


## Abstract
Recent works empirically find finetuning learning rate is critical to the final performance in neural network structured pruning. Further researches show it is the network *trainability* broken by pruning that plays behind, thus calling for an urgent need to recover trainability before finetuning. Existing attempts propose to exploit weight orthogonalization to achieve dynamical isometry aiming for improved trainability. However, they only work for *linear* MLP networks. How to develop a filter pruning method that maintains or recovers trainability *and* is scalable to modern deep networks remains elusive. In this paper, we present *trainability preserving pruning* (TPP), a regularization-based structured pruning method that can effectively maintain trainability during sparsification. Specifically, TPP regularizes the gram matrix of convolutional kernels so as to *de-correlate* the pruned filters from the kept filters. Besides the convolutional layers, we also propose to regularize the BN parameters for better preserving trainability. Empirically, TPP can compete with the ground-truth dynamical isometry recovery method on linear MLP networks. On non-linear networks (ResNet56/VGG19, CIFAR datasets), it outperforms the other counterpart solutions *by a large margin*. Moreover, TPP can also work effectively with modern deep networks (ResNets) on ImageNet, delivering encouraging performance in comparison to many recent filter pruning methods. To our best knowledge, this is the *first* approach that effectively maintains trainability during pruning for the *large-scale* deep neural networks.

## TODO
Code and trained models will be released soon. Stay tuned!

## Acknowledgments
In this code we refer to the following implementations: [Regularization-Pruning](https://github.com/MingSun-Tse/Regularization-Pruning), [pytorch imagenet example](https://github.com/pytorch/examples/tree/master/imagenet), [rethinking-network-pruning](https://github.com/Eric-mingjie/rethinking-network-pruning), [EigenDamage-Pytorch](https://github.com/alecwangcq/EigenDamage-Pytorch), [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). Great thanks to them!
