===================================================
Densely Connected Convolutional Networks by Chainer
===================================================

This is an experimental implementation of `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_ using Chainer framework.

- Original paper: https://arxiv.org/abs/1608.06993
- Official implementation: https://github.com/liuzhuang13/DenseNet


Requirements
============

- `Chainer <http://chainer.org>`_  1.13+


Usage
=====

Train a DenseNet on CIFAR-10 dataset using::

    python codes/train.py --gpu 0
