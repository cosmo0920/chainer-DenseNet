#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import argparse
import pkg_resources

from chainer import serializers
import chainer.links as L
from chainer.cuda import to_gpu
from chainer.cuda import to_cpu
import dataset
from chainer.datasets import cifar
from model import DenseNet
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

matplotlib_pkg = None
try:
    matplotlib_pkg = pkg_resources.get_distribution('matplotlib')
    import matplotlib.pyplot as plt
except pkg_resources.DistributionNotFound:
    pass

cls_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']

def predict(model, image_id, test):
    x, t = test[image_id]
    model.to_cpu()
    y = model.predictor(x[None, ...]).data.argmax(axis=1)[0]
    print('predicted_label:', cls_names[y])
    print('answer:', cls_names[t])

    if matplotlib_pkg:
        plt.imshow(x.transpose(1, 2, 0))
        plt.show()

def main():
    # define options
    parser = argparse.ArgumentParser(
        description='Training script of DenseNet on CIFAR-10 dataset')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Validation minibatch size')
    parser.add_argument('--numlayers', '-L', type=int, default=40,
                        help='Number of layers')
    parser.add_argument('--growth', '-G', type=int, default=12,
                        help='Growth rate parameter')
    parser.add_argument('--dropout', '-D', type=float, default=0.2,
                        help='Dropout ratio')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=('cifar10', 'cifar100'),
                        help='Dataset used for training (Default is C10)')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()

    # setup model
    model = L.Classifier(DenseNet(args.numlayers, args.growth, 16,
                                  args.dropout, 10))

    # そのオブジェクトに保存済みパラメータをロードする
    serializers.load_npz('result/model_300.npz', model)

    for i in range(10, 15):
        predict(model, i, test)


if __name__ == '__main__':
    main()
