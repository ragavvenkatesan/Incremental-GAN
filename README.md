This is the code for the paper:

[Ragav Venkatesan](http://www.ragav.net), Hemanth Venkateshwara, Sethuraman Panchanathan, Baoxin Li "[A strategy for an uncompromising incremental learner](https://arxiv.org/abs/1705.00744)"arXiv:1705.00744, 2017.

How to run the code
-------------------

There are three different codes in this git one for mnist, cifar10 and svhn datasets, each in its own directory. Each directory has a 
``site_.py`` for site Sb and a ``site_2.py`` for Si as mentioned in the paper. ``site_1.py`` will learn both Nb and Gb, each saving its 
parameters, confusion matrices and some activities in the ``records\site_1`` directory. These will be loaded when running the ``site_2.py``
which, should be run next. 

To run the codes simply do:

.. code-block:: bash

    python mnist\site_1.py
    python mnist\site_2.py

Run similarly for other datasets also. The directory ``records`` will be created which will hold all results and model parameters, including
layer-wise activities and confusion matrices as described in the paper. All you need will be available in this directory and it is easily
navigable as directories are documented by nomenclature.


The MNIST block works as it says in the paper. The CIFAR-10 and SVHN are still begin setup to produce results using a deconvolutional GAN.
Set it up as per the paper and it will work, but at the moment, we are experimenting for better results. 


Pre-requisites
--------------

These codes use the [yann toolbox](http://wwww.yann.network) internally to run, so that needs to be setup properly.

Thanks for using the code, hope you had fun.
Ragav Venkatesan http://www.ragav.net