===============
Saint Lightning
===============

This repository contains an implementation of SAINT using Pytorch-Lightning_ as a framework
and Hydra_ for the configuration.
Find the paper on arxiv_

.. _arxiv: https://arxiv.org/abs/2106.01342
.. _Pytorch-Lightning: https://www.pytorchlightning.ai/
.. _Hydra: https://hydra.cc/




Network Architecture
--------------------
| SAINT is a deep learning approach to solving tabular data problems. It performs attention over both rows and columns, and it includes an enhanced embedding method.
There are also contrastive self-supervised pre-training method for use when
labels are scarce

.. image:: ./pipeline.png
    :alt: saint-image

How Use it
----------

#. Create an yaml file that contains the configuration needed by the application and create an instance of SaintConfig using Hydra

#. Create the Dataframe that will be used for the model. In order to split correctly the data you need to add a new column where you assign the label "train" to the rows of the training set, "validation" for the ones of the validation set and "test" for the test set

#. Create an instance of SaintDataModule and SAINT

#. Used the Trainer define by Pytorch lightning to fit the model



Credits
-------

We would like to thank the repo with the official implementation of SAINT:
https://github.com/somepago/saint
