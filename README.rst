===============
Saint Lightning
===============

This repository contains an implementation of SAINT using Pytorch-Lightning_ as a framework
and Hydra_ for the configuration.
Find the paper on arxiv_

.. _arxiv: https://arxiv.org/abs/2106.01342
.. _Pytorch-Lightning: https://www.pytorchlightning.ai/
.. _Hydra: https://hydra.cc/

How to install
--------------

.. code-block:: bash

    pip install lit-saint


Network Architecture
--------------------
| SAINT is a deep learning approach to solving tabular data problems. It performs attention over both rows and columns, and it includes an enhanced embedding method.
There are also contrastive self-supervised pre-training methods that can be used when
labels are scarce.

.. image:: ./pipeline.png
    :alt: saint-image

How to Use it
-------------

1. Create an yaml file that contains the configuration needed by the application and create an instance of SaintConfig using Hydra

2. Create the Dataframe that will be used for the model. In order to split correctly the data, you need to add a new column where you assign the label "train" to the rows of the training set, "validation" for the ones of the validation set and "test" for the testing one

.. code-block:: python3

    data_module = SaintDatamodule(df=df, target="TARGET", split_column="SPLIT", pretraining=True)

3. Create an instance of SaintDataModule and SAINT

.. code-block:: python3

    model = SAINT(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=cfg, pretraining=True)

4. Use the Trainer defined by Pytorch lightning to fit the model

.. code-block:: python3

    pretrainer = Trainer(max_epochs=10)
    pretrainer.fit(model, data_module)

5. After the pretraining, you can train the model using a supervised objective function

.. code-block:: python3

    model.pretraining = False
    data_module.pretraining = False
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data_module)

6. Then you can define the data for the prediction step

.. code-block:: python3

    data_module.set_predict_set(df_test)
    prediction = trainer.predict(model, datamodule=data_module)
    df_test["prediction"] = torch.cat(prediction).numpy()


Credits
-------

We would like to thank the repo with the official implementation of SAINT:
https://github.com/somepago/saint
