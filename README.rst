===============
Saint Lightning
===============

This repository contains an implementation of SAINT (Self-Attention and Intersample Attention Transformer) using Pytorch-Lightning_ as a framework
and Hydra_ for the configuration.
Find the paper on arxiv_

Check the website_ for more information.

.. _arxiv: https://arxiv.org/abs/2106.01342
.. _Pytorch-Lightning: https://www.pytorchlightning.ai/
.. _Hydra: https://hydra.cc/
.. _website: https://actis92.github.io/lit-saint/

How to install
--------------

.. code-block:: bash

    pip install lit-saint


How to Use it
-------------

1. Create an yaml file that contains the configuration needed by the application or use default values

2. Create an instance of SaintConfig using Hydra

3. Create the Dataframe that will be used for the model. In order to split correctly the data, you need to add a new column where you assign the label "train" to the rows of the training set, "validation" for the ones of the validation set and "test" for the testing one

.. code-block:: python3

    data_module = SaintDatamodule(df=df, target="TARGET", split_column="SPLIT")

4. Create an instance of SaintDataModule and SAINT

.. code-block:: python3

    model = Saint(categories=data_module.categorical_dims, continuous=data_module.numerical_columns,
                  config=cfg, dim_target=data_module.dim_target)

5. Create the Trainers defined by Pytorch lightning to fit the model

.. code-block:: python3

    pretrainer = Trainer(max_epochs=1)
    trainer = Trainer(max_epochs=5)

6. Create the SaintTrainer that will be used in order to fit the model and make predictions

.. code-block:: python3

    saint_trainer = SaintTrainer(pretrainer=pretrainer, trainer=trainer)
    saint_trainer.fit(model=model, datamodule=data_module, enable_pretraining=True)

7. Then you can define the data for the prediction step

.. code-block:: python3

    prediction = saint_trainer.predict(model=model, datamodule=data_module, df=df_to_predict)
    df_test["prediction"] = np.argmax(prediction, axis=1)

Preprocessing
^^^^^^^^^^^^^^

1. The numerical columns are filled with zeros in case of missing values
2. The categorical columns are filled with a new category with the value SAINT_NAN in case of missing values
3. The numerical columns are scaled using a StandardScaler unless you specify a different scaler inside the SaintDataModule
4. The columns that are of type ["object", "category"] are considered categorical
5. The columns that are of type ["int64", "float64", "int32", "float32"] are considered numerical
6. All the columns that have a type different from the one specified before aren't used inside the model
7. During the pretraining the rows that have the target column with nan value are used, instead are dropped before start the training

Some suggestions are:

* If you want to fill the columns in a different way from the default one you need to do it before to use in the SaintDataModule
* If you want to use columns that contains datetime, you need to extract some features (i.e day of week) or convert them in epoch
* If you have a classification problem the column that contain the target must be of type "object" or "category"


How to Generate Yaml
--------------------
.. code-block:: python3

    from lit_saint import SaintConfig
    from omegaconf import OmegaConf


    conf = OmegaConf.create(SaintConfig)
    with open("<FILE_NAME>", "w+") as fp:
        OmegaConf.save(config=conf, f=fp.name)


In order to make type validation at runtime, you need to add at the beginning of your file the following lines:

.. code-block:: yaml

    defaults:
      - base_config


Credits
-------

We would like to thank the repo with the official implementation of SAINT:
https://github.com/somepago/saint
