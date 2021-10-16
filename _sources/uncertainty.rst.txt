======================
Uncertainty Estimation
======================

Monte Carlo Dropout
--------------------

Monte Carlo Dropout, proposed by `Gal & Ghahramani <https://arxiv.org/abs/1506.02142>`_ (2016),
is a clever realization that the use of the regular dropout can be interpreted
as a Bayesian approximation of a well-known probabilistic model: the Gaussian process.
We can treat the many different networks (with different neurons dropped out)
as Monte Carlo samples from the space of all available models.
This provides mathematical grounds to reason about the model’s uncertainty and, as it turns out,
often improves its performance.

How does it work? We simply apply dropout at test time, that's all!

Implementation
^^^^^^^^^^^^^^

In the function predict_step of the LightningModule is added this code:

.. code-block:: python3

    for m in self.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

In this way all the Dropout layers become trainable during the predict step and so
we will get different prediction anytime we call the model on the same data


How to use it
^^^^^^^^^^^^^^
Using the SaintTrainer is possible to specify the number of iterations to use for the
MCDropout in the predict, when it is zero it doesn't use the algorithm and return the standard result.
The dimension of the output will be [N_SAMPLE, N_CLASS, N_ITERATION] where:

* N_SAMPLE is equal to the number of rows of the dataset used for the prediction;
* N_CLASS is equal to the number of classes of the problem(in case of regression is equal to 1);
* N_ITERATION is equal to the number of iteration of the MCDropout process

Remember to set the value for the dropout in the Transformer or/and in the embeddings different from zero,
otherwise all the predictions will have the same value and this technique will be not applicable.

This technique can be used to perform Active Learning, and find the the samples where the model has a greatest uncertainty
and that if we can obtain labels for these samples we can improve the performance of the network

.. code-block:: python3

    prediction = saint_trainer.predict(model=model, datamodule=data_module, df=df_test, mc_dropout_iterations=2)