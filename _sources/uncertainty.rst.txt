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