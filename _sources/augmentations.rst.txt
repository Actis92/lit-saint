============================
Data Augmentation Techniques
============================

Data augmentation techniques generate different versions of a real dataset


CutMix
------
CutMix is an image data augmentation strategy.
Instead of simply removing pixels, it replace the removed regions with a patch from another image

In this case we have applied this technique replacing a subset of values of the tensor with other
values that are present in some random position of the tensor

.. list-table::

    * - .. figure:: ./_images/table.png

           Fig 1. Original Data

      - .. figure:: ./_images/permutation.png

           Fig 2. Permutation

      - .. figure:: ./_images/cutmix.png

           Fig 3. CutMix with lam=0.3


Mixup
------
Mixup is a data augmentation technique that generates a weighted combinations of random image pairs from the training data.

In this case we have applied this technique replacing each value with a weighted average between the value and another random value of the tensor

.. list-table::

    * - .. figure:: ./_images/table.png

           Fig 1. Original Data

      - .. figure:: ./_images/permutation.png

           Fig 2. Permutation

      - .. figure:: ./_images/mixup.png

           Fig 3. Mixup with lam=0.5
