import os

from setuptools import setup

package_root = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(package_root, "src/lit_saint/version.py")) as fp:
    exec(fp.read(), version)
version = version["__version__"]

setup(
    name='lit-saint',
    version=version,
    license='MIT',
    description='Pytorch Lightning implementation of SAINT Model',
    author='Luca Actis Grosso',
    author_email='lucaactisgrosso@gmail.com',
    url='https://github.com/Actis92/lit-saint.git',
    download_url=f'https://github.com/Actis92/lit-saint.git/archive/{version}.tar.gz',
    keywords=['TABULAR', 'SELF SUPERVISED', 'PYTORCH LIGHTNING'],
    install_requires=['pytorch-lightning>=1.3.0,<2',
                      'torch>=1.4',
                      'einops>=0.3.0',
                      'pandas>=1.0',
                      'scikit-learn>=0.24.2',
                      'hydra-core>=1.1.0'
                      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ]
)