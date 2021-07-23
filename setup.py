from setuptools import setup, find_packages

setup(
    name='saint-lightning',
    version='0.0.1',
    description='Pytorch Lightning version of SAINT Model',
    author='Luca Actis Grosso',
    author_email='lucaactisgrosso@gmail.com',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/Actis92/saint-lightning.git',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)