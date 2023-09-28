from distutils.core import setup
from os.path import isdir
from itertools import product

# folders names here are the same as the package names
all_packages = ['data', 'mems', 'mems_obs', 'algos', 'viz', 'offlinelrkit']
packages = list(filter(isdir, all_packages))

setup(
    name='MCNN',
    packages=packages,
    version='0.1')