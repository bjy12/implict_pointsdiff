import os.path as osp
from setuptools import find_packages, setup

requirements = ["hydra-core==0.11.3"]


exec(open(osp.join("model", "_version.py")).read())

setup(
    name="implict_diff_pointnet",
    version=__version__,
    packages=find_packages(),
    install_requires=requirements,
)
