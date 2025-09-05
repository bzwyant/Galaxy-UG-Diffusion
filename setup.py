from setuptools import setup, find_packages
import os


def get_requirements(filename="requirements.txt"):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), "r") as f:
        requirements = f.read().splitlines()
    return requirements
    

setup(
    name="Galaxy-UG-Diffusion",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=get_requirements()
)