from setuptools import find_packages,setup

with open("requirements.txt") as f:
    requirements=f.read().splitlines()



setup(
name="disease detection",
version='0.0.1',
author="shahid azim",
author_email='shahidcst@gmail.com',
packages=find_packages(),
install_requires=requirements
)
