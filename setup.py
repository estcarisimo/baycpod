from setuptools import setup

import baycpod

setup(
    name='baycpod',
    version=baycpod.__version__,
    description='Some Bayesian changepoint detection algorithms',
    author='Johannes Kulick & Esteban Carisimo',
    author_email='mail@johanneskulick.net',
    url='http://github.com/estcarisimo/baycpod',
    packages=['baycpod'],
    install_requires=['scipy', 'numpy', 'decorator', 'torch'],
    extras_require={
        'dev': ['pytest'],
        'multivariate': ['scipy>=1.6.0'],
        'plot': ['matplotlib'],
    }
)
