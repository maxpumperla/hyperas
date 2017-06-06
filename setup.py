from setuptools import setup
from setuptools import find_packages

install_requirements = [
    'keras',
    'hyperopt',
    'entrypoints',
    'ipython<6.0',
    'jupyter',
    'nbformat',
    'nbconvert'
]

setup(name='hyperas',
      version='0.3',
      description='Simple wrapper for hyperopt to do convenient hyperparameter optimization for Keras models',
      url='http://github.com/maxpumperla/hyperas',
      download_url='https://github.com/maxpumperla/hyperas/tarball/0.3',
      author='Max Pumperla',
      author_email='max.pumperla@googlemail.com',
      install_requires=install_requirements,
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
