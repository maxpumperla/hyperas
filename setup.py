from setuptools import setup
from setuptools import find_packages

setup(name='hyperas',
      version='0.1.1',
      description='Simple wrapper for hyperopt to do convenient hyperparameter optimization for Keras models',
      url='http://github.com/maxpumperla/hyperas',
      download_url='https://github.com/maxpumperla/hyperas/tarball/0.1.1',
      author='Max Pumperla',
      author_email='max.pumperla@googlemail.com',
      install_requires=['keras', 'hyperopt'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
