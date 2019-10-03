# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 20:14:25 2019

@author: Nick Anthony
"""
from setuptools import setup, find_packages

setup(name='autoRoi',
      version='0.0.1',
      description='A module for using a UNet convolutional neural network to segment the nuclei of PWS images.',
      author='Nick Anthony',
      author_email='nicholas.anthony@northwestern.edu',
      install_requires=['numpy',
						'tensorflow',
                        'pwspy',
                        'h5py'],
      package_dir={'': 'src'},
      package_data={'autoRoi': []},
      packages=find_packages('src'))
