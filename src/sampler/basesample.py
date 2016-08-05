"""Basic Sampling Object 
"""
import abc
import os
import time
import logging
import numpy as np

from core.slurm import systemsettings

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)


class SamplerBasic(object):
  """Basic Sampler provides an abstract layer for writing/processing 
  a sampling algorithm
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, name, **kwargs):
    self.name = name
    logging.info('Sampler Object created for:  %s', name)

  @abc.abstractmethod
  def execute(self, num):
    """Method to check if service is running on the given host
    """
    pass


class UniformSampler(SamplerBasic):
  """ Uniform sampler takes a list of object and uniformly samples among them
  This is the most simple and naive sampler
  """

  #  TODO:  Ensure catalog and cache run on separate servers
  def __init__(self, choice_list):
    SamplerBasic.__init__(self, "Uniform")
    self.choices = choice_list

  def execute(self, num):
    logging.info('UNIFORM SAMPLER:  sampling called for %d  items', num)
    need_replace = (len(self.choices) < num)
    candidates = np.random.choice(self.choices, size=num, replace=need_replace)
    return candidates  

