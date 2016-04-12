"""Overlay Service Abstract Class
"""
import os
import time
import logging

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)


class OverlayNotAvailable(Exception):
  def __init__(self):
      self.value = 0
  def __str__(self):
      return repr(self.value)

