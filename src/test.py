#!/usr/bin/env python
"""
Test module
"""
import pytest
import argparse
import logging

import overlay.redisOverlay

# import core.slurm as slurm

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.0.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # parser.add_argument('-t', '--test', action='store_true')
  # # parser.add_argument('-p', '--pytest', action='store_true')
  # args = parser.parse_args()

  # if args.pytest:
  pytest.main()
  # overlay.redisOverlay.test_redisclient('test_overlay')
  # test_redisclient('test_overlay')
    # mod = args.test
    # slurm.test_slurm()
    # sys.exit(0)

  logging.debug('Execution Complete for: %s', __file__)