#!/usr/bin/env python
"""Wrapper module to start/stop overlay service
"""
import pytest
import argparse
import logging
import sys

from overlay.redisOverlay import RedisService
from overlay.alluxioOverlay import AlluxioService

# import core.slurm as slurm

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.0.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(format='%(module)s> %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('service')
  parser.add_argument('command')
  parser.add_argument('-n', '--name')
  parser.add_argument('-r', '--role', default='MASTER')
  args = parser.parse_args()

  logging.info('\n\nOverlay Executor:  service=`%s`   cmd=`%s`', args.service, args.command)

  valid_roles = ['MASTER', 'REPLICA', 'SLAVE']
  role = args.role
  if role not in valid_roles:
    logging.error('Undefined role provided. Role must be one of: %s', str(valid_roles))
    sys.exit(1)

  role = args.role
  if args.service == 'redis':
    if args.name is None:
      logging.error('ERROR. Service requires a application name with corresponding input (json) confile file')
      sys.exit()
    service = RedisService(args.name)
  elif args.service == 'alluxio':
    if args.name is None:
      logging.error('ERROR. Service requires a application name with corresponding input (json) confile file')
      sys.exit()
    service = AlluxioService(args.name, role)
  else:
    logging.error('Unrecognized Service. Quitting')
    sys.exit()


  if args.command == 'start':
    logging.info('Service: START')
    t = service.start()
    t.join()
  elif args.command == 'stop':
    logging.info('Service: STOP')
    service.stop()
  else:
    logging.warning("Command not recognized:  %s", args.command)

  logging.debug('Execution Complete for: %s', __file__)