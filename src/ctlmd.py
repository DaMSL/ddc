import argparse
import sys
import os
import sys

import mdtraj as md
import numpy as np
from numpy import linalg as LA
import nearpy
from nearpy.storage.storage_redis import RedisStorage
from nearpy.hashes import RandomBinaryProjections

import redisCatalog
from common import DEFAULT, executecmd
from macrothread import macrothread
from slurm import slurm

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)



def initialize():

  pass


class controlJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'simmd')
      # State Data for Simulation MacroThread -- organized by state
      self.setInput('LDIndexList')
      self.setTerm('JCComplete', 'processed')
      self.setExec('indexSize')
      self.setSplit('anlSplitParam')
      
      # exec incl hash key-name
      # TODO: wildcard loading of data

    def term(self):
      # For now
      return False

    def split(self):
      catalog = self.getCatalog()
      immed = catalog.slice('dcdFileList', split)
      return immed

    def execute(self, i):

      # TODO: Better Job ID Mgmt, for now hack the filename
      i.replace(':', '_').replace('-', '_')
      jobnum = os.path.basename(i).split('.')[0].split('_')[-1]
      logging.debug("jobnum = " + jobnum)
      
      # Fetch (akin to param fetching)
      ldhash = self.catalog.hgetall(i)

      archive = redisCatalog.dataStore(DEFAULT.INDEX_LOCKFILE)
      redis_storage = RedisStorage(archive)

      config = redis_storage.load_hash_configuration('rbphash')
      if not config:
        logging.ERROR("LSHash not configured")
        #TODO: Gracefully exit
  
      # Create empty lshash and load stored hash
      lshash = RandomBinaryProjections(None, None)
      lshash.apply_config(config)

      engine = nearpy.Engine(self.data['indexSize'], 
            lshashes=[lshash], 
            storage=redis_storage)

      for key, value in ldhash:
        neigh = engine.neighbours(value)
        if len(neigh) == 0:
          logging.info ("Found no near neighbors for %s", key)
        else:

          # For now, just grab top NN
          nnkey = neigh[0][1]
          #Back-project
          coord = backProject(getFrame(nnkey))
          jc = writePDB(coord)  #and other paramdata
          self.data['JCQueue'].append(jc)




if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--workinput')
  parser.add_argument('-i', '--init', action='store_true')
  parser.add_argument('-a', '--archive')
  args = parser.parse_args()


  sampleSimJobCandidate = dict(
    psf     = DEFAULT.PSF_FILE,
    pdb     = DEFAULT.PDB_FILE,
    forcefield = DEFAULT.FFIELD,
    runtime = 2000)

  initParams = {'simmd:0001':sampleSimJobCandidate}

  schema = dict(  
        JCQueue = list(initParams.keys()),
        JCComplete = 0,
        JCTotal = len(initParams),
        simSplitParam =  1, 
        dcdFileList =  [], 
        processed =  0,
        indexSize = 852*DEFAULT.NUM_PCOMP,
        anlSplitParam =  1,
        omega =  [0, 0, 0, 0],
        omegaMask = [False, False, False, False],
        converge =  0.)

  threads = {'anlmd': analysisJob(schema, __file__)}


  registry = redisCatalog.dataStore('redis.lock')


  mt = analysisJob(schema, __file__)
  mt.setCatalog(redisCatalog.dataStore('redis.lock'))

  archive = redisCatalog.dataStore(DEFAULT.INDEX_LOCKFILE)

  if args.init:
    logging.debug("Initializing the archive.....")
    initialize(archive)
    sys.exit(0)


  if args.archive:
    logging.info("Running Analysis on " + args.archive)
    mt.execute(args.archive)

  # mt.manager(fork=True)
