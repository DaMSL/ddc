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
from common import *
from macrothread import macrothread
from slurm import slurm

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)



def initialize():

  pass


def generateNewJC(rawfile, frame=-1):


    # Get a new uid
    jcuid = getUID()

    # Write out coords (TODO: should this go to catalog or to file?)
    newCoordFile = os.path.join(DEFAULT.COORD_FILE_DIR, '%s.pdb' % jcuid)

    # Retrieve referenced file from storage
    #   TODO: Set up historical archive for retrieval (this may need to be imported)
    traj  = md.load(rawfile)

    #  If no frame ref is provided, grab the middle frame
    #  TODO:  ID specific window reference point
    if frame < 0:
      frame = traj.n_fames // 2
    coord = traj.slice(frame)
    coord.save_pdb(newCoordFile)

    newsimJob = dict(
        psf     = DEFAULT.PSF_FILE,
        pdb     = newCoordFile,
        forcefield = DEFAULT.FFIELD,
        runtime = 100000)

    self.data['JCQueue'].apend(dict(jcuid=newsimJob))

    return jcuid



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

      self.modules.extend(['redis'])


    def term(self):
      # For now
      return False

    def split(self):
      catalog = self.getCatalog()

      # TODO:  Back Trace
      split = 1  #int(self.data['simSplitParam'])
      immed = catalog.slice('LDIndexList', split)
      return immed

    def execute(self, i):
      logging.debug('CTL MT. Input = ' + i)

      # Fetch all indices
      ld_index = {k.decode():v.decode() for k, v in self.catalog.hgetall(i).items()}

      archive = redisCatalog.dataStore(**archiveConfig)
      redis_storage = RedisStorage(archive)

      config = redis_storage.load_hash_configuration('rbphash')
      if not config:
        logging.error("LSHash not configured")
        #TODO: Gracefully exit
  
      # Create empty lshash and load stored hash
      lshash = RandomBinaryProjections(None, None)
      lshash.apply_config(config)

      engine = nearpy.Engine(self.data['indexSize'], 
            lshashes=[lshash], 
            storage=redis_storage)

      for key, v in ld_index.items():

        # *************  HERE  PACK/UNPACK data !!!!!!
        value = np.array(v)
        logging.debug("VAL-> %s" % type(v))

        logging.info('  VALUE:   Shape=%s,  Type=%s', str(value.shape), str(value.dtype))

        neigh = engine.neighbours(value)
        if len(neigh) == 0:
          logging.info ("Found no near neighbors for %s", key)
        else:

          # For now, just grab top NN
          logging.info ("Found %d neighbours:", len(neigh))

          for n in neigh:
            logging.info ("    NN:  %s   dist = %f", n[1], n[2])


          nnkey = neigh[0][1]
          trajNum, seqNum = decodeLabel(nnkey)

          # Back-project  <--- Move to separate Function tied to decision history
          archiveFile = os.path.join(DEFAULT.WORK, 'bpti-all-%s.dcd' % trajNum)
          frameRef = int(seqNum) * DEFAULT.HIST_SLIDE + (DEFAULT.HIST_WINDOW // 2)
          newJC = generateNewJC(archiveFile, frameRef)

          self.data['JCQueue'].append(newJC)




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-w', '--workinput')
  parser.add_argument('-i', '--init', action='store_true')
  parser.add_argument('-d', '--debug')
  args = parser.parse_args()

  registry = redisCatalog.dataStore('catalog')
  archive = redisCatalog.dataStore(**archiveConfig)

  mt = controlJob(schema, __file__)
  mt.setCatalog(registry)

  if args.init:
    logging("Nothing to intialize for control")
    sys.exit(0)


  if args.debug:
    logging.info("Running Analysis on " + args.debug)
    mt.worker(args.debug)
    sys.exit(0)

  if args.manager:
    mt.manager(fork=False)
  else:
    mt.worker(args.workinput)
