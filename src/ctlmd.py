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



def psfgen(params):
  return '''psfgen << ENDMOL

# 1. Load Topology File
topology %(topo)s

# 2. Load Protein
segment BPTI {pdb %(coord)s}

# 3. Patch protein segment
patch DISU BPTI:5 BPTI:55
patch DISU BPTI:14 BPTI:38
patch DISU BPTI:30 BPTI:51

# 4. Read protein coordinates from PDB file
pdbalias atom ILE CD1 CD ; 
coordpdb %(coord)s BPTI

guesscoord

writepsf %(psf)s
writepdb %(pdb)s

ENDMOL''' % params




def generateNewJC(rawfile, frame=-1):

    logging.debug("Generating new coords from:  %s", rawfile)

    # Get a new uid
    jcuid = getUID()
    jcuid = 'DEBUG'

    # Write out coords (TODO: should this go to catalog or to file?)
    # tmpCoord = os.path.join(DEFAULT.COORD_FILE_DIR, '%s_tmp.pdb' % jcuid)
    jobdir = os.path.join(DEFAULT.JOB_DIR,  jcuid)
    coordFile  = os.path.join(jobdir, '%s_coord.pdb' % jcuid)
    newPdbFile = os.path.join(jobdir, '%s.pdb' % jcuid)
    newPsfFile = os.path.join(jobdir, '%s.psf' % jcuid)

    logging.debug("Files to use: %s, %s", coordFile, newPsfFile)


    if not os.path.exists(jobdir):
      os.makedirs(jobdir)

    # Retrieve referenced file from storage
    #   TODO: Set up historical archive for retrieval (this may need to be imported)
    
    #  Load in Historical Referenced trajectory file, filter out proteins & slice
    traj  = md.load(rawfile, top=DEFAULT.PDB_FILE)
    filt = traj.top.select('protein')    
    traj.atom_slice(filt, inplace=True)
    
    #  If no frame ref is provided, grab the middle frame
    #  TODO:  ID specific window reference point
    if frame < 0:
      frame = traj.n_fames // 2
    coord = traj.slice(frame)

    logging.debug("Working source traj: %s", str(coord))

    # Save this as a temp file to set up simulation input file
    coord.save_pdb(coordFile)

    logging.debug("Coord file saved.")

    newsimJob = dict(coord = coordFile,
        pdb     = newPdbFile,
        psf     = newPsfFile,
        topo    = DEFAULT.TOPO,
        parm    = DEFAULT.PARM)

    cmd = psfgen(newsimJob)

    logging.debug("  PSFGen new simulation:\n " + cmd)

    stdout = executecmd(cmd)

    logging.debug("  PSFGen COMPLETE!!\n" + stdout)


    return jcuid, newsimJob



class controlJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'simmd')
      # State Data for Simulation MacroThread -- organized by state
      self.setInput('LDIndexList')
      self.setTerm('JCComplete', 'processed')
      self.setExec('indexSize', 'JCQueue')
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
      ld_index = {k.decode():np.fromstring(v, dtype=np.float64) for k, v in self.catalog.hgetall(i).items()}
      archive = redisCatalog.dataStore(**archiveConfig)
      redis_storage = RedisStorage(archive)

      config = redis_storage.load_hash_configuration(DEFAULT.HASH_NAME)
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
          trajectory, seqNum = nnkey.split(':')

          # Back-project  <--- Move to separate Function tied to decision history
          archiveFile = os.path.join(DEFAULT.RAW_ARCHIVE, '%s.dcd' % trajectory)
          # frameRef = int(seqNum) * DEFAULT.HIST_SLIDE + (DEFAULT.HIST_WINDOW // 2)
          frameRef = int(seqNum) + DEFAULT.HIST_WINDOW // 2
          jcID, params = generateNewJC(archiveFile, frameRef)


          # NOTE: Update Additional JC Params, as needed
          jcConfig = dict(params,
              name    = jcID,
              runtime = 100000,
              temp    = 310)



          self.data['JCQueue'].append(jcID)
          self.catalog.save({jcID: jcConfig})

          logging.info("New JC Complete:  %s" % jcID)
          return




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


  if args.debug:
    logging.info("Running Analysis on " + args.debug)
    mt.worker(args.debug)
    sys.exit(0)

  if args.manager:
    mt.manager(fork=False)
  else:
    mt.worker(args.workinput)
