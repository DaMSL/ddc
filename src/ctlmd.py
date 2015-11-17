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

# 4. Define aliases
pdbalias atom ILE CD1 CD ;
pdbalias atom ALA H HN ;
#pdbalias atom ALA OXT O ;
pdbalias atom ARG H HN ;
#pdbalias atom ARG H2 HN;
#pdbalias atom ARG H3 HN;
pdbalias atom ARG HB3 HB1 ;
pdbalias atom ARG HD3 HD1 ;
pdbalias atom ARG HG3 HG1 ;
pdbalias atom ASN H HN ;
pdbalias atom ASN HB3 HB1 ;
pdbalias atom ASP H HN ;
pdbalias atom ASP HB3 HB1 ;
pdbalias atom CYS H HN ;
pdbalias atom CYS HB3 HB1 ;
pdbalias atom GLN H HN ;
pdbalias atom GLN HB3 HB1 ;
pdbalias atom GLN HG3 HG1 ;
pdbalias atom GLU H HN ;
pdbalias atom GLU HB3 HB1 ;
pdbalias atom GLU HG3 HG1 ;
pdbalias atom GLY H HN ;
pdbalias atom GLY HA3 HA1 ;
pdbalias atom ILE H HN ;
pdbalias atom ILE HD11 HD1 ;
pdbalias atom ILE HD12 HD2 ;
pdbalias atom ILE HD13 HD3 ;
pdbalias atom ILE HG13 HG11 ;
pdbalias atom LEU H HN ;
pdbalias atom LEU HB3 HB1 ;
pdbalias atom LYS H HN ;
pdbalias atom LYS HB3 HB1 ;
pdbalias atom LYS HD3 HD1 ;
pdbalias atom LYS HE3 HE1 ;
pdbalias atom LYS HG3 HG1 ;
pdbalias atom MET H HN ;
pdbalias atom MET HB3 HB1 ;
pdbalias atom MET HG3 HG1 ;
pdbalias atom PHE H HN ;
pdbalias atom PHE HB3 HB1 ;
pdbalias atom PRO HB3 HB1 ;
pdbalias atom PRO HD3 HD1 ;
pdbalias atom PRO HG3 HG1 ;
pdbalias atom SER H HN ;
pdbalias atom SER HB3 HB1 ;
pdbalias atom SER HG HG1 ;
pdbalias atom THR H HN ;
pdbalias atom TYR H HN ;
pdbalias atom TYR HB3 HB1 ;
pdbalias atom VAL H HN ;

# 5. Read protein coordinates from PDB file & set coords
coordpdb %(coord)s BPTI
guesscoord

# 6. Output psf/pdb files
writepsf %(psf)s
writepdb %(pdb)s

ENDMOL''' % params




def generateNewJC(rawfile, frame=-1):

    logging.debug("Generating new coords from:  %s", rawfile)

    # Get a new uid
    jcuid = getUID()
    # jcuid = 'DEBUG'

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

    newsimJob = dict(workdir=jobdir,
        coord = coordFile,
        pdb     = newPdbFile,
        psf     = newPsfFile,
        topo    = DEFAULT.TOPO,
        parm    = DEFAULT.PARM)

    cmd = psfgen(newsimJob)

    logging.debug("  PSFGen new simulation:\n " + cmd)

    stdout = executecmd(cmd)

    logging.debug("  PSFGen COMPLETE!!  Cleaning up\n" + stdout)

    os.remove(coordFile)

    return jcuid, newsimJob



class controlJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'ctl')
      # State Data for Simulation MacroThread -- organized by state
      self.setInput('LDIndexList')
      self.setTerm('JCComplete', 'processed')
      self.setExec('indexSize', 'JCQueue')
      self.setSplit('anlSplitParam')
      
      # exec incl hash key-name
      # TODO: wildcard loading of data

      self.modules.add('namd')

      #  This thread's execution will run "supervised"
      self.fork = False


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
      ld_index = {k.decode():np.fromstring(v, dtype=np.float64) for k, v in self.catalog.hgetall(wrapKey('idx', i)).items()}
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

          logging.info("New Simulation Job Created: %s", jcID)
          for k, v in jcConfig.items():
            logging.debug("   %s:  %s", k, str(v))

          jckey = wrapKey('jc', jcID)

          self.data['JCQueue'].append(jcID)
          self.catalog.save({jckey: jcConfig})

          logging.info("New JC Complete:  %s" % jcID)
          
          break




if __name__ == '__main__':
  mt = controlJob(schema, __file__)
  mt.run()


  # parser = argparse.ArgumentParser()
  # parser.add_argument('-w', '--workinput')
  # parser.add_argument('-i', '--init', action='store_true')
  # parser.add_argument('-d', '--debug')
  # args = parser.parse_args()

  # registry = redisCatalog.dataStore('catalog')
  # archive = redisCatalog.dataStore(**archiveConfig)

  # mt = controlJob(schema, __file__)
  # mt.setCatalog(registry)


  # if args.workinput:
  #   mt.worker(args.workinput)
  # else:
  #   mt.manager(fork=False)



