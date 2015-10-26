import argparse
import sys
import os

import redisCatalog
from common import DEFAULT, executecmd
from macrothread import macrothread
from slurm import slurm

import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


# CHECK NEED HERE
  # WORKDIR = os.environ['HOME'] + '/bpti/'

class EXT:
    rms  = 'rms'
    lmd  = 'lmd'
    map  = 'map'






class analysisJob(macrothread):
    def __init__(self, schema, fname):
      macrothread.__init__(self, schema, fname, 'simmd')

      # self.name = str(dataid)

      # State Data for Simulation MacroThread -- organized by state
      self.setInput('dcdFileList')
      self.setTerm('JCComplete', 'processed')
      self.setExec()
      self.setSplit('anlSplitParam')
 

    def dcd2xyz(self):
      cmd = 'catdcd -o ' + self.xyzfile + ' -otype xyz -stype psf -s ' + self.psf + ' ' + self.dcdfile
      result = executecmd(cmd)
      logging.info(result)

    def inFile(self, ext):
      return '%s.%s' % (self.name, ext)

    def logFile(self, ext=None):
      if ext:
        return 'log/%s.%s.log' % (self.name, ext)
      else:
        return 'log/%s.log' % self.name


    def genInput_RMSD(self):
        # Write the input file for RMDS fortran program
        nneigh=DEFAULT.NUM_NEIGH
        rms = os.path.join(self.workdir, self.inFile(EXT.rms))
        with open(rms, 'w') as input:
          input.write(self.xyzfile + '\n')

          # TODO:  Should all subsets start at frame #1????
          input.write('%d %d\n' % (1, self.size))
          input.write('%d\n' % nneigh)


    def genInput_LMDS(self):
        # Write input file for local_MDS fortran program
        args = dict(size=10, nneigh=5, mds=10)
        lmd = os.path.join(self.workdir, self.inFile(EXT.lmd))
        with open(lmd, 'w') as input:
          input.write(self.xyzfile + '\n')
          input.write(self.xyzfile + '_neighbor\n')

          # TODO:  Should all subsets start at frame #1????
          input.write('%d %d\n' % (1, args['size']))
          input.write('%d\n' % 1000)
      
          # INPUT PARAMS
          input.write('%d %d %d %d\n' % (args['nneigh'], args['mds'], DEFAULT.MDS_START, DEFAULT.MDS_STEP))
          input.write('%f %f %d\n' % (DEFAULT.NOISE_CUT_START, DEFAULT.NOISE_CUT_STEP, DEFAULT.NOISE_CUT_NUM))
          input.write('%d\n' % 4)


    def genInput_LMAP(self):
        # Write input file for local_MDS fortran program
        lmap = os.path.join(self.workdir, self.inFile(EXT.map))
        with open(lmap, 'w') as w:
          
          # Hard code following line -- NOTE: very confusing from source (README is backwards)
          w.write('0 1\n')
          w.write('%d\n' % self.size)
          w.write(self.xyzfile + '_rmsd\n')
          w.write('weight.dat\n')   # This will be ignored
          w.write(self.results + '\n')
          w.write('0.0\n')

          #  MAY NEED TO FIGURE OUT EPS_COL HERE
          w.write('%s %d\n' % (self.epsFile, 18))


    def loadShell(self):
        shell = open(self.sbatchFile, 'w')
        # with open('scripts/slurm_shell.sh', 'r') as slurm:
        #   for line in slurm.readlines():
        #     shell.write(line)
        shell.close()


    def makeScript(self, partition='shared'):

        nnodes=DEFAULT.NODES

        mode = os.stat(self.sbatchFile).st_mode
        mode |= (mode & 0o444) >> 2    # copy R bits to X
        os.chmod(self.sbatchFile, mode)

        cpus = nnodes * DEFAULT.CPU_PER_NODE

        with open(self.sbatchFile, 'w') as shell:
          shell.write('''#!/bin/bash

#SBATCH
#SBATCH --job-name=bpti_%s
#SBATCH --time=720
#SBATCH --nodes=%d
#SBATCH --partition=%s
#SBATCH --ntasks-per-node=%d
#SBATCH --workdir=%s

module use ~/privatemodules

module unload openmpi/intel/1.8.4
module load gcc/4.9.2
module load openblas/0.2.14
module load gmp/6.0
module load mpfr/3.1.3
module load mpc/1.0.3
module load arpack/1.0
module load mpich/ge/gcc/64/3.1
module load lsdmap/4.0
''' % (self.name, nnodes, partition, DEFAULT.CPU_PER_NODE, self.workdir))

          # Convert DCD to XYZ format
          shell.write('\n')
          shell.write('pwd\n')
          shell.write('\n')
          shell.write('echo Convert DCD to XYZ format\n')
          shell.write('catdcd -o ' + self.xyzfile + ' -otype xyz -stype psf -s ' + self.psf + ' ' + self.dcdfile)

          # Call rmds program
          shell.write('\n')
          shell.write('# Call rmds program\n')
          shell.write('mpiexec -n %d p_rmsd_neighbor<%s>%s\n' % (nnodes*DEFAULT.CPU_PER_NODE, self.inFile(EXT.rms), self.logFile(EXT.rms)))

          # Neighbor Files are hard-coded in underlyng fortran
          shell.write('\n')
          shell.write('# Prepare neighbor files for Local MDS\n')
          neighFile = 'neighbor/' + self.name + '.xyz_neighbor'
          shell.write('cat ' + neighFile + '_9* > ' + neighFile +'\n')
          shell.write('rm ' + neighFile + '_9*\n')

          # Call local_mds program 
          shell.write('\n')
          shell.write('# Call local_mds program \n')
          shell.write('mpiexec -n %d p_local_mds<%s>%s\n' % (nnodes*DEFAULT.CPU_PER_NODE, self.inFile(EXT.lmd), self.logFile(EXT.lmd)))

          # EPS Files are hard-coded
          shell.write('\n')
          shell.write('# Collect epsilon values from local MDS\n')
          # epsFile = 'localscale/' + job.epsFile
          epsFile = 'localscale/' + self.epsFile
          shell.write('cat ' + epsFile + '_1* > ' + self.epsFile +'\n')
          shell.write('rm ' + epsFile + '_1*\n')

          # Call wlsdmap program
          shell.write('\n')
          shell.write('# Conduct Eigen decomposition\n')
          shell.write('mpiexec -n %d p_wlsdmap<%s>%s\n' % (nnodes*DEFAULT.CPU_PER_NODE, self.inFile(EXT.map), self.logFile(EXT.map)))

          # Clean Up
          shell.write('\n')
          shell.write('\n')
          shell.write('# Clean up\n')
          for ext in [EXT.rms, EXT.lmd, EXT.map]:
            shell.write('cat ' + self.logFile(ext) + ' >> ' + self.logFile() + '\n')
            shell.write('rm ' + self.logFile(ext) + '\n')
            shell.write('rm ' + self.inFile(ext) + '\n')

          shell.write('\n')
          shell.write('mv %s %s/%s\n' % (self.epsFile, self.workdir, self.name + '.eps'))
          shell.write('mv %s.e* %s\n' % (self.results, self.workdir))
          shell.write('rm neighbor/%s*\n' % self.name)
          shell.write('rm rmsd/%s*\n' % self.name)
          shell.write('rm %s\n' % self.xyzfile)



    def term(self):
      # For now
      return False

    def split(self):
      split = int(self.data['anlSplitParam'])
      catalog = self.getCatalog()
      immed = catalog.slice('JCQueue', split)
      return immed

    def execute(self, i):

      # TODO: Better Job ID Mgmt
      jobnum = i


      # Define analysis config environ for LSDMap Program
      self.workdir    = os.path.join(DEFAULT.WORKDIR, str(jobnum))
      self.dcdfile    = os.path.join(self.workdir, jobnum + '.dcd')
      self.xyzfile    = os.path.join(self.workdir, self.name + '.xyz')
      self.psf        = os.path.join(DEFAULT.WORKDIR, 'bpti.psf')
      self.epsFile    = self.xyzfile + '_eps'
      self.sbatchFile = os.path.join(self.workdir, self.name + '_anl.sh')
      self.results    = self.name
      self.size       = 20          # NEED TO DETERMINE INPUT DATA SIZE!!!!!

      for dname in ['neighbor', 'localscale', 'rmsd', 'log']:
        d = os.path.join(self.workdir, dname)
        if not os.path.exists(d):
          os.mkdir(d)      

      logging.info("Gen RMSD input")
      self.genInput_RMSD()

      logging.info("Gen LMDS input")
      self.genInput_LMDS()

      logging.info("Gen LMAP input")
      self.genInput_LMAP()

      logging.info("Creating sbatch Script")
      self.loadShell()
      self.makeScript()
      logging.info("Script written to: " + self.sbatchFile)

      logging.debug("Scheduling analysis task")
      executecmd("sbatch " + self.sbatchFile)






if __name__ == '__main__':

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
        anlSplitParam =  1,
        omega =  [0, 0, 0, 0],
        omegaMask = [False, False, False, False],
        converge =  0.)

  threads = {'anlmd': analysisJob(schema, __file__)}


  registry = redisCatalog.dataStore('redis.lock')


  mt = analysisJob(schema, __file__)
  mt.setCatalog(redisCatalog.dataStore('redis.lock'))
  mt.manager(fork=True)
