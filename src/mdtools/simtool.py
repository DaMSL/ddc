"""
Methods to support simulation set up and execution
"""
import argparse
import sys
import os

import mdtraj as md

from core.common import *
import mdtools.deshaw as deshaw

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

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

def generateNewJC(trajectory, topofile=deshaw.TOPO, parmfile=deshaw.PARM, jcid=None):
    """Creates input parameters for a new simulation. A source trajectory
    with starting coordinates is needed along with a topology/force field files
    (defaults are provided from DEShaw).
    """
    config = systemsettings()
    logging.debug("Generating new simulation coordinates from:  %s", str(trajectory))
    # Get a new uid (if needed)
    jcuid = getUID() if jcid is None else jcid

    # Prep file structure
    jobdir = os.path.join(config.JOBDIR,  jcuid)
    coordFile  = os.path.join(jobdir, '%s_coord.pdb' % jcuid)
    newPdbFile = os.path.join(jobdir, '%s.pdb' % jcuid)
    newPsfFile = os.path.join(jobdir, '%s.psf' % jcuid)
    if not os.path.exists(jobdir):
      os.makedirs(jobdir)
    # Save this as a temp file to set up simulation input file
    trajectory.save_pdb(coordFile)

    # Create new params
    newsimJob = dict(workdir=jobdir,
        coord   = coordFile,
        pdb     = newPdbFile,
        psf     = newPsfFile,
        topo    = topofile,
        parm    = parmfile)

    # run PSFGen to create the NAMD config file
    logging.info("  Running PSFGen to set up simulation pdf/pdb files.")
    stdout = executecmd(psfgen(newsimJob))
    logging.debug("  PSFGen COMPLETE!!\n")
    os.remove(coordFile)
    del newsimJob['coord']
    return jcuid, newsimJob

def generateExplJC(basin, jcid=None):
    """Creates input parameters for a running explcit solvenet simulation.
    """
    config = systemsettings()
    logging.debug('BASIN:  %s', basin)
    logging.debug("Generating new simulation coordinates from:  %s", str(basin['id']))
    # Get a new uid (if needed)
    jcuid = getUID() if jcid is None else jcid

    # Prep file structure
    jobdir = os.path.join(config.JOBDIR,  jcuid)
    pdbfile = coordfile = basin['pdbfile']

    if not os.path.exists(jobdir):
      os.makedirs(jobdir)

    # Create new params
    newsimJob = dict(workdir=jobdir,
        coord   = coordfile,
        pdb     = pdbfile,
        src_traj = basin['traj'],
        src_basin = basin['id'])

    return jcuid, newsimJob


def getSimParameters(state, origin='gen'):
    """ Load standard set of sim parameters from
        the current global state and return a dict 
        for processing into a new job candidate simulation
    """
    settings = systemsettings()
    keys = settings.sim_params.keys()
    params = {}
    for k in keys:
        if k not in keys:
            logging.error('ERROR. Simulation Param, %s, is not loaded into state')
            return
        params[k] = state[k]

    params['psffile'] = os.path.join(settings.workdir, params['psffile'])
    params['ffield_dir'] = os.path.join(settings.workdir, params['ffield_dir'])
    params['interval'] = int(int(params['dcdfreq']) * float(params['sim_step_size']))
    params['gc'] = 1
    params['origin'] = origin
    params['application'] = settings.name

    return params
