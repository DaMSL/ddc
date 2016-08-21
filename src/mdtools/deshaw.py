"""
Methods to support analysis and operators working on/with DEShaw BPTI data
""" 

import os
from collections import namedtuple
import logging
import math

import mdtraj as md
from mdtraj.formats.pdb.pdbstructure import PdbStructure
import numpy as np
from numpy import linalg as LA

from core.common import *
from datatools.datareduce import *

__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2016, Data Driven Control"
__version__ = "0.1.1"
__email__ = "ring@cs.jhu.edu"
__status__ = "Development"

logging.basicConfig(level=logging.DEBUG)

# Predefined topology and parameters for DEShaw BPTI
TOPO  = os.getenv('HOME') +  "/bpti/amber/top_all22_prot.inp"
PARM  = os.getenv('HOME') +  "/bpti/amber/par_all22_prot.inp"

# TODO:  Make a Raw Archiver class (& link in with data mover)
# Hard Coded for now
RAW_ARCHIVE = os.getenv('HOME') + '/work/bpti'
PDB_ALL    = RAW_ARCHIVE + '/bpti-all.pdb'
PDB_PROT   = RAW_ARCHIVE + '/bpti-prot.pdb'

DCD_ALL  = lambda i: RAW_ARCHIVE + '/bpti-all-' + \
  '%03d.dcd'%i if int(i) < 1000 else '%04d.dcd'%i
DCD_PROT = lambda i: RAW_ARCHIVE + '/bpti-prot-%02d.dcd' % i

label =namedtuple('window', 'time state')

#  GLOBAL Topology and Filter data
# TODO: Move to class
if os.path.exists(PDB_ALL):
  topo_all = md.load(PDB_ALL)

topo_prot = md.load(PDB_PROT)
topo = topo_prot

FILTER = {
      'minimal'  : topo.top.select_atom_indices('minimal'),
      'alpha'    : topo.top.select_atom_indices('alpha'),
      'heavy'    : topo.top.select('name =~ "[C.,N.,O.]"'),
      'backbone' : topo.top.select('backbone'),
      'protein'  : topo.top.select('protein')
    }

# OLD HEAVY:  prot.top.select('name =~ "[C.,N.,O.]"')



topo_alpha = topo_prot.atom_slice(FILTER['alpha'])




def convert_topology(src_filename, set_backbone=True, in_place=False, split_dir=None):
  """ Converts the D.E.Shaw topology to Charm-usable force field with
  TIP4P water molecules. THis re-sorts the atoms in the water solvent chains
  (C2,C3) to group all 4 atoms as one contiguous residue. It also aliases the
  ion charges 'pseu' atom to 'OM' for use with the standard tip4p.par parameter
  file. The set_backbone option is to set constraint during initial minimzation"""

  # Grab unit cell description (should be on first few lines:
  cryst = None
  with open(src_filename) as src:
    for line in src.readlines():
      if line.startswith('CRYST1'):
        cryst = line
        break

  # Read in source PDB (DEShaw original format)
  src_pdb = PdbStructure(open(src_filename))
  atoms = list(src_pdb.iter_atoms())
  topo = md.load(src_filename).top

  # Break into 4 segments
  segment_list = ['C1', 'C2', 'C3', 'C4']
  segment = {l:[] for l in segment_list}
  for i in atoms: 
    segment[i.segment_id].append(i)

  # Set temperature factor (for gradual heating) 
  if set_backbone:
    backbone = topo.select("backbone")
    for i in range(0, len(segment['C1'])):
      if i in backbone:
        segment['C1'][i].location.temperature_factor = 1.0

  # Resort water segements and alias "pseu" to OM (tip4p forcefield)
  for wat in ['C2', 'C3']:
    segment[wat] = sorted(segment[wat], key = lambda i: i.residue_number)
    start_serial_num = min(segment[wat], key= lambda i: i.serial_number)
    for i in range(0, len(segment[wat])):
      newsn = i + start_serial_num.serial_number
      segment[wat][i].serial_number = newsn
      if segment[wat][i].get_name == 'pseu':
        segment[wat][i].set_name_with_spaces(' OM ')

  # FOR RE-RUNNING THE PSFGEN
  if split_dir is not None:
    for s in segment_list:
      with open(split_dir + '/%s.pdb' % s, 'w') as dest:
        for atom in segment[s]:
          _=dest.write(str(atom) + '\n')

  # Writeout new file
  if in_place:
    dest = open(src_filename, 'w')
    if cryst is not None:
      dest.write(cryst)
    for s in segment_list:
      for atom in segment[s]:
        _=dest.write(str(atom) + '\n')
    _=dest.write('END')
    dest.close()



def reset_pdb(src_filename):
  """ Updates PDB file with crytalline unit cell data and backbone temp
  control to enable gradual heating"""

  # Standard unit cell description (first lines)
  cryst= 'CRYST1   51.263   51.263   51.263  90.00  90.00  90.00 P 1           1\n'

  # Read in source PDB (DEShaw original format)
  src_pdb = PdbStructure(open(src_filename))
  atoms = list(src_pdb.iter_atoms())
  topo = md.load(src_filename).top

  # Set temperature factor (for gradual heating) 
  backbone = topo.select("backbone")
  for i in range(0, len(atoms)):
    if i in backbone:
      atoms[i].location.temperature_factor = 1.0

  # Write out to file
  dest = open(src_filename, 'w')
  if cryst is not None:
    dest.write(cryst)
  dest.write('REMARK Dynamically Generated from data driven controller (DDC)\n')
  for atom in atoms:
    _=dest.write(str(atom) + '\n')
  _=dest.write('END')
  dest.close()





def atomfilter(filt):
  global atom_filter, topo

  # TODO:  handle more complicated queries (or pass thru)
  if filt in atom_filter.keys():
    return atom_filter[filt]
  else:
    return None

def indexToRef(index, scale=40):
  numFramePerFile
  numFiles = scale

def loadLabels(fn=None):
  """Load all pre-labeled states from DEShaw (as named Tuple)
  """
  if fn is None:
    fn = os.path.join(os.getenv('HOME'), 'ddc', 'bpti_labels_ms.txt')
  label =namedtuple('window', 'time state')
  win = []
  with open(fn) as f:
    for line in f.readlines():
      t, s = line.split()
      win.append(label(float(t), int(s)))
  return win

def loadlabels_aslist(filename=None):
  """Load all pre-labeled states from DEShaw (as list of int)
  """
  if filename is None:
    filename = os.path.join(os.getenv('HOME'), 'ddc', 'data', 'bpti_labels_ms.txt')
  with open(filename) as src:
    lines = src.read().strip().split('\n')
  label = [int(l.split()[1]) for l in lines]
  # Account for the last one:
  label.append(label[-1])
  return label

def getLabelList(labels):
  """Get list of states (future: for dynamic state discovery)
  """
  labelset = set()
  for lab in labels:
    labelset.add(lab.state)
  return sorted(list(labelset))

def load_all_traj():
  """Load all pre-labeled states from DEShaw (as list of int)
  """
  pdb='/bpti/bpti-prot/bpti-prot.pdb'
  dcd = lambda x: '/bpti/bpti-prot/bpti-prot-%02d.dcd' % x
  tr = []
  for i in range(11):
    print ('loading ', i)
    start = dt.datetime.now()
    tr.append(md.load(DCD_ALL(i), top=PDB_ALL))
    end = dt.datetime.now()
    print((end-start).total_seconds())
  return tr

def load_all_traj_skip(skip=200):
  """Load all pre-labeled states from DEShaw (as list of int)
  """
  pdb='/bpti/bpti-prot/bpti-prot.pdb'
  dcd = lambda x: '/bpti/bpti-prot/bpti-prot-%02d.dcd' % x
  tr = []
  for i in range(42):
    print ('loading ', i)
    tr.append(md.load(DCD_PROT(i), top=PDB_PROT, stride=skip))
  return tr


def loadpts(skip=40, filt=None, ref_frame=None):
  """Loads all DEShaw Points as one long NDarray. Skip value is used 
  to load fewer frames
  """
  pts = []
  for i in range(42):
    print('loading file: ', i)
    if filt is not None:
      traj = md.load(DCD_PROT(i), top=PDB_PROT, atom_indices=filt, stride=skip)
    else:
      traj = md.load(DCD_PROT(i), top=PDB_PROT, stride=skip)
    traj.center_coordinates()
    if ref_frame is not None:
      traj.superpose(ref_frame)
    for i in traj.xyz:
      pts.append(i)
  return np.array(pts)

def loadpts_traj(tnum, skip=40, filt=None):
  """Loads all DEShaw Points as one long NDarray. Skip value is used 
  to load fewer frames
  """
  pts = []
  print('loading file: ', tnum)
  traj = md.load(DCD_PROT(tnum), top=PDB_PROT, stride=skip)
  if filt is not None:
    traj.atom_slice(filt, inplace=True)
  return traj.xyz


def loadpts_trajlist(trajlist, skip=40, filt=None):
  """Loads all DEShaw Points as one long NDarray. Skip value is used 
  to load fewer frames
  """
  pts = []
  for i in trajlist:
    print('loading file: ', i)
    traj = md.load(DCD_PROT(i), top=PDB_PROT, stride=skip)
    if filt is not None:
      traj.atom_slice(filt, inplace=True)
    for i in traj.xyz:
      pts.append(i)
  return np.array(pts)

def refFromIndex(index):
  """Convert a single frame index to (fileno, frame) tuple
  """
  fileno = math.floor(index // 100000)
  frame  = index % 100000
  return fileno, frame

def getDEShawfilename(seqnum, fullpath=False):
    filename = 'bpti-all-%03d.dcd' if int(seqnum) < 1000 else 'bpti-all-%04d.dcd'
    if fullpath:
      filename = os.path.join(RAW_ARCHIVE, filename)
    return filename

def getDEShawfilename_prot(seqnum, fullpath=False):
    filename = 'bpti-prot-%02d.dcd' % seqnum
    if fullpath:
      filename = os.path.join(RAW_ARCHIVE, filename)
    return filename


def getHistoricalTrajectory(seqnum):
    fname = getDEShawfilename(seqnum)
    dfile = os.path.join(RAW_ARCHIVE, fname % int(seqnum))
    pfile = PDB_ALL
    return pfile, dfile

def getHistoricalTrajectory_prot(seqnum):
    fname = getDEShawfilename_prot(seqnum)
    dfile = os.path.join(RAW_ARCHIVE, fname)
    pfile = PDB_PROT
    return pfile, dfile


def labelDEShaw_rmsd(filename='bpti-rmsd-alpha-dspace.npy'):
  """label ALL DEShaw BPTI observations by state & secondary state (A, B)
  Returns frame-by-frame labels  (used to seed jobs)
  """
  settings = systemsettings()
  logging.info('Loading Pre-Calc RMSD Distances from: %s ','bpti-rmsd-alpha-dspace.npy')
  rms = np.load(filename)
  prox = np.array([np.argsort(i) for i in rms])
  theta = settings.RMSD_THETA
  logging.info('Labeling All DEShaw Points. Usng THETA=%f', theta)
  rmslabel = []
  for i in range(len(rms)):
    A = prox[i][0]
    proximity = abs(rms[i][prox[i][1]] - rms[i][A])    #abs
    B = prox[i][1] if proximity < theta else A
    rmslabel.append((A, B))
  return rmslabel


def loadDEShawTraj(start, end=-1, filt='heavy'):
  if end == -1:
    end = start +1
  trajectory = None
  for seqnum in range(start, end):
    pdbfile, dcdfile = getHistoricalTrajectory(seqnum)
    if not os.path.exists(dcdfile):
      logging.info('%s   File not exists. Continuing with what I got', dcdfile)
      break
    logging.info("LOADING:  %s", os.path.basename(dcdfile))
    traj = md.load(dcdfile, top=pdbfile)
    traj.atom_slice(traj.top.select_atom_indices(selection=filt), inplace=True)
    trajectory = trajectory.join(traj) if trajectory else traj
  return trajectory

DEShawReferenceFrame = None
def deshawReference(atomfilter='heavy'):
  global DEShawReferenceFrame
  if DEShawReferenceFrame:
    return DEShawReferenceFrame
  pdbfile, dcdfile = getHistoricalTrajectory(0)
  traj = md.load(dcdfile, top=pdbfile, frame=0)
  filt = traj.top.select_atom_indices(selection=atomfilter)
  traj.atom_slice(filt, inplace=True)
  DEShawReferenceFrame = traj
  return traj



def psfgen(params):

  force_field_list = '/top_all22_prot.inp', '/tip4p.top'
  for f in force_field_list:
    if not os.path.exists(params['ffield_dir'] + f):
      logging.error('Missing Protein CHARMM Force field:  %s', params['ffield_dir'] + f)
      raise FileNotFoundError

  seg_list = ['/C1.pdb', '/C2.pdb', '/C3.pdb', '/C4.pdb']
  for s in seg_list:
    if not os.path.exists(params['tmploc'] + s):
      logging.error('Missing Source Segement PDB File:  %s', params['tmploc'] + s)
      raise FileNotFoundError

  return """psfgen << ENDMOL

# 1. Load Topology File
topology %(ffield_dir)s/top_all22_prot.inp

# 2. Load Protein
segment C1 {pdb %(tmploc)s/C1.pdb}

# 3. Patch protein segment
patch DISU C1:5 C1:55
patch DISU C1:14 C1:38
patch DISU C1:30 C1:51

# 4. Define aliases
pdbalias atom ILE CD1 CD ;
pdbalias atom ALA H HN ;
pdbalias atom ALA OXT O ;
pdbalias atom ARG H HN ;
pdbalias atom ARG H2 HN;
pdbalias atom ARG H3 HN;
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
coordpdb %(tmploc)s/C1.pdb C1


# 6. Build Water Segement (HOH)
topology %(ffield_dir)s/tip4p.top
pdbalias residue HOH TIP4 ;
pdbalias atom HOH O OH2 ;
pdbalias atom HOH pseu OM;
segment C2 {
 auto none
 pdb %(tmploc)s/C2.pdb
}
coordpdb %(tmploc)s/C2.pdb C2 ;

# 7.  Build Water Segment #2 (SPC)
pdbalias residue SPC TIP4 ;
pdbalias atom SPC O OH2 ;
pdbalias atom SPC pseu OM;
segment C3 {
 auto none
 pdb %(tmploc)s/C3.pdb
}
coordpdb %(tmploc)s/C3.pdb C3 ;

# 8.  Add in Chlorine
pdbalias residue CL CLA;
pdbalias atom CLA CL CLA ;
segment C4 {
 auto none
 pdb %(tmploc)s/C4.pdb
}
coordpdb %(tmploc)s/C4.pdb C4

guesscoord

# 9. Output psf/pdb files
writepsf %(workdir)s/%(name)s.psf
writepdb %(workdir)s/%(name)s.pdb

ENDMOL""" %  params



def calc_bpti_centroid(traj_list):
  """Calculate the centroids for a list of BPTI trajectories from DEShaw
  This is assumed to use the pre-labeled set of trajectories for
  conform grouping (by state) and subsequent average centroid location calc

  Current implementation assumes distance space (with alpha-filter)
  """
  sums = np.zeros(shape=(5, 1653))
  cnts = [0 for i in range(5)]
  label = getLabelList()
  for n, traj in enumerate(prdist):
    for i in range(0, len(traj), 40):
      try:
        idx = (n*400)  + (i // 1000)
        state = label[idx]
        # Exclude any near transition frames
        if idx < 3 or idx > 4121:
          continue
        if state == label[idx-2] == label[idx-1] == label[idx+1] == label[idx+2]:
          sums[state] += traj[i]
          cnts[state] += 1
      except IndexError as err:
        pass # ignore idx errors due to tail end of DEShaw data
  cent = [sums[i] / cnts[i] for i in range(5)]
  return (np.array(cent))

def check_bpti_rms(observations, centroid, skip=40):
  hit = 0
  miss = 0
  for n, pt in enumerate(observations):
    if n % 10000 == 0:
      print ('checking pt #', n)
    idx = math.floor(n/1000)
    dist = [(LA.norm(pt - C)) for C in centroid]
    prox = np.argsort(dist)
    predicted_state = np.argmin(dist)
    if labeled_state == predicted_state:
      hit += 1
    else:
      miss += 1
  print ('Hit rate:  %5.2f  (%d)' % ((hit/(hit+miss)), hit))
  print ('Miss rate: %5.2f  (%d)' % ((miss/(hit+miss)), miss))




def load_prot():
  desh = [dr.filter_alpha(md.load(dedcd%i, top=depdb)) for i in range(40)]
  demid = []
  for tr in desh:
    for i in range(500, tr.n_frames, 1000):
      demid.append(tr.xyz[i])


if __name__ == '__main__':
  #  FOR Calculting Centroids and RMSD of ALL conforms in D.E.Shaw Dataset
  settings = systemsettings()
  settings.applyConfig('debug.json')
  win = loadLabels()
  win.append(win[-1])  
  mean = []
  w = []
  traj_ref = deshawReference()
  for dcd in range(4125):
    if dcd % 100 == 0:
      logging.debug("Processing dcd file: %d", dcd)
    traj = loadDEShawTraj(dcd, filt='heavy')
    traj.superpose(traj_ref)
    mean.append(np.mean(traj.xyz, axis=0))
  centroid = np.zeros(shape=(5, 454, 3))
  for i in range(5):
    accum = []
    for k in range(len(mean)):
      if win[k].state == i:
        accum.append(mean[k])
    mn = np.mean(np.array(accum), axis=0)
    np.copyto(centroid[i], mn)
  np.save('centroid.npy', centroid)
  centroid = np.load('centroid.npy')
  rmsd = np.zeros(shape=(4125, 1000, 5))
  with open('rms.out', 'a') as rmsfile:
    for dcd in range(4125):
      traj = loadDEShawTraj(dcd, filt='heavy')
      traj.superpose(traj_ref)
      for f in range(len(traj.xyz)):
        rms = np.array([LA.norm(traj.xyz[f]-centroid[k]) for k in range(5)])
        np.copyto(rmsd[dcd][f], rms)
        rmsfile.write('%d;%d;%d;%d;%s\n' %(dcd, f, win[dcd].state, np.argmin(rms),str(rms.tolist())))
  np.save('rmsd', rmsd)


# rms = np.zeros(shape=(len(tr0), 5))
# for i in range(len(tr0)):
#   np.copyto(rms[i], np.array([LA.norm(tr0.xyz[i]-c1[k]) for k in range(5)]))

# r34 = np.zeros(shape=(1000, 5))
# for i in range(1000):
#   np.copyto(r34[i], np.array([LA.norm(tr34.xyz[i]-c1[k]) for k in range(5)]))




# count = np.zeros(5)
# for i in r33: count[np.argmin(i)] += 1
# rms  = []
# for i, r in enumerate(rm):
#   res = r.replace('[','').replace(']','').split(',')
#   rms.append((i//1000, int(res[1]), int(res[2]), int(res[3]), np.array([float(v) for v in res[4:]])))




# rms_ps = []
# for frame in range(len(rms)):
#   lo = max(0, frame-4)
#   hi = min(frame+5, len(rms))
#   ps = np.mean(np.array([r[4] for r in rms[lo:hi]]), axis=0)
#   rms_ps.append((rms[frame][0], rms[frame][1], rms[frame][2], np.argmin(ps), ps))

# counts = {}
# for i in rms_ps:
#   if i[2] != i[3]:
#     if i[0] not in counts.keys():
#       counts[i[0]] = 0
#     counts[i[0]] += 1