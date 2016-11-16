import pickle, string, os, logging, argparse, sys
import itertools as it
from collections import defaultdict, OrderedDict
import datatools.datareduce as DR
import datatools.lattice as lat
import mdtraj as md
import redis
import numpy as np
import numpy.linalg as LA
from datetime import datetime as dt

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


home = os.getenv('HOME')
getseq  = lambda r: [x[0] for x in sorted(r.hgetall('anl_sequence').items(), key=lambda x: int(x[1]))] 
getjc   = lambda r: [r.hgetall('jc_'+i) for i in getseq(r)]

parser = argparse.ArgumentParser()
parser.add_argument('support', type=int)
args = parser.parse_args()
support = args.support

logging.basicConfig(format='%(message)s', filename=home + '/work/latt_intrinsics/latt_%d.log'%support, level=logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

logging.info("SUPPORT   :  %d", support)

cutoff = 8.

DS = 10*np.load('../data/de_ds_mu.npy')
CM = (DS<cutoff)
Kr = [2, 52, 56, 60, 116, 258, 311, 460, 505, 507, 547, 595, 640, 642, 665, 683, 728, 767, 851, 1244, 1485, 1629, 1636]
CMr, Dr = CM[:,Kr], DS[:,Kr]
logging.info('FINAL Input Matrix:  %s', Dr.shape)
logging.info('Reduction Rate:  %7.4f', CMr.sum()/np.multiply(*CM.shape))

MFIS, low_fis = lat.maxminer(CMr, support)
dlat, Ik = lat.derived_lattice(MFIS, Dr, CMr)
logging.info('LATTICE,mfis,%d', len(MFIS))
logging.info('LATTICE,lfis,%d', len(low_fis))
logging.info('LATTICE,dlat,%d', len(dlat))
logging.info('LATTICE,iset,%d', len(Ik))
logging.info('LATTICE,edges,%d', sum([len(v) for v in Ik.values()]))
logging.info('SIZE,mfis,%d', sys.getsizeof(MFIS))
logging.info('SIZE,lfis,%d', sys.getsizeof(low_fis))
logging.info('SIZE,dlat,%d', sys.getsizeof(dlat))
logging.info('SIZE,iset,%d', sys.getsizeof(Ik))
logging.info('\n ALL DONE! Pickling Out')

outfile = open(home + '/work/latt_intrinsics/iset_%d.p'%support, 'wb')
pickle.dump(Ik, outfile)
outfile.close()
logging.info('Iset written')

outfile = open(home + '/work/latt_intrinsics/mfis_%d.p'%support, 'wb')
pickle.dump(MFIS, outfile)
outfile.close()
logging.info('MFIS written')

outfile = open(home + '/work/latt_intrinsics/lowfis_%d.p'%support, 'wb')
pickle.dump(low_fis, outfile)
outfile.close()
logging.info('LFIS written')

outfile = open(home + '/work/latt_intrinsics/dlat_%d.p'%support, 'wb')
pickle.dump(dlat, outfile)
outfile.close()
logging.info('DLAT written')

logging.info('\n Pickling Complete')