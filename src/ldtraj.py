import pickle, string, os, logging, argparse
import itertools as it
from collections import defaultdict, OrderedDict
import datatools.datareduce as DR
import mdtraj as md
import redis
import numpy as np
import numpy.linalg as LA
from datetime import datetime as dt

home = os.getenv('HOME')
getseq  = lambda r: [x[0] for x in sorted(r.hgetall('anl_sequence').items(), key=lambda x: int(x[1]))] 
getjc   = lambda r: [r.hgetall('jc_'+i) for i in getseq(r)]

parser = argparse.ArgumentParser()
parser.add_argument('port', type=int)
parser.add_argument('--start', type=int, default=0)
args = parser.parse_args()
port = args.port
startnum = args.start
logging.info("PORT   :  %d", port)

cent = np.load(home + '/ddc/data/bpti-alpha-dist-centroid.npy')
red = redis.StrictRedis(host='login-node02', port=port, decode_responses=True)
expname = red.get('name')
logging.info("EXPNAME:  %s", expname)

jclist = getjc(red)
logging.info("# JOBS :  %d", len(jclist))
topo   = md.load(jclist[0]['pdb'])
pfilt  = topo.top.select('protein')
afilt  = topo.top.select_atom_indices('alpha')
topo_p = topo.atom_slice(pfilt)

wkdir = home+'/work/results/{0}'.format(expname)
if not os.path.exists(wkdir):
  os.mkdir(wkdir)

jc_id = [jc['name'] for jc in jclist]
with open(wkdir + '/jclist', 'w') as out:
  for jc in jc_id:
    out.write('%s\n' % jc)
logging.info("JC_ID  :  %s", wkdir + '/jclist')

all_label = np.zeros(shape=(len(jclist), 4500))
sumlog = open(home + '/work/results/trajout_{0}'.format(expname), 'w')
for i, jc in enumerate(jclist[startnum:]):
  if 'dcd' not in jc:
    jc['dcd'] = jc['pdb'].replace('dcd', 'pdb')
  if not os.path.exists(jc['dcd']):
    logging.info("%d %s NO_DCD", i, jc['name'])
    continue
  tr = md.load(jc['dcd'], top=topo, stride=4)
  if tr.n_frames < 4503:
    logging.info("%d %s Frames: %d", i, jc['name'], tr.n_frames)
    continue
  prot  = tr.atom_slice(pfilt)
  prot.save_dcd(wkdir + '/%s_%03d.dcd' % (expname, (i + startnum)))
  alpha = prot.atom_slice(afilt)
  ds    = DR.distance_space(alpha)
  rms   = [LA.norm(cent-i, axis=1) for i in ds]
  label = np.array([np.argmin(i) for i in rms[3:]])
  all_label[i] = label
  label_str = ''.join([str(i) for i in label])
  red.hset('basin:' + jc['name'], 'label', label_str)
  bincnt = ','.join([str(i) for i in np.bincount(label, minlength=5)])
  src_basin = jc['src_basin'] if 'src_basin' in jc else 'NONE'
  logging.info("%d,%s,%s,%s", i, jc['name'], src_basin, bincnt)
  sumlog.write('%d,%s,%s,%s\n' % (i, jc['name'], src_basin, bincnt))

sumlog.close()
np.save(home + '/work/results/label_{0}'.format(expname), all_label)
logging.info('ALL Done!')



sys.exit(0)

# all_label = np.zeros(shape=(len(jclist), 4500))
# sumlog = open(home + '/work/results/trajout_{0}'.format(expname), 'w')
# for i, jc in enumerate(jclist[startnum:]):
#   tr = md.load(wkdir + '/%s_%03d.dcd' % (expname, (i + startnum)), top=topo_p)
#   alpha = tr.atom_slice(afilt)
#   ds    = DR.distance_space(alpha)
#   rms   = [LA.norm(cent-i, axis=1) for i in ds]
#   label = np.array([np.argmin(i) for i in rms[3:]])
#   all_label[i] = label
#   label_str = ''.join([str(i) for i in label])
#   bincnt = ','.join([str(i) for i in np.bincount(label, minlength=5)])
#   src_basin = jc['src_basin'] if 'src_basin' in jc else 'NONE'
#   logging.info("%d,%s,%s,%s", i, jc['name'], src_basin, bincnt)
#   sumlog.write('%d,%s,%s,%s\n' % (i, jc['name'], src_basin, bincnt))

# sumlog.close()

