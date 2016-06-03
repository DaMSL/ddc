import dateutil.parser as du
import os
import bench.db as db

home = os.getenv('HOME')


elist =  [i[0] for i in db.runquery('select expname from expr')]

def parse_dblog(name):

  # Ensure log exists
  logfile = home+'/work/%s.log' % name
  if not os.path.exists(logfile):
    logfile = home+'/work/db/%s.log' % name
  if not os.path.exists(logfile):
    print("LOG FILE NOT FOUND: ", name)
    return None

  # Check db file and get its size (if exists)
  dbfile = home+'/work/%s.rdb' % name
  if not os.path.exists(dbfile):
    dbfile = home+'/work/db/%s.rdb' % name
  if not os.path.exists(dbfile):
    print("LOG FILE NOT FOUND: ", name)
    dbsize = 0
  else:
    dbsize = os.path.getsize(dbfile)

  # Read log
  print('Parsing: ', name)
  with open (logfile) as src:
    log_raw = src.read().strip().split('\n')
    rlog = [i for i in log_raw if len(i) > 0 and i[0].isdigit()]

  # Parse lines
  pidlog = {}
  for i in rlog:
    idx = i.find(':')
    if idx < 0:
      continue
    p, msg = int(i[:idx]), i[idx+1:]
    if msg[0] not in ['S', 'M'] or not msg[3].isdigit():
      continue
    entry = (i[idx+1], du.parse(i[idx+3:idx+22]), i[idx+25:])
    if p not in pidlog:
      pidlog[p] = []
    pidlog[p].append(entry)

  # Extract load and handover times
  loadlist = []
  handlist = []
  for p, mlist in pidlog.items():
    start_slave = 0
    htime = 0
    handover = False
    load = 0
    for r, t, m in mlist:
      if 'DB loaded' in m:
        x = m.split()
        load = float(x[-2])
      if 'SLAVE OF' in m:
        start_slave = t
      if 'MASTER MODE' in m:
        handover = True
        htime = (t-start_slave).total_seconds()
    if load > 0:
      loadlist.append(load)
    if handover:
      handlist.append(htime)

  output = dict(size=dbsize, load=loadlist, handover=handlist)
  return output

def parse_all():
  output = {}
  for e in elist:
    logs = parse_dblog(e)
    if logs:
      output[e] = logs
  return output