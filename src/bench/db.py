import sqlite3
import sys
import os
import traceback
import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple

def namedtuple_factory(cursor, row):
    """
    Usage:
    con.row_factory = namedtuple_factory
    """
    fields = [col[0] for col in cursor.description]
    Row = namedtuple("Row", fields)
    return Row(*row)

HOME = os.environ['HOME']
DB_FILE = os.path.join(HOME, 'ddc', 'results', 'ddc_data.db')
GRAPH_LOC = os.path.join(HOME, 'ddc', 'graph')

tables = {
'expr':
"""CREATE TABLE IF NOT EXISTS expr (
  expid integer,
  expname text
);
""",
'conv':
"""CREATE TABLE IF NOT EXISTS conv (
  expname text,
  ts integer,
  label text,
  val real
);
""",
'bench_ctl':
"""CREATE TABLE IF NOT EXISTS bench_ctl (
  expid integer,
  ctlid integer,
  num integer,
  runtime real,
  deltatime real,
  label text
);
""",
'ctl':
"""CREATE TABLE IF NOT EXISTS ctl (
  cid PRIMARY KEY AUTOINCREMENT,
  expid integer,
  ctlid integer,
  npts integer,
  chit integer,
  cmiss integer,
  njobs integer,
  FOREIGN KEY(expid) REFERENCES expr(expid)
);
""",
'rms_de':
"""CREATE TABLE IF NOT EXISTS rms_de (
  cid integer,
  s0 integer,
  s1 integer,
  s2 integer,
  s3 integer,
  s4 integer,
  FOREIGN KEY(cid) REFERENCES ctl(cid)
);
""",
'kmeans':
"""CREATE TABLE IF NOT EXISTS kmeans (
  cid PRIMARY KEY AUTOINCREMENT,
  st integer,
  cnt integer,
  totstate integer
);
"""

}

insertion = {
  'expr': "INSERT INTO expr (expid, expname) VALUES (%d, '%s');",
  'conv': "INSERT INTO conv (expname, ts, label, val) VALUES (%s, %d, %s, %f);",
  'bench_ctl': "INSERT INTO bench_ctl (expid, ctlid, num, runtime, deltatime, label) VALUES (%d, %d, %d, %f, %f, '%s');"}

autoid = []

exprRecord = namedtuple('exprRecord', 'expid expname')
convRecord = namedtuple('convRecord', 'expname ts label val')
benchctlRecord = namedtuple('benchctlRecord', 'expid ctlid num runtime deltatime label')

conn = sqlite3.connect(DB_FILE)

# for emp in map(EmployeeRecord._make, cursor.fetchall()):
#     print(emp.name, emp.title)


def getConn():
  global conn
  return conn

def close():
  global conn
  conn.close()

def createTables():
  global conn
  cur = conn.cursor()
  for table, query in tables.items():
    try:
      cur.execute(query)
      conn.commit()
    except Exception as inst:
      print("Failed to create tables:" )
      print(inst)


def dropTable(table):
  global conn
  try:
    cur = conn.cursor()
    query = 'DROP TABLE IF EXISTS %s;' % table
    cur.execute(query)
    conn.commit()
  except Exception as inst:
    print("Failed to drop table:" )
    print(inst)


def dropAllTables():
  for table in tables.keys():
    try:
      query = 'DROP TABLE IF EXISTS %s CASCADE;' % table
      cur.execute(query)
      conn.commit()
    except Exception as inst:
      print("Failed to drop table(s):" )
      print(inst)


def adhoc(query):
  global conn
  try:
    cur = conn.cursor()
    cur.execute(query)
    for row in cur.fetchall():
      print(','.join(str(elm) for elm in row))
  except Exception as inst:
    print("Ad Hoc Query Failed:" )
    traceback.print_exc()

def runquery(query, withheader=False):
  global conn
  try:
    cur = conn.cursor()
    cur.execute(query)
    if withheader:
      names = [d[0] for d in cur.description]
      materialized = [names]
      materialized.extend(cur.fetchall())
      return materialized
    else:
      return cur.fetchall()
  except Exception as inst:
    print("Ad Hoc Query Failed:" )
    traceback.print_exc()

def insert(table, *values):
  try:
    print('TABLE IS ', table)
    query = insertion[table] % values
    print(query)
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    if table in autoid:
      val = int(cur.fetchone()[0])
      return val
  except Exception as inst:
      print("Failed to insert Experiment: ")
      print(inst)

def get_expid(name):
  cur = conn.cursor()
  qry = "SELECT expid FROM expr WHERE expname='%s';" % name
  print(qry)
  cur.execute(qry)
  return int(cur.fetchone()[0])

def loadbenchctl(name):
  srcfile = os.path.join(os.environ['HOME'], 'ddc', 'results', 'bench_ctl_%s.log')  
  try:
    eid = get_expid(name)
    with open(srcfile) as src:
      entry = src.read().strip().split('\n')
      for e in entry:
        cid, n, r, d, l = e.split(',')
        insert('bench_ctl', int(eid), int(cid), int(n), float(r), float(d), l)
  except Exception as inst:
    print("Failed to insert values:" )
    traceback.print_exc()

def qrygraph_line(query, title, rowhead=False):
  data = runquery(query, True)
  name = data[0]
  series = [list(x) for x in zip(*data[1:])]
  if rowhead:
    X = series[0]
  else:
    X = np.arange(len(series[0]))
  plt.clf()
  start = 1 if rowhead else 0
  for y in range(start, len(series)):
    plt.plot(X, series[y], label=name[y])
  plt.xlabel(title)
  plt.legend()
  plt.savefig(os.path.join(GRAPH_LOC, title + '.png'))
  plt.close()  


def benchmark_graph(datalabel, expname=None, Xseries=None):
  if expname is not None:
    eid = get_expid(expname)
    qry = "SELECT deltatime FROM bench_ctl WHERE expid='%d' AND label ='%s';" % (eid, datalabel)
    qrygraph_line(qry, expname + '-' + datalabel)
  else:
    series = {}
    datalen = 10000000
    for exp in expname:
      eid = get_expid(exp)
      qry = "SELECT deltatime FROM bench_ctl WHERE expid='%d' AND label ='%s';" % (eid, datalabel)
      series[exp] = runquery(query)
      datalen = min(datalen, len(series[exp]))
    minsize = min([len(s) for s in series.values()])
    if Xseries is not None:
      X = Xseries[:min(len(Xseries), len(data))]
    else:
      X = np.arange(len(minsize))
    plt.clf()
    for key, Y in series.items():
      plt.plot(X, Y, label=key)
    plt.xlabel(datalabel)
    plt.legend()
    plt.savefig(os.path.join(GRAPH_LOC, 'bench-' + datalabel + '.png'))
    plt.close()  


# EXAMPLE:
# db.qrygraph_line("SELECT deltatime from bench_ctl where label='Boostrap';", 'time')

# Insert an experiment and return the experiment_id associated with it

def removebrace(s):
  for br in ['[', ']', '(', ')', '{', '}', ',']:
    s = s.replace(br, '')
  return s


def scrape_bench_ctl(name):
  eid = get_expid(name)
  with open((HOME + '/ddc/results/benchcons/ctl_%s.txt' % name)) as sfile:
    src = sfile.read().strip()
    bench = []
    data = []
    postdata = False
    collect = False
    cid = 0
    for line in src.split('\n'):
      if line.startswith('==>'):
        postdata = True
        collect = True
      if line.startswith('CATALOG APP'):
        collect = False
      if postdata and len(data) > 0:
        last = 0
        insertcnt = 0
        for n, tick in enumerate(data):
          if tick[0] == 'TIME':
            continue
          r = float(tick[0])
          l = tick[1]
          d = r - last if len(tick) == 2 else float(tick[2])
          insert('bench_ctl', int(eid), cid, n, float(r), float(d), l)
          insertcnt += 1
          last = r
        print('Insert for %d:  %d' % (cid, insertcnt))
        cid += 1
        data = []
        postdata = False
      if collect and line.startswith('##'):
        stat = line.split()
        val = stat[1].strip()
        label = stat[2].strip()
        if len(label) > 6:
          label = label[:6]
        data.append((val, label))


def ctl_file_parser(name):
  os.chdir(HOME + '/work/log/%s/' % name)
  print(len(os.listdir()))
  cwfilelist = [i for i in os.listdir() if i.startswith('cw')]
  print(len(cwfilelist))
  nums = []
  for cwfile in cwfilelist:
    with open(cwfile) as sfile:
      src = sfile.read().strip()
      kmeansnext = False
      for line in src.split('\n'):
        elm = line.split()
        if 'JOB NAME:' in line:
          jname = elm[-1].split('-')[1]
          a, b = jname.split('.')
          cid = '%06d'% (int(a)*100 + int(b))
        if '##NUM_RMS_THIS_ROUND' in line:
          numpts = int(removebrace(elm[-1]))
        if 'Total Observations by state' in line:
          bincnt = eval(line[line.index(':')+1:])
        # if kmeansnext:
        #   kmcnt = [int(removebrace(i)) for i in line.strip().split()]
        #   kmeansnext = False
        # if 'KMeans complete' in line:
        #   kmeansnext = True
        # if '##CACHE_HIT_MISS' in line:
          hit, miss, ratio = elm[-3:]
        if 'Updated Job Queue length' in line:
          numres = int(elm[-1])
    nums.append((cid, numpts))
  return nums



# Queries to run:

# run("SELECT expname, avg(deltatime), max(deltatime) FROM bench_ctl B, expr E where label='Sample' and E.expid=B.expid GROUP BY expname;")


run("SELECT expname, label, avg(deltatime), avg(num) as num FROM bench_ctl B, expr E where E.expid=B.expid and B.expid=8 GROUP BY expname, label order by expname, num;")

run("SELECT expname, max(runtime) FROM bench_ctl B, expr E where E.expid=B.expid GROUP BY expname order by expname;")

run("select expname, avg(time) from (SELECT expid, ctlid, max(runtime) as time FROM bench_ctl GROUP BY expid, ctlid) T,  expr E where E.expid=T.expid GROUP BY expname;")
