import time
import sys
import sqlite3
import redis
import subprocess
import logging, logging.handlers

logger = logging.getLogger("")

# TODO: Move common from class to module
class common:
  catFile = "cat.log"


class serverLess:
  def __init__(self, name):
    self.dbName = name + ".db"
    self.conn = 0
    self.cur  = 0
    self.start()

  def initialize(self):
    self.start()
    table = 'CREATE TABLE IF NOT EXISTS store (key text, value text);'
    self.query(table)    

  def query(self, qry):
    try:
      self.cur = self.conn.cursor()
      self.cur.execute(qry)
      self.conn.commit()
    except Exception as ex:
      print("Failed to execute query: %s" % qry)
      print(ex)
      sys.exit(1)

  def start(self):
  #  conn = psycopg2.connect(host=HOST, dbname=DNAME, user=USER, password=PASSWD)
    self.conn = sqlite3.connect(self.dbName)

  def register(self, key, val):
    logger.debug(" Register: %s: %s" % (key, val))
    if type(val) == list:
      logger.debug("  Registering a list")
      insert = "INSERT INTO store VALUES ('%s', '%s');" % (key, 'LIST')
      self.query(insert)
      table = 'CREATE TABLE IF NOT EXISTS %s (item text);' % key
      self.query(table)
      for i in val:
        logger.debug("  Inserting  " + str(i))
        insert = "INSERT INTO %s VALUES ('%s');" % (key, i)
        self.query(insert)
    else:
      insert = "INSERT INTO store VALUES ('%s', '%s');" % (key, val)
      self.query(insert)

  def load(self, key):
    logger.debug(" Loading: %s", key)
    project = "SELECT value FROM store WHERE key='%s';" % key
    self.query(project)
    result = self.cur.fetchone()
    if len(result) == 0:
      print("Key retrieval error. Value not initialized in store")
      return 0
    else:
      value = result[0]
      logger.debug("  Loaded: %s", value)
      if value == 'LIST':
        project = "SELECT * FROM %s;" % key
        self.query(project)
        return self.cur.fetchall()
      return value

  def save(self, key, val):
    logging.debug(" Saving: %s: %s" % (key, val))
    insert = "UPDATE store SET value = '%s' WHERE key='%s';" % (val, key)
    self.query(insert)

  def incr(self, key):
    # TODO: Not most efficient
    logging.debug(" Incrementing: %s" % (key))
    val = int(self.load(key))
    self.save(key, val + 1)

  def append(self, key, val):
    logging.debug(" Appending: %s: %s" % (key, val))
    # TODO: Not most efficient
    curVal = list(self.load(key))
    self.save(key, curVal.append(val))

class dataStore:
  def __init__(self):
    self.r = 0

  def start(self):
    if not self.r or not self.r.ping():
      err = subprocess.call(['redis-server', 'redis.conf'])
      if err:
        print ("ERROR starting/codennecting to local redis service")    
    self.conn()

  def stop(self):
    self.r.save()
    self.r.shutdown()
    self.r = 0

  def conn(self):
    pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
    self.r = redis.StrictRedis(connection_pool=pool)

  def insert(self, key, val):
    # TODO: Exception handling
    self.r.set(key, value)



class macrothread(object):
  def __init__(self, name):
    self._input = 0
    self._term  = 0
    self._exec  = 0
    self._split = 0
    self.catalog = 0
    self.name = name
  # def __init__(self, inp, term, user, split):
  #   self._input = inp
  #   self._term  = term
  #   self._exec  = user
  #   self._split = split

  def term(self):
    pass

  def split(self):
    print ("passing")

  def execute(self, item):
    pass

  def fetch(self, i):
    pass


  def manager(self):
    logger.debug("MANAGER for %s", self.name)

    # TODO: Catalog Service Check here
    catalog = serverLess(self.name)
    if self.term(catalog):
      sys.exit(0)
    immed = self.split(catalog)
    print ("SPLIT OFF:  ", immed)
    # TODO: save back state, if nec'y

    #registry = catalog.retrieve(common.catFile)

    for i in immed:
      # TODO: Enum or dev. k-v indexing system
      # catalog.notify(i, 'pending')
    #  job = pyslurm.reservation
    #  job.create(JOB_INFO_DICT_HERE)
      print ("SCHEDULING:  ", i)
      self.worker(i)

  def worker(self, i):
    print ("FETCHING: ", i)
    jobInput = self.fetch(i)

    print ("EXECUTING: ", jobInput)

    # TODO: Catalog Service Check here
    catalog = serverLess(self.name)

    #  catalog.notify(i, "active")
    data = self.execute(catalog, jobInput)
    print ("RESULTS: ", data)

    #  catalog.notify(i, "complete")
    for d in data:
    #  catalog.notify(d, "ready")
          print ("  data: ", d)  

class simThread(macrothread):
  def __init__(self, name):
    macrothread.__init__(self, name)
    self._input = 'JCQueue'
    self._term  = 'JCComplete'
    self._split = 'splitParam'
    self._exec  = 'rawFileList'

    # macrothread.__init__(self, initParams, len(initParams), 0, 3)
    # self.JCComplete = 0
    # self.JCIndex = initParams
    # self.JC_Queue = initParams
    # self.rawFileList = []

  def initialize(self, catalog, initParams):
    catalog.register('JCQueue', initParams)
    catalog.register('JCComplete', 0)
    catalog.register('JCTotal', len(initParams))
    catalog.register('splitParam', 3)        
    catalog.register('rawFileList', [])

  def term(self, catalog):
    logger.debug("Checking Term")
    jccomplete = catalog.load('JCComplete')
    jctotal    = catalog.load('JCTotal')
    return (jccomplete == jctotal)

  def split(self, catalog):
    print ("splitting....")
    split = int(catalog.load('splitParam'))
    jcqueue = catalog.load('JCQueue')
    immed = jcqueue[:split]
    defer = jcqueue[split:]
    catalog.save('JCQueue', defer)
    return immed

  def execute(self, catalog, i):
#    target = exec(DO_SIM)
    target = "sample.out"
    catalog.append('rawFileList', 'sample.out')
    catalog.incr('JCComplete')
    return [target]

  def fetch(self, i):
    return i


if __name__ == "__main__":
  logging.Formatter(fmt='[%(asctime)s %(levelname)-5s %(name)s] %(message)s',datefmt='%H:%M:%S')
  log_console = logging.StreamHandler()
  # log_console.setFormatter(log_fmt)
  logger.setLevel(logging.DEBUG)
  logger.addHandler(log_console)

  params = [1, 2, 3, 4, 5, 6]
  sim = simThread("sim")
  catalog = serverLess("sim")
  catalog.initialize()
  sim.initialize(catalog, params)
  sim.manager()
