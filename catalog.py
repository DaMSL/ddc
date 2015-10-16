import sqlite3
import redis
import socket
import sys
import os
import common
import subprocess as proc

import logging
logger = common.setLogger()


class catalog:
  def __init__(self):
    pass
  def exists(self):
    return False
  def initialize(self):
    pass
  def start(self):
    pass
  def stop(self):
    pass
  def register(self, key, val):
    pass
  def load(self, key):
    pass
  def save(self, key, val):
    pass
  def append(self, key, val):
    pass
  def incr(self, key):
    pass


class serverLess(catalog):
  def __init__(self):
    catalog.__init__(self)
    self.dbName = "data.db"
    self.conn = 0
    self.cur  = 0

  def exists(self):
    db = self.dbName
    result = os.path.exists(db)
    logger.debug(" CHECK EXIST: " + db + "   (%s)" % result)
    return result

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

  def stop(self):
  #  conn = psycopg2.connect(host=HOST, dbname=DNAME, user=USER, password=PASSWD)
    self.conn.close()


  def register(self, key, val):
    logger.debug(" Register: %s: %s" % (key, val))
    self.conn = sqlite3.connect(self.dbName)
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
    self.conn.close()

  def load(self, key):
    logger.debug(" Loading: %s", key)
    project = "SELECT value FROM store WHERE key='%s';" % key
    self.conn = sqlite3.connect(self.dbName)
    self.query(project)
    row = self.cur.fetchone()
    if len(row) == 0:
      print("Key retrieval error. Value not initialized in store")
      result = 0
    else:
      value = row[0]
      logger.debug("  Loaded: %s", value)
      if value == 'LIST':
        project = "SELECT * FROM %s;" % key
        self.query(project)
        result = []
        for i in self.cur.fetchall():
          result.append(i[0])
        logger.debug ("     list is len=%d" % len(result))
      else: 
        result = value
    self.conn.close()
    return result

  def save(self, key, val):
    logging.debug(" Saving: %s: %s" % (key, val))
    self.conn = sqlite3.connect(self.dbName)
    if type(val) == list:
      logger.debug("Saving a list")
      # DELETE OLD LIST, FOR NOW
      delete = "DELETE FROM %s;" % key
      self.query(delete)
      for i in val:
        logger.debug("  Inserting  " + str(i))
        insert = "INSERT INTO %s VALUES ('%s');" % (key, i)
        self.query(insert)
    else:
      insert = "UPDATE store SET value = '%s' WHERE key='%s';" % (val, key)
      self.query(insert)
    self.conn.close()

  def incr(self, key):
    # TODO: Not most efficient
    logging.debug(" Incrementing: %s" % (key))
    val = int(self.load(key))
    self.save(key, val + 1)

  def append(self, key, val):

    logger.debug("Appending: %s: %s" % (key, val))
    # # TODO: Not most efficient
    # curVal = list(self.load(key))
    # logger.debug("  curVal = %s, len=%d" % (str(curVal), len(curVal)))
    # newList = [val] if len(curVal) == 0 else curVal.append(val)
    # logger.debug("  newVal = %s" % str(newList))
    # self.save(key, newList)
    self.conn = sqlite3.connect(self.dbName)
    insert = "INSERT INTO %s VALUES ('%s');" % (key, val)
    self.query(insert)
    self.conn.close()


    logger.debug("DONE APPENDING\n")

class dataStore(catalog):
  def __init__(self, lockfile):
    catalog.__init__(self)
    self.r = 0
    self.lockfile = lockfile

    self.host = ''
    self.port = 6379
    self.database = 0

    self.start()

  def exists(self):
    if os.path.exists(self.lockfile):
      return self.r and self.r.ping()

    return False


  def initialize(self):
    self.start()

  def clear(self):
    self.r.flushdb()

  def start(self):
    # Check if it's already started and connected
    if self.r and self.r.ping():
      return

    # If already locked by another node, that node's connection info
    if os.path.exists(self.lockfile):
      with open(self.lockfile, 'r') as connect:
        h, p, d = connect.read().split(',')
        self.host = h
        self.port = int(p)
        self.database = int(d)

    # Otherwise, start it locally as a daemon server process
    else:
      self.host = socket.gethostname()
      with open(self.lockfile, 'w') as connect:
        connect.write('%s,%d,%d' % (self.host, self.port, self.database))
      err = proc.call(['redis-server', 'redis.conf'])
      if err:
        logger.error("ERROR starting local redis service on %s", self.host)    
      logger.debug('Started redis locally on ' + self.host)

    # Connect to redis as client
    self.conn()

    if not self.r or not self.r.ping():
      logger.error("ERROR connecting to redis service on %s", self.host)

  # TODO: Graceful shutdown and hand off
  def stop(self):
    if self.r:
      self.r.save()
      self.r.shutdown()
      self.r = 0
    else:
      if os.path.exists(self.lockfile):
        os.remove(self.lockfile)


  def register(self, key, val):
    if type(val) == list:
      if len(val) == 0:
        return
      expr = 'self.r.rpush(key'
      for v in val:
        expr += ','+str(v)
      expr += ')'
      eval(expr)      
    else:
      self.r.set(key, val)

  # Slice off data in-place. Asssume key stores a list
  def slice(self, key, num):
    data = self.r.lrange(key, 0, num-1)
    self.r.ltrim(key, num-1, self.r.llen(key)-1)
    return [d.decode() for d in data]

  # Check if key exists in db
  def check(self, key):
    if self.r.type(key).decode() == 'none':
      return False
    else:
      return True

  # Retrieve data stored at key
  def load(self, key):
    if self.r.type(key).decode() in ['list', 'none']:
      data = self.r.lrange(key, 0, self.r.llen(key)-1)
      return [d.decode() for d in data]
    else:
      return self.r.get(key).decode()

  # Save data in place (note: for data store, this is the same as register)
  def save(self, key, val):
    self.register(key, val)

  # Asssume key stores a list
  def append(self, key, val):
    self.r.rpush(key, val)

  # Asssume key stores a number
  def incr(self, key):
    self.r.incr(key)

  # Connect as client
  def conn(self):
    self.r = redis.StrictRedis(host=self.host, port=self.port, db=self.database)

  # TODO:  Additional notification logic, as needed
  def notify(self, key, state):
    if state == 'ready':
      self.r.set(key, state)

