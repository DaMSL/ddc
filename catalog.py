import sqlite3
import redis
import sys
import common

import logging, logging.handlers
logger = common.setLogger()


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
