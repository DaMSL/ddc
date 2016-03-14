import sqlite3
import sys
from collections import namedtuple

def namedtuple_factory(cursor, row):
    """
    Usage:
    con.row_factory = namedtuple_factory
    """
    fields = [col[0] for col in cursor.description]
    Row = namedtuple("Row", fields)
    return Row(*row)



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
"""
}


insertion = {
  'expr': "INSERT INTO expr (expid, expname) VALUES (%d, %s);",
  'conv': "INSERT INTO conv (expname, ts, label, val) VALUES (%s, %s);"}

exprRecord = namedtuple('exprRecord', 'expid expname')
convRecord = namedtuple('convRecord', 'expname ts label val')

conn = sqlite3.connect('ddc_data.db')

for emp in map(EmployeeRecord._make, cursor.fetchall()):
    print(emp.name, emp.title)


def getConn():
  return conn

def createTables(conn):
cur = conn.cursor()
for table, query in tables.items():
  try:
    cur.execute(query)
    conn.commit()
  except Exception as inst:
    print("Failed to create tables:" )
    print(inst)
    sys.exit(1)

def dropTables(conn):
  for table in tables.keys():
    try:
      query = 'DROP TABLE IF EXISTS %s CASCADE;' % table
      cur.execute(query)
      conn.commit()
    except Exception as inst:
      print("Failed to create tables:" )
      print(inst)
      sys.exit(1)

def insert(table, *values):
  try:
    query = "INSERT INTO %s (expid, expname) VALUES (%s, %s);"
    cur = conn.cursor()
    cur.execute(query, exp.tup())
    val = int(cur.fetchone()[0])
    conn.commit()
    return val
  except Exception as inst:
      print("Failed to insert Experiment: ")
      print(inst)
      sys.exit(1)





# Insert an experiment and return the experiment_id associated with it


# Insert a trial and return the trial_id associated with it.
def insertTrial(conn, trial):
  if dbDisabled:
      return
  try:
    query = "INSERT INTO trials (experiment_id, trial_num, system, ts) VALUES (%s, %s, %s, %s) RETURNING trial_id"
    cur = conn.cursor()
    cur.execute(query, trial.tup())
    val = int(cur.fetchone()[0])
    conn.commit()
    return val

  except Exception as inst:
      print("Failed to insert Trial: ")
      print(inst)
      sys.exit(1)

def insertResult(conn, result):
  if dbDisabled:
      return
  try:
    query = "INSERT INTO results (trial_id, status, elapsed_ms, notes) VALUES (%s, %s, %s, %s)"
    cur = conn.cursor()
    cur.execute(query, result.tup())
    conn.commit()
  except Exception as inst:
      print("Failed to insert Result: ")
      print(inst)
      sys.exit(1)

def insertOperator(conn, operator):
  if dbDisabled:
      return
  try:
    query = "INSERT INTO operator_metrics VALUES (%s, %s, %s, %s, %s, %s, %s)"
    cur = conn.cursor()
    cur.execute(query, operator.tup())
    conn.commit()
  except Exception as inst:
      print("Failed to insert Operator: ")
      print(inst)
      sys.exit(1)

def getPlotData(conn):
  if dbDisabled:
      return
  try:
    query = "select * from experiment_stats e where not exists (select * from plots where experiment_id = e.experiment_id);"
    cur = conn.cursor()
    cur.execute(query)
    results = cur.fetchall()
    return results
  except Exception as inst:
      print("Failed to get new plot data: ")
      print(inst)
      sys.exit(1)

def getMetricPlotData(conn):
  if dbDisabled:
      return
  try:
    query = "SELECT T.experiment_id,trial_id, trial_num, system, query, dataset, workload FROM trials as T, experiments AS E WHERE T.experiment_id = E.experiment_id and NOT EXISTS (select * from metric_plots where trial_id = T.trial_id);"
    cur = conn.cursor()
    cur.execute(query)
    results = cur.fetchall()
    return results
  except Exception as inst:
      print("Failed to get new metric plot data: ")
      print(inst)
      sys.exit(1)

def registerPlot(conn, exp_id):
  if dbDisabled:
      return
  try:
    query = "INSERT INTO plots VALUES (%s);"
    cur = conn.cursor()
    cur.execute(query, (exp_id,) )
    conn.commit()
  except Exception as inst:
      print("Failed to insert plot: ")
      print(inst)
      sys.exit(1)

def registerMetricPlot(conn, trial_id):
  if dbDisabled:
      return
  try:
    query = "INSERT INTO metric_plots VALUES (%s);"
    cur = conn.cursor()
    cur.execute(query, (trial_id,) )
    conn.commit()
  except Exception as inst:
      print("Failed to insert metric plot: ")
      print(inst)
      sys.exit(1)