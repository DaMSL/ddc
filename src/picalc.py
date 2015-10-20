from random import uniform
from math import sqrt, pow
import time

jc = [(0., 1., 0., 1.), 
      (0., 1., 0., -1.), 
      (0., -1., 0, 1.), 
      (0., -1., 0, -1.)]


def piSim(f, param):
  x0, x1, y0, y1 = param
  with open(f, 'w') as output:
    for i in range(100):
      x = uniform(x0, x1)
      y = uniform(y0, y1)
      output.write("%f %f\n" % (x, y))
  time.sleep(5)

def piAnl(f):
  circ = 0
  num  = 0
  with open(f) as infile:
    for line in infile.readlines():
      num += 1
      x,y = [float(v) for v in line.split()]
      if sqrt(x*x + y*y) <= 1.:
        circ += 1
  time.sleep(5)
  return circ, num

def piEst(circ, num):
  time.sleep(5)
  if num == 0:
    return 0
    
  return 4. * circ / num



