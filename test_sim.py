import time
from random import random, randint
import argparse

def gendata(x):
  return randint(0, x)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--file')
  parser.add_argument('-i', '--input', type=int)
  args = parser.parse_args()

  time.sleep(2)
  with open(args.file, 'w') as o:
    for i in range(100):
      data = gendata(args.input)
      o.write(str(data) + '\n')
  time.sleep(2)
