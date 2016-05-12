


def groupby (src):
  dest = {}
  for key, val in src.items():
    if key not in dest:
      dest[key] = []
    dest[key].append(val)
  return dest


def groupby_pair (src):
  dest = {}
  for key, val in src:
    if key not in dest:
      dest[key] = []
    dest[key].append(val)
  return dest

def groupby_cnt (src):
  dest = {}
  for val in src:
    if val not in dest:
      dest[val] = 0
    dest[val] += 1
  return dest