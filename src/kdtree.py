from collections import deque
import numpy as np
import sys
import json

class KDTree(object):

  class Node:
    '''
    Subclass representing a single Node in the KD-Tree
      Data is abtracted from the Node (hence the split function
      resides at the higher KD-class). The Node only knows the 
      single-dimensional extents (high and low values) of itself
      (which includes all children) 
    '''
    def __init__(self, depth):
      self.hi = None
      self.lo = None
      self.mid = None
      self.depth = depth
      self.elm = None
      self.left = None
      self.right = None

    def configure(self, lo, hi, idx_list):
      self.hi = hi
      self.lo = lo
      self.elm = deque(idx_list)

    def size(self):
      if self.elm is not None:
        return len(self.elm)
      elif self.left == None and self.right == None:
        return 0
      else:
        return self.left.size() + self.right.size()

    def collect(self):
      if self.elm is not None:
        return self.elm
      else:
        if self.left is None or self.right is None:
          print ("ERROR! left or right is None")

        return list(self.left.collect()) + list(self.right.collect())

    def encode(self, mykey):
      if self.mid is None:
        val = self.lo if mykey[-1] == '0' else self.hi
        return {mykey: {'val': val, 'elm': None if self.elm is None else list(self.elm)}}
      else:
        encoding = {mykey: {'val': self.mid, 'elm': None}}
        encoding.update(self.left.encode(mykey+'0'))
        encoding.update(self.right.encode(mykey+'1'))
        return encoding

    def decode(self, key):
      print ('  DECODE --> ', key, end=' ')
      if key == '' or self.elm is not None:
        print ('  COLLECT ')
        return self.collect()
      else:
        if key[0] == '0':
          print ('  GO LEFT: ', self.left)
          return self.left.decode(key[1:]) 
        else:
          print ('  GO RIGHT ', self.right)
          return self.right.decode(key[1:])

    def insert(self, idx):
      self.elm.append(int(idx))
      return ''

    def reconstruct(self, key, parentMid, mapping):

      # Leaf Node
      if key == '0':
        print ('[TRACE]  ', key)
      if key != '' and mapping[key]['elm'] is not None:
        if key == '0':
          print ('[TRACE]  ', 'ELM NOT NONE')
        self.elm = deque(mapping[key]['elm'])
        if key[-1] == '0':
          self.lo = mapping[key]['val']
          self.hi = parentMid
          return mapping[key]['val']
        else:
          self.lo = parentMid
          self.hi = mapping[key]['val']
          return mapping[key]['val']

      #  For Root Node, mid is the parent's mid value
      mid = parentMid if key == '' else mapping[key]['val']
      if key == '0':
        print ('[TRACE]  mid = ', mid)
       
      self.mid = mid
      self.elm = None
      self.depth = len(key)
      left  = KDTree.Node(self.depth+1)
      right = KDTree.Node(self.depth+1)
      if key == '0':
        print ('[TRACE]  L = ', str(left))
        print ('[TRACE]  R = ', str(right))
      self.lo = left.reconstruct(key+'0', self.mid, mapping)
      if key == '0':
        print ('[TRACE]  L = ', str(left))
        print ('[TRACE]  R = ', str(right))
      self.hi = right.reconstruct(key+'1', self.mid, mapping)
      if key == '0':
        print ('[TRACE]  L = ', str(left))
        print ('[TRACE]  R = ', str(right))
      self.left = left
      self.right = right
      if key == '00':
        print ('  Trace 00  --> ', self.__str__())

      if key == '':
        return
      elif key[-1] == '0':
        return self.lo
      else:
        return self.hi


    def __str__(self):
      leaf = '[INNER]' if self.elm is None else '[LEAF]'
      r = 'N/A' if self.right is None else str(self.right.size())
      l = 'N/A' if self.left  is None else str(self.left.size())
      s = 'KDNode  depth=%d  size=%d   type=%s   left=%s  right=%s' % (self.depth, self.size(), leaf, l, r)
      return s



  def __init__(self, leafsize, data=None):
    '''
    Created new KD-Tree. If data (in ND-Array form) is provided, builds
    the KD-Tree with the data
    '''

    self.leafsize = leafsize

    # New Tree
    if data is not None:
      if not isinstance(data, np.ndarray):
        logging.error("Can only instantiate with an NDArray.")
        sys.exit(0)
      print('DATA Dimensions =', data.shape)
      self.dim = data.shape[1]
      self.root = KDTree.Node(0)
      self.root.configure(np.min(data.T[-1]), np.max(data.T[-1]), deque(np.arange(len(data))))
      print('ROOT NODE:  ', self.root)
      self.build(data)


  def split(self, node, debug='N'):
    '''
    Recursively splits the given node until both child nodes are leaves
    '''
    axis = node.depth % self.dim
    vals = [self.data(i,axis) for i in node.elm]

    mid = np.median(vals)
    node.mid = mid
    left = deque()
    right = deque()
    while len(node.elm) > 0:
      pt = int(node.elm.popleft())
      if self.data(pt,axis) >= mid:
        right.append(pt)
      else:
        left.append(pt)
    node.elm   = None
    node.left  = KDTree.Node(node.depth+1)
    node.right = KDTree.Node(node.depth+1)
    node.left.configure(np.min(vals), mid, left)
    node.right.configure(mid, np.max(vals), right)
    # print(node.depth, debug, node.left.size(), node.right.size(), self.leafsize, end=',   ')
    # if node.depth==10:
    #   sys.exit()
    if len(left)> self.leafsize:
      # print(" SPLIT LEFT")
      self.split(node.left, debug='L')
    if len(right)> self.leafsize:
      # print(" SPLIT RIGHT")
      self.split(node.right, debug='R')


  def build(self, dataArray):
    '''
    Constructs the KD with given data set (of K-dimensional points)
    '''
    self.data = lambda i, j: dataArray[i][j]
    self.split(self.root)

  def retrieve(self, key):
    '''
    Given binary key, returns the list of indices in the node (and all children)
    '''
    return self.root.decode(key)

  def insertPt(self, ptIdx, node):
    '''
    Inserts the given indexed point into the node (recursively); 
    assumes: KD.data is an function to retrieve point with the given ptIndex
    '''
    if node.elm is not None:
      if node.size() >= self.leafsize:
        self.split(node)
        return self.insertPt(ptIdx, node)
      else:
        return node.insert(ptIdx)
    else:
      axis = node.depth % self.dim
      if self.data[ptIdx][axis] < node.mid:
        return '0' + self.insertPt(ptIdx, node.left)
      else:
        return '1' + self.insertPt(ptIdx, node.right)

  def insert(self, point, index=None):
    '''
    Inserts a new point using given index.
    IF Index is None: assume underling data locally stored (not index-reference)
      and adds point to data list
    '''
    # Insert new point (grow as needed)
    if index is None:
      index = len(self.data)
      self.data.append(point)
    return self.insertPt(index, self.root)

  def encode(self):
    encoding =  {'_': self.root.mid, 'leafsize':self.leafsize, 'dim':self.dim}
    encoding.update(self.root.encode(''))
    # encoding.update(self.root.encode('1'))
    return encoding

  @classmethod
  def reconstruct(cls, mapping, dataAccessorFunc):
    tree = KDTree(mapping['leafsize'])
    tree.dim = mapping['dim']
    tree.data = dataAccessorFunc
    tree.root       = KDTree.Node(0)
    print ('ROOT BEFORE: ', str(tree.root))
    tree.root.reconstruct('', mapping['_'], mapping)
    print ('ROOT AFTER: ', str(tree.root))
    print ('      LEFT: ', str(tree.root.left))
    print ('     RIGHT: ', str(tree.root.right))


    # tree.root.left  = KDTree.Node(1)
    # tree.root.right = KDTree.Node(1)
    # tree.root.split = mapping['_']
    # tree.root.lo = tree.root.left.reconstruct('0', tree.split, mapping)
    # tree.root.hi = tree.root.right.reconstruct('1', tree.split, mapping)
    return tree


def testKDTree(X, r):
  print("TEST KD Tree")
  # X = np.random.rand(1000, 3)
  # X = np.load('bpti10.npy', mmap_mode='r')
  kd = KDTree(100, data=X)
  print("Build Complete.")
  print("TEST decoding (retrieve a cell)....")
  for k in ['0', '1', '00']:
    print("Retrieving key: %s" % k)
    hcube = kd.retrieve(k)
    print('  HCube size = %d:   ' % len(hcube), str(list(hcube)[:5]))
  print("TEST encoding (json up)....")
  encoded = kd.encode()
  print("Encoded tree into %d keys" % len(encoded.keys()))
  print('  _Root  Node: %s' % encoded['_'])
  print('  _LEFT  Node: %s' % encoded['0'])
  print('  _RIGHT Node: %s' % encoded['1'])
  j = 0
  for k, v in encoded.items():
    print(k, v)
    j+=1
    if j == 2:
      break
  serialized = json.dumps(encoded)
  print (' # of keys = ', len(encoded.keys()))
  r.delete('hcube:pca')
  r.set('hcube:pca', serialized)

def testRebuild(r):
  deserialized = json.loads(r.get('hcube:pca'))
  print (' # of keys = ', len(deserialized.keys()))
  func = lambda i,j: np.fromstring(r.lindex('subspace:pca', i))[j]
  print("Reconstructing the tree...")
  tree = KDTree.reconstruct(deserialized, func)
  print (' Recheck source # of keys = ', len(deserialized.keys()))
  print("Retrieving key: %s" % '00')
  hcube = tree.retrieve('00')
  print('  HCube size = %d:   ' % len(hcube), str(list(hcube)[:5]))

  encoded = tree.encode()
  print (' ReRecheck source # of keys = ', len(deserialized.keys()))
  print("Encoded tree into %d keys" % len(encoded.keys()))
  print('  _Root  Node: %s' % encoded['_'])
  print('  _LEFT  Node: %s' % encoded['0'])
  print('  _RIGHT Node: %s' % encoded['1'])
  print('eKEY 101 is:  ', encoded['101'])
  print('sKEY 101 is:  ', deserialized['101'])
  print("Reconstruction Complete.")
  for k in ['0', '1', '00', '00100011100101', '10100011100101']:
    print("Retrieving key: %s" % k)
    hcube = tree.retrieve(k)
    print('  HCube size = %d:   ' % len(hcube), str(list(hcube)[:5]))


if __name__ == "__main__":
  import redis
  r = redis.StrictRedis(decode_responses=True)
  raw = r.lrange('subspace:pca', 0, -1)
  X = np.array([np.fromstring(x) for x in raw])
  testKDTree(X, r)
  print("Testing the Rebuild")
  testRebuild(r)