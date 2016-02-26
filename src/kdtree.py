#!/usr/bin/env python
"""KD Tree Implementation

  KDTrees describe a binary tree envoding for a geometric K-Dimensional space
    Each level it the tree divides a single dimension evenly into two hyperplanes.
    Dimensional axises are determined by modulo divsion (e.g. depth level 5 for
    a 3D space will subdivide the 2nd (or Y-) axis. 
"""

from collections import deque
import numpy as np
import sys
import json


__author__ = "Benjamin Ring"
__copyright__ = "Copyright 2015, Data Driven Control"
__version__ = "0.0.1"
__email__ = "bring4@jhu.edu"
__status__ = "Development"


class KDTree(object):
  """
    The KD-Tree contains the leafsize, the backing data accessor function
      and a root Node. All nodes in the tree can be one of the following:
      Inner --> will contain the mid or "split" value: the median of all 
        points along its specified axis when the split routine was called. Inner
        Nodes also have left and right children
      Leaf --> contains list of indices and its hi/lo extents. A leaf is a hypercube
      Note that the root node is a special case

    KDTrees are created using a numpy NDArray -- the data field holds an accessor
      function. Thus, for reconstruction, the accessor will normally hold a 
      function call to a data store holding the actual points

    This implementation is designed for encoding and (offline) storage. The encoding
      is a binary key whose length represents a node's deth and value (from R->L)
      encodes the traversal down the tree (e.g. '101' is a node from :
        root -> right -> left -> right). If its "val" field is set, then the node
      is an inner node and the is the mid point for its respective axis. If its
      a leaf node, the val field is either the lower extent (of its dimension) 
      if its a left child or the higher extent (if right). This allows the tree
      to logically encode the entents along any dimension rooted at any
      aprbitrary node by retrieving the left most D-children (for lower) and 
      right most D-children (for higher extents). Note that the key is not
      stored with the node, but rather used to encode/reconstruct it.

    Setting leaf size can tune performance: A larger leaf size is a smaller tree, 
      but will entail more I/O requests for insertions. 
  """


  class Node:
    """
    Subclass representing a single Node in the KD-Tree
      Data is abtracted from the Node (hence the split function
      resides at the higher KD-class). The Node only knows the 
      single-dimensional extents (high and low values) of itself
      (which includes all children) 
    """
    def __init__(self, depth):
      self.hi = None
      self.lo = None
      self.mid = None
      self.depth = depth
      self.elm = None
      self.left = None
      self.right = None

    def configure(self, lo, hi, idx_list):
      """
      Sets a new node to a given hi/lo value and list of indexes
      """
      self.hi = hi
      self.lo = lo
      self.elm = deque(idx_list)

    def size(self, node_only=False):
      """
      Returns size of this node (if leaf) or of all children nodes (if inner)
      Set node_only to true to only return the amount for this node (for debugging)
      """
      if self.elm is not None:
        return len(self.elm)
      elif node_only or (self.left == None and self.right == None):
        return 0
      else:
        return self.left.size() + self.right.size()

    def collect(self):
      """
      Returns collection of all indices rooted at this node
      """
      if self.elm is not None:
        return self.elm
      else:
        if self.left is None or self.right is None:
          print ("ERROR! left or right is None")

        return list(self.left.collect()) + list(self.right.collect())

    def encode(self, mykey):
      """
      Returns a mapped dictionary from key (binary code) --> node description
      TODO:  Change encoding to allow for top-level key-value insertions into
        a redis hash (e.g. '1101': '<val>' or '1101':'[list]') 
      """
      if self.mid is None:
        val = self.lo if mykey[-1] == '0' else self.hi
        return {mykey: {'val': val, 'elm': None if self.elm is None else list(self.elm)}}
      else:
        encoding = {mykey: {'val': self.mid, 'elm': None}}
        encoding.update(self.left.encode(mykey+'0'))
        encoding.update(self.right.encode(mykey+'1'))
        return encoding

    def decode(self, key):
      """
      Recursively traverses the tree using the given key and returns all the
      indices
      """
      if key == '' or self.elm is not None:
        return self.collect()
      else:
        if key[0] == '0':
          return self.left.decode(key[1:]) 
        else:
          return self.right.decode(key[1:])

    def insert(self, idx):
      """
      Inserts a new index into this node
      """
      self.elm.append(int(idx))
      return ''

    def reconstruct(self, key, parentMid, mapping):
      """
      Given a mapping of all nodes, reconstruct the node recursively.
      The parentMid is needed to propagate hi/lo values down the tree during
      reconstruction:  Each leaf contains one unique lo (for left child) or
      hi (for right) value; correspondingly, the left child hi val is its parent's
      mid (and a right child lo is its parent's mid).
      """
      # Leaf Node
      if key != '' and mapping[key]['elm'] is not None:
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
       
      self.mid = mid
      self.elm = None
      self.depth = len(key)
      left  = KDTree.Node(self.depth+1)
      right = KDTree.Node(self.depth+1)
      self.lo = left.reconstruct(key+'0', self.mid, mapping)
      self.hi = right.reconstruct(key+'1', self.mid, mapping)
      self.left = left
      self.right = right

      if key == '':
        return
      elif key[-1] == '0':
        return self.lo
      else:
        return self.hi


    def __str__(self):
      """
      Formatted output
      """
      leaf = '[INNER]' if self.elm is None else '[LEAF]'
      r = 'N/A' if self.right is None else str(self.right.size())
      l = 'N/A' if self.left  is None else str(self.left.size())
      s = 'KDNode  depth=%d  size=%d   type=%s   left=%s  right=%s' % (self.depth, self.size(), leaf, l, r)
      return s



  def __init__(self, leafsize, data=None):
    """
    Created new KD-Tree. If data (in ND-Array form) is provided, builds
    the KD-Tree with the data
    """

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
    """
    Recursively splits the given node until both child nodes are leaves
    """
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
    """
    Constructs the KD with given data set (of K-dimensional points)
    """
    self.data = lambda i, j: dataArray[i][j]
    self.split(self.root)

  def retrieve(self, key):
    """
    Given binary key, returns the list of indices in the node (and all children)
    """
    return self.root.decode(key)

  def insertPt(self, ptIdx, node):
    """
    Inserts the given indexed point into the node (recursively); 
    assumes: KD.data is an function to retrieve point with the given ptIndex
    """
    if node.elm is not None:
      if node.size() >= self.leafsize:
        self.split(node)
        return self.insertPt(ptIdx, node)
      else:
        return node.insert(ptIdx)
    else:
      axis = node.depth % self.dim
      if self.data(ptIdx,axis) < node.mid:
        return '0' + self.insertPt(ptIdx, node.left)
      else:
        return '1' + self.insertPt(ptIdx, node.right)

  def insert(self, point, index=None):
    # TODO: fix point - index reference here
    """
    Inserts a new point using given index.
    IF Index is None: assume underling data locally stored (not index-reference)
      and adds point to data list
    Returns the key to the node in which the point was inserted
    """
    # Insert new point (grow as needed)
    if index is None:
      index = len(self.data)
      self.data.append(point)
    return self.insertPt(index, self.root)


  def project(self, point, node=None):
    """
    projects this point into the tree and Returns the key to the node
    Does NOT insert the point
    """
    if node is None:
      node = self.root
    if node.elm is not None:
      return ''
    else:
      axis = node.depth % self.dim
      if point[axis] < node.mid:
        return '0' + self.project(point, node.left)
      else:
        return '1' + self.project(point, node.right)


  def encode(self):
    """
    Convert KD-Tree in a serializable form (python dict)
    """
    encoding =  {'_': self.root.mid, 'leafsize':self.leafsize, 'dim':self.dim}
    encoding.update(self.root.encode(''))
    return encoding

  @classmethod
  def reconstruct(cls, mapping, dataAccessorFunc):
    """
    Create a new KD Tree Object an reconstruct it from the given encoded mapping.
    """
    tree = KDTree(mapping['leafsize'])
    tree.dim = mapping['dim']
    tree.data = dataAccessorFunc
    tree.root       = KDTree.Node(0)
    tree.root.reconstruct('', mapping['_'], mapping)
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