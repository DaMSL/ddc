import redis
import logging
import os
import datatools.datareduce as datareduce
import numpy as np

logging.basicConfig(format=' %(message)s', level=logging.DEBUG)

def backProjection(db, index_list):
      """Perform OFFLINE back projection function for a list of indices using
      given DB. Return a list of high dimensional points (one per index). 
      Assumes NO CACHE or DESHAW.
      """
      logging.debug('--------  BACK PROJECTION:  %d POINTS ---', len(index_list))
      # Derefernce indices to file, frame tuple:
      pipe = db.pipeline()
      for idx in index_list:
        pipe.lindex('xid:reference', int(idx))
      generated_framelist = pipe.execute()
      # Group all Generated indidces by file index 
      groupbyFileIdx = {}
      for i, idx in enumerate(generated_framelist):
        try:
          file_index, frame = eval(idx)
        except TypeError as e:
          print('Bad Index:', str(idx))
          continue
        if file_index not in groupbyFileIdx:
          groupbyFileIdx[file_index] = []
        groupbyFileIdx[file_index].append(frame)
      # Dereference File index to filenames
      generated_frameMask = {}
      generated_filemap = {}
      for file_index in groupbyFileIdx.keys():
        filename = db.lindex('xid:filelist', file_index)
        if filename is None:
          logging.warning('Error file not found in catalog: %s', filename)
        if not os.path.exists(filename):
          logging.warning('DCD File not found: %s', filename)
        else:
          key = os.path.splitext(os.path.basename(filename))[0]
          generated_frameMask[key] = groupbyFileIdx[file_index]
          generated_filemap[key] = filename
      # Add high-dim points to list of source points in a trajectory
      # Optimized for parallel file loading
      logging.debug('Sequentially Loading all trajectories')
      source_points = []
      for key, framelist in generated_frameMask.items():
        traj = datareduce.load_trajectory(generated_filemap[key])
        traj = datareduce.filter_alpha(traj)
        selected_frames = traj.slice(framelist)
        source_points.extend(selected_frames.xyz)
      return np.array(source_points)     
