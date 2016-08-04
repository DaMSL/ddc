

def calc_new_centroids():

bpti_ds_cent = np.load('../bpti-alpha-dist-centroid.npy')

  rt = []
  for i in range(6391, 6399):
    rt.append(F.ExprAnl(port=i))
  for i in range(8):
    rt.load(25)
  del rt[2]   # bad data for some reason

well_traj = [[] for i in range(5)]
for exp in rt:
    for state in range(5):
      traj = exp.trlist[state*6]
      traj.center_coordinates()
      alpha = dr.filter_alpha(traj)
      well_traj[state].append(alpha)

well_ds = [dr.distance_space(trlist) for trlist in well_traj]

well_ds_flat = [op.flatten(trlist) for trlist in well_ds]
gen_centroids = [np.mean(wf, axis=0) for wf in well_ds_flat]

well_rms = [rmsd.calc_rmsd(wf, gen_centroids) for wf in well_ds_flat]

well_rms_de = [rmsd.calc_rmsd(wf, bpti_ds_cent) for wf in well_ds_flat]
well_min_de = [np.bincount([np.argmin(rms) for rms in wr]) for wr in well_rms_de]
well_label_de = [[np.argmin(rms) for rms in wr] for wr in well_rms_de]

well_label = well_label_de
adapt_centroids = bpti_ds_cent.copy()
last_count = [0 for i in range(5)]
for itr in range(200):  # or until no progress
  well_filter = [[] for i in range(5)]
  for state in range(5):
    for W, L in zip(well_ds_flat[state], well_label[state]):
      if L == state:
        well_filter[state].append(W)
  new_centroids = [np.mean(wf, axis=0) for wf in well_filter]
  well_rms_nc = [rmsd.calc_rmsd(wf, new_centroids) for wf in well_ds_flat]
  well_label = [[np.argmin(rms) for rms in wr] for wr in well_rms_nc]
  well_min_nc = [np.bincount([np.argmin(rms) for rms in wr]) for wr in well_rms_nc]
  print('New Centroid Performance:')
  for st, cnt in enumerate(well_min_nc):
    retain = ''
    if cnt[st] > last_count[st]:
      adapt_centroids[st] = new_centroids[st]
      last_count[st] = cnt[st]
      retain = '*'
    print(cnt[st], cnt, retain)

well_rms_ac = [rmsd.calc_rmsd(wf, adapt_centroids) for wf in well_ds_flat]
well_min_ac = [np.bincount([np.argmin(rms) for rms in wr]) for wr in well_rms_ac]
for st, cnt in enumerate(well_min_ac):
    print(cnt[st], cnt)

np.save('../adapted-dspace-centoids', adapt_centroids)

