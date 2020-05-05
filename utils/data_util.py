import numpy as np


def normalize_pc(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / float(m)


def save_pcnorm_to_ply(pc, norm, ply_fn):
    num = pc.shape[0]
    v_array = np.empty(num, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    v_array['x'] = pc[:,0]
    v_array['y'] = pc[:,1]
    v_array['z'] = pc[:,2]
    v_array['nx'] = norm[:, 0]
    v_array['ny'] = norm[:, 1]
    v_array['nz'] = norm[:, 2]
    PLY_v = PlyElement.describe(v_array, 'vertex')
    PlyData([PLY_v]).write(ply_fn)

def normalize_norm():