import numpy as np
from plyfile import PlyData, PlyElement
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing'))
import normal_gen

def normalize_pc(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / float(m)


def save_pcnorm_to_ply(pc, norm, dist, ply_fn):
    num = pc.shape[0]
    v_array = np.empty(num, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('dist', 'f4')])
    v_array['x'] = pc[:,0]
    v_array['y'] = pc[:,1]
    v_array['z'] = pc[:,2]
    v_array['nx'] = norm[:, 0]
    v_array['ny'] = norm[:, 1]
    v_array['nz'] = norm[:, 2]
    v_array['dist'] = dist
    PLY_v = PlyElement.describe(v_array, 'vertex')
    PlyData([PLY_v]).write(ply_fn)

def read_pcnorm_ply(file):
    with open(file, 'rb') as f:
        plydata = PlyData.read(f)
        x = plydata.elements[0].data['x']
        y = plydata.elements[0].data['y']
        z = plydata.elements[0].data['z']
        nx = plydata.elements[0].data['nx']
        ny = plydata.elements[0].data['ny']
        nz = plydata.elements[0].data['nz']
        dists = plydata.elements[0].data['dist']
        locs = np.stack([x, y, z], axis=1)
        norms = np.stack([nx, ny, nz], axis=1)
        return locs, norms, dists

def normalize_norm(batch_data, norm):
    divide = np.linalg.norm(batch_data, axis=1, keepdims=True) * norm
    return batch_data/divide


def add_normal_jitters(points, normals, height=0.1, span=0.05):
    jitterx = np.random.uniform(-height, height, points.shape[0]).reshape([points.shape[0], 1])
    R = normal_gen.norm_z_matrix(normals, rect=False)
    round_points = np.matmul(R, np.expand_dims(round_sample(span, points.shape[0]), axis=2))
    # print("jitterx.shape, R.shape, round_points.shape", jitterx.shape, R.shape, round_points.shape)
    jitterx = np.multiply(jitterx, normals) + np.squeeze(round_points)
    return points + jitterx

def round_sample(radius, num):
    angle = np.random.uniform(0,1, size=num) * 2 * np.pi
    r = np.sqrt(np.random.uniform(0,1, size=num)) * radius
    return np.stack([np.multiply(r, np.cos(angle)), np.multiply(r, np.sin(angle)), np.zeros_like(r)], axis=1)