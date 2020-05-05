import numpy as np
import trimesh
from plyfile import PlyData, PlyElement
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../preprocessing'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
import data_util
import cal_field
import time

def save_pc_to_ply(pc, ply_fn):
	num = pc.shape[0]
	v_array = np.empty(num, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	v_array['x'] = pc[:,0]
	v_array['y'] = pc[:,1]
	v_array['z'] = pc[:,2]
	PLY_v = PlyElement.describe(v_array, 'vertex')
	PlyData([PLY_v]).write(ply_fn)
	
def clear_duplicated_vertices(pc):
	pc = np.round(pc, 6)

	print ("Clear duplicated vertices")
	print ("original num", pc.shape)

	pc_dict = dict()
	for p in pc:
		key = p.tostring()
		pc_dict[key] = p
	new_pc =[]
	for k in pc_dict:
		new_pc +=[pc_dict[k]]

	new_pc = np.array(new_pc)
	print ("after num", new_pc.shape)

	return new_pc

def create_map_from_obj(fn, count=10000, size=32):
	mesh = trimesh.load_mesh(fn)
	#nodes, face_index = trimesh.sample.sample_surface_even(mesh, count)
	#normalize nodes
	mesh.vertices = data_util.normalize_pc(mesh.vertices)
	np.savetxt("gt_pc.txt", mesh.vertices, delimiter=';')
	mesh.export("gt_mesh.obj")
	return mesh.vertices

def compute_force_verts(gt_nodes, vertices):
	tic = time.time()
	gvfs = cal_field.cal_field(vertices, gt_nodes)
	print("force time: {}".format(time.time()-tic))
	return gvfs

def compute_force_field(gt_nodes, size=32, displacement = np.array([0.001,0.001,0.001])):

	x = np.linspace(0, size-1,size)/(size*0.5) - 1.0
	y = np.linspace(0, size-1,size)/(size*0.5) - 1.0
	z = np.linspace(0, size-1,size)/(size*0.5) - 1.0
	xv, yv, zv = np.meshgrid(x, y, z) #size*size*size
	vertices = np.concatenate((xv.reshape(-1,1),yv.reshape(-1,1),zv.reshape(-1,1)),1) #(size^3)*3
	field = compute_force_verts(gt_nodes, vertices)
	return field, vertices

def visualize_field(field, vertices, out_fn):
	out = np.concatenate((vertices,field),1) #(size^3)*6
	np.savetxt(out_fn, out, delimiter=';')

def animation(vertices, gt_nodes, steps):
	vert_num = vertices.shape[0]
	v=vertices*1
	for s in range(steps):
		print(s)
		v = v+ (np.random.uniform(size=vert_num*3)/(100.00*(s+1))).reshape(vert_num,3)
		field = compute_force_verts(gt_nodes,v)
		v=v+field
		print("v shape", v.shape)
		# v = clear_duplicated_vertices(v)
		save_pc_to_ply(v, "step%02d"%s+".ply")


size=32
count=100000
print ("sample vertices num", size*size*size)
print ("sample gt nodes num", count )
print("create nodes")
gt_nodes = create_map_from_obj('chair.obj', count=count,size=size)
print("compute force")
field, vertices=compute_force_field( gt_nodes, size=size, displacement = np.array([0.0,0.0,0.0]))
print("save force field obj")
visualize_field(field, vertices, out_fn="field0.txt")
print("steps")
animation(vertices, gt_nodes, steps=7)

"""
print ("compute force field")
field, g_nodes = compute_force_field(map) 

print ("visualize field")
visualize_field(field, g_nodes)
animation(field,g_nodes,10)

img_force_mag = pow(pow(field, 2).sum(2),0.5)

plt.figure("mag")
plt.imshow(img_force_mag)
plt.figure("force x")
plt.imshow(field[:,:,0])
plt.figure("force y")
plt.imshow(field[:,:,1])
plt.figure("map")
plt.imshow(map)

plt.show()
"""
