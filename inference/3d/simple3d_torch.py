import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

import trimesh

import torch

from plyfile import PlyData, PlyElement

def save_pc_to_ply(pc , ply_fn):
	num = pc.shape[0]
	channel = pc.shape[1]

	v_array=[]
	v_array = np.empty(num, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

	v_array['x'] = pc[:,0]
	v_array['y'] = pc[:,1]
	v_array['z'] = pc[:,2]

	PLY_v = PlyElement.describe(v_array, 'vertex')

	
	PlyData([PLY_v]).write(ply_fn)
	
def clear_duplicated_vertices(pc):
	pc = np.round(pc, 6)

	print ("Clear duplicated vertices")
	print ("original num", pc.shape[0])

	pc_dict=  dict()
	for p in pc:
		key = p.tostring()
		pc_dict[key] = p
	new_pc =[]
	for k in pc_dict:
		new_pc +=[p]

	new_pc = np.array(new_pc)
	print ("after num", new_pc.shape[0])

	return new_pc



#>0
def compute_force_torch(x):
	y = 1/(torch.pow(x*1000,8)+1e-6)
	#print (x.min(),x.max())
	#y = 1/(pow(np.e, x*10))
	return y


def create_map_from_obj_torch(fn, count=10000, size=32):
	mesh = trimesh.load_mesh(fn)

	#nodes, face_index = trimesh.sample.sample_surface_even(mesh, count)

	nodes = mesh.vertices

	

	min_coordinate= nodes.min()
	max_coordinate= nodes.max()

	#normalize nodes
	nodes = ((nodes-min_coordinate)/(max_coordinate-min_coordinate)/10*9+0.05)

	mesh.vertices=((mesh.vertices - min_coordinate)/(max_coordinate-min_coordinate)/10*9+0.05)

	np.savetxt("gt_pc.txt", nodes,delimiter=';')

	mesh.export("gt_mesh.obj")

	nodes = torch.FloatTensor(nodes).cuda()
	return  nodes

# gt_nodes node_num*3
# verts    vert_num*3
def compute_force_verts_torch(gt_nodes, vertices):
	node_num = gt_nodes.shape[0]
	total_vert_num = vertices.shape[0]


	batch_size = 1024
	start_id = 0

	
	force_total = torch.FloatTensor([]).cuda()

	while (True):

		end_id = min(start_id+batch_size, total_vert_num)

		verts = vertices[start_id:end_id]
		vert_num = verts.shape[0]

		force = gt_nodes.view(1,node_num, 3).repeat(vert_num,1,1) - verts.view(vert_num, 1, 3).repeat(1,node_num,1)
		field_dist = torch.pow(torch.pow(force, 2).sum(2), 0.5) #vert_num*node_num

		field_force = compute_force_torch(field_dist) #vert_num*node_num

		field_weight = field_force/ field_force.sum(1).view(vert_num,1).repeat( 1, node_num) #normalize the weight

		force = force * field_weight.view(vert_num,node_num,1).repeat(1,1,3) #vert_num*node_num*3
		force = force.sum(1) #vert_num*3

		force_total = torch.cat((force_total, force), 0)

		start_id = end_id
		
		if(end_id == total_vert_num):
			break

	#print (force_total.shape, total_vert_num)

	return force_total




def compute_force_field_torch( gt_nodes, map_size=32, displacement = np.array([0.001,0.001,0.001])):
	size = map_size

	x=  np.linspace(0,size-1, size)/(size*1.0) +displacement[0]
	y = np.linspace(0, size-1,size)/(size*1.0) +displacement[1]
	z = np.linspace(0, size-1,size)/(size*1.0) +displacement[2]
	xv , yv, zv = np.meshgrid(x, y, z) #size*size*size
	vertices = np.concatenate((xv.reshape(-1,1),yv.reshape(-1,1),zv.reshape(-1,1)),1) #(size^3)*3

	vertices = torch.FloatTensor(vertices).cuda()

	node_num = gt_nodes.shape[0]
	
	field = compute_force_verts_torch(gt_nodes, vertices)

	return field, vertices

def visualize_field_torch(field, vertices, gt_nodes, out_fn):
	node_num = gt_nodes

	v= vertices #+field
	normal= field

	out = torch.cat((v,normal),1) #(size^3)*6

	out=np.array(out.data.tolist())

	np.savetxt(out_fn, out, delimiter=';')


	


def animation_torch(vertices, gt_nodes, steps):
	node_num=gt_nodes.shape[0]
	vert_num = vertices.shape[0]
	v=vertices*1

	torch.random.manual_seed(0)
	for s in range(steps):
		print (s)
		v=v+ torch.randn(vert_num,3).cuda()/100/(s+1)
		field = compute_force_verts_torch(gt_nodes,v)
		v=v+field

		v_np = np.array(v.data.tolist())

		v_np = clear_duplicated_vertices(v_np)

		save_pc_to_ply(v_np, "step%02d"%s+".ply")
		#np.savetxt("step%02d"%s+".ply", v_np, delimiter=";")
		

size=16
count=100000
print ("sample vertices num", size*size*size)
print ("sample gt nodes num", count )
print("create nodes")
gt_nodes = create_map_from_obj_torch('chair.obj', count=count,size=size)
print("compute force")
field, vertices=compute_force_field_torch( gt_nodes, map_size=size, displacement = [0.0,0.0,0.0])
print("save force field obj")
visualize_field_torch(field, vertices, gt_nodes, out_fn="field0.txt")
print("steps")
animation_torch(vertices, gt_nodes, steps=20)

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
