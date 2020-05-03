import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

import trimesh

import torch

#>0
def compute_force_torch(x):
	y = 1/torch.pow(x+1e-6,6)
	return y


def create_map_from_obj_torch(fn, count=10000, size=32):
	mesh = trimesh.load_mesh(fn)

	nodes, face_index = trimesh.sample.sample_surface_even(mesh, count)

	

	min_coordinate= nodes.min()
	max_coordinate= nodes.max()

	#normalize nodes
	nodes = ((nodes-min_coordinate)/(max_coordinate-min_coordinate)/4*3+0.125)*size

	np.savetxt("gt_pc.txt", nodes,delimiter=';')

	nodes = torch.FloatTensor(nodes).cuda()
	return  nodes

# gt_nodes node_num*3
# verts    vert_num*3
def compute_force_verts_torch(gt_nodes, vertices):
	node_num = gt_nodes.shape[0]
	total_vert_num = vertices.shape[0]


	batch_size = 4096
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




def compute_force_field_torch( gt_nodes, map_size=32, displacement = np.array([0.1,0.1,0.1])):
	size = map_size

	x=  np.linspace(0,size-1, size) +displacement[0]
	y = np.linspace(0, size-1,size) +displacement[1]
	z = np.linspace(0, size-1,size) +displacement[2]
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
		v=v+ torch.randn(vert_num,3).cuda()/20
		field = compute_force_verts_torch(gt_nodes,v)
		v=v+field

		v_np = np.array(v.data.tolist())
		np.savetxt("step%02d"%s+".txt", v_np, delimiter=";")
		

size=64
count=10000
print ("sample vertices num", size*size*size)
print ("sample gt nodes num", count )
print("create nodes")
gt_nodes = create_map_from_obj_torch('chair.obj', count=count,size=size)
print("compute force")
field, vertices=compute_force_field_torch( gt_nodes, map_size=size, displacement = [0.0,0.0,0.0])
print("save force field obj")
visualize_field_torch(field, vertices, gt_nodes, out_fn="field0.txt")
print("steps")
animation_torch(vertices, gt_nodes, steps=10)

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
