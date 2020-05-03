import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

import trimesh


#>0
def compute_force(x):
	y = 1/pow(x+1e-6,8)
	return y


def create_map_from_obj(fn, count=10000, size=32):
	mesh = trimesh.load_mesh(fn)

	nodes, face_index = trimesh.sample.sample_surface_even(mesh, count)

	

	min_coordinate= nodes.min()
	max_coordinate= nodes.max()

	#normalize nodes
	nodes = ((nodes-min_coordinate)/(max_coordinate-min_coordinate)/2+0.25)*size

	np.savetxt("gt_pc.txt", nodes,delimiter=';')
	return  nodes

# gt_nodes node_num*3
# verts    vert_num*3
def compute_force_verts(gt_nodes, verts):
	node_num = gt_nodes.shape[0]
	vert_num = verts.shape[0]

	force = gt_nodes.reshape(1,node_num, 3).repeat(vert_num,0) - verts.reshape(vert_num, 1, 3).repeat(node_num,1)

	field_dist = pow(pow(force, 2).sum(2), 0.5) #vert_num*node_num

	field_force = compute_force(field_dist) #vert_num*node_num

	field_weight = field_force/ field_force.sum(1).reshape(vert_num,1).repeat( node_num,1) #normalize the weight

	force = force * field_weight.reshape(vert_num,node_num,1).repeat(3,2) #size*size*size*node_num*3
	force = force.sum(1) #vert_num*3

	return force




def compute_force_field( gt_nodes, map_size=32, displacement = [0.1,0.1,0.1]):
	size = map_size

	x=  np.linspace(0,size-1, size) +displacement[0]
	y = np.linspace(0, size-1,size) +displacement[1]
	z = np.linspace(0, size-1,size) +displacement[2]
	xv , yv, zv = np.meshgrid(x, y, z) #size*size*size
	vertices = np.concatenate((xv.reshape(-1,1),yv.reshape(-1,1),zv.reshape(-1,1)),1) #(size^3)*3

	node_num = gt_nodes.shape[0]
	
	field = compute_force_verts(gt_nodes, vertices)

	return field, vertices

def visualize_field(field, vertices, gt_nodes, out_fn):
	node_num = gt_nodes

	v= vertices #+field
	normal= field

	out = np.concatenate((v,normal),1) #(size^3)*6

	np.savetxt(out_fn, out, delimiter=';')


	


def animation(vertices, gt_nodes, steps):
	node_num=gt_nodes.shape[0]
	vert_num = vertices.shape[0]
	v=vertices*1
	for s in range(steps):
		print (s)
		v=v+np.random.randn(vert_num,3)/10
		field = compute_force_verts(gt_nodes,v)
		v=v+field
		np.savetxt("step%02d"%s+".txt", v, delimiter=";")
		

size=32
print ("sample vertices num", size*size*size)
print ("sample gt nodes num", count)
print("create nodes")
gt_nodes = create_map_from_obj('chair.obj', count=10000,size=size)
print("compute force")
field, vertices=compute_force_field( gt_nodes, map_size=size, displacement = [0.0,0.0,0.0])
print("save force field obj")
visualize_field(field, vertices, gt_nodes, out_fn="field0.txt")
print("steps")
animation(vertices, gt_nodes, steps=10)

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
