import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random

#>0
def compute_force(x):
	y = 1/pow(x+1e-6,4)
	return y


def create_map_line(width, height):
	map = np.zeros((width, height))

	left =int( width/4)

	right = int(left+width/2)

	map[int(height/2), left:right] += 1

	return map

def create_map_point(width, height):
	map = np.zeros((width, height))

	map[int(height/2), int(width/2)] += 1

	return map

def create_map_circle(w, h):
	map = np.zeros((w, h))

	x=np.linspace(-1,1, w)
	y = np.linspace(-1, 1,h)
	xv , yv = np.meshgrid(x, y)
	

	r = pow(pow(xv,2) + pow(yv, 2),0.5)

	r = (r<0.501) * (r>0.499)

	map = r.astype(np.float32)
	return map

def create_map_from_img(img_fn):
	img = mpimg.imread(img_fn)[:,:,0]

	img = img<0.5

	img = img.astype(np.float32)

	return img


def compute_force_field(map):
	w = map.shape[1]
	h = map.shape[0]

	x = np.linspace(0, w-1, w)
	y = np.linspace(0, h-1, h)
	xv , yv = np.meshgrid(x, y)

	g_nodes = []
	for i in range(h):
		for j in range(w):
			if(map[i,j]>0):
				g_nodes += [[j,i]]
	
	g_nodes = np.array(g_nodes) ## num*2
	node_num = g_nodes.shape[0]


	field = np.zeros((w, h, node_num, 2))

	for i in range(node_num):
		field[:,:,i,0] = g_nodes[i,0] - xv
		field[:,:,i,1] = g_nodes[i,1] -yv

	field_dist = pow(pow(field, 2).sum(3), 0.5) #h,w,node_num

	field_force = compute_force(field_dist) #h,w, node_num

	field_weight = field_force/ field_force.sum(2).reshape(h,w,1).repeat( node_num,2) #normalize the weight

	field = field * field_weight.reshape(h,w,node_num,1).repeat(2,3) #h,w,node_num,2
	field = field.sum(2) #h,w,2

	return field, g_nodes

def visualize_field(field, g_nodes):
	w = field.shape[1]
	h = field.shape[0]

	sample_num=100
	points_x = np.random.randint(0, w, sample_num)
	points_y = np.random.randint(0, h, sample_num)
	# print(points_x)
	# print(points_y)
	plt.plot(g_nodes[:,0], g_nodes[:,1], 'o', color='r', alpha=0.1)

	for n in range(sample_num):
		y = points_y[n]
		x = points_x[n]

		f = field[y,x]
		# print (f)

		new_x = x+f[0]
		new_y = y+f[1]

		plt.plot([x, new_x], [y,new_y],'-', color='b')

		plt.plot([x],[y],'o', color='b')


	plt.show()


def animation(field, g_nodes, steps):
	w = field.shape[1]
	h = field.shape[0]

	sample_num=100
	# points_x = np.random.uniform(0, w, sample_num)
	# points_y = np.random.uniform(0, h, sample_num)
	x = np.linspace(0, w - 2, w//5) + 0.1
	y = np.linspace(0, h - 2, h//5) + 0.1
	points_x, points_y = np.meshgrid(x, y)
	points_x = points_x.reshape(-1)
	points_y = points_y.reshape(-1)
	### first step
	fig = plt.figure("0")
	for n in range(points_y.shape[0]):
		y = points_y[n]
		x = points_x[n]
		plt.plot([x],[y],'o', color='b')

	plt.plot(g_nodes[:,0], g_nodes[:,1], 'o', color='r', alpha=0.1)
	plt.show()

	for s in range(steps):
		# plt.figure("%02d"%s)
		print(points_y[0],points_x[0])
		new_points_y = []
		new_points_x = []
		for n in range(points_y.shape[0]):
			y = points_y[n]
			x = points_x[n]


			f = interp(field, x,y)
			# print(f)
			new_x = max(min(x+f[0],w-1),0)
			new_y = max(min(y+f[1],h-1),0)
			plt.plot([x, new_x], [y, new_y], '-', color='b')
			plt.plot([new_x],[new_y],'o', color='b')

			## update the coordinates and add a bit permutation to avoid stuck at balanced place

			new_points_y.append(new_y)
			new_points_x.append(new_x)
			# new_points_y.append(new_y + [-1,1][random.randrange(2)]*np.random.uniform(8,10))
			# new_points_x.append(new_x + [-1,1][random.randrange(2)]*np.random.uniform(8,10))
			# new_points_y.append(new_y + [-1,1][random.randrange(2)]*np.random.uniform(8,10))
			# new_points_x.append(new_x + [-1,1][random.randrange(2)]*np.random.uniform(8,10))
		points_y = np.array(new_points_y)
		points_x = np.array(new_points_x)

		plt.plot(g_nodes[:,0], g_nodes[:,1], 'o', color='r', alpha=0.1)
		# fig.canvas.draw()
		# fig.canvas.flush_events()
		plt.show()

def interp(field, x,y):
	max_x, max_y = field.shape[1]-1, field.shape[0]-1
	x1,x2,y1,y2 = max(0,int(np.floor(x))), min(max_x, int(np.ceil(x))), max(0,int(np.floor(y))), min(max_y, int(np.ceil(y)))
	q11 = field[y1,x1]
	q12 = field[y2,x1]
	q21 = field[y1,x2]
	q22 = field[y2,x2]
	return (q11 * (x2 - x+ 1e-5) * (y2 - y+ 1e-5) +
			q21 * (x - x1+ 1e-5) * (y2 - y+ 1e-5) +
			q12 * (x2 - x+ 1e-5) * (y - y1+ 1e-5) +
			q22 * (x - x1+ 1e-5) * (y - y1 + 1e-5)
			) / ((x2 - x1 + 1e-5) * (y2 - y1 + 1e-5) )




# map=create_map_point(101,101)
#map=create_map_line(101,101)
#map = create_map_circle(101, 101)
map = create_map_from_img('bunny.png')
#map = create_map_from_img('chair.png')

print ("compute force field")
field, g_nodes = compute_force_field(map) 

print ("visualize field")
# visualize_field(field, g_nodes)
# animation(field,g_nodes,4)

img_force_mag = pow(pow(field, 2).sum(2),0.5)
#
plt.figure("mag")
plt.imshow(img_force_mag)
# plt.figure("force x")
# plt.imshow(field[:,:,0])
# plt.figure("force y")
# plt.imshow(field[:,:,1])
# plt.figure("map")
# plt.imshow(map)

plt.show()

