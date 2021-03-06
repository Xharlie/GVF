import numpy as np
import cv2
import random
import math
import os
import threading
import queue
import sys
import h5py
import copy
import trimesh
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data_util'))
import data_util

FETCH_BATCH_SIZE = 32
BATCH_SIZE = 32
HEIGHT = 192
WIDTH = 256
POINTCLOUDSIZE = 16384
OUTPUTPOINTS = 1024
REEBSIZE = 1024


class Pt_sdf_img(threading.Thread):
    
    def __init__(self, FLAGS, listinfo=None, info=None, qsize=64, cats_limit=None, shuffle=True):
        super(Pt_sdf_img, self).__init__()
        self.queue = queue.Queue(qsize)
        self.stopped = False
        self.bno = 0
        self.listinfo = listinfo
        # self.num_uni_pnts = FLAGS.num_uni_pnts
        # self.num_surf_pnts = FLAGS.num_surf_pnts
        # self.batch_size = FLAGS.batch_size
        self.img_dir = info['rendered_dir']
        self.gvf_dir = info['gvf_dir']
        self.norm_mesh_dir = info['norm_mesh_dir']
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 60000
        self.data_num = len(self.listinfo)
        self.FLAGS = FLAGS
        self.shuffle = shuffle
        self.num_batches = int(self.data_num / self.FLAGS.batch_size)
        self.cats_limit, self.epoch_amount = self.set_cat_limit(cats_limit)
        self.data_order = list(range(len(listinfo)))
        self.order = self.data_order
        print("filename",FLAGS.filename)

    def set_cat_limit(self, cats_limit):
        epoch_amount = 0
        for cat, amount in cats_limit.items():
            cats_limit[cat] = min(self.FLAGS.cat_limit, amount)
            epoch_amount += cats_limit[cat]
        print("epoch_amount ", epoch_amount)
        print("cats_limit ", cats_limit)
        return cats_limit, epoch_amount

    def get_img_dir(self, cat_id, obj):
        img_dir = os.path.join(self.img_dir, cat_id, obj)
        return img_dir, None


    def get_gvf_h5_filenm(self, cat_id, obj, view_id):
        return os.path.join(self.gvf_dir, cat_id, obj, view_id, self.FLAGS.unitype, "{}.h5".format(self.FLAGS.filename)), os.path.join(self.gvf_dir, cat_id, obj, view_id, self.FLAGS.unitype, "gt_pnt.txt")

    def get_gvf_init_filenm(self, cat_id, obj, view_id):
        return os.path.join(self.gvf_dir, cat_id, obj, view_id, self.FLAGS.unitype, "{}.ply".format(self.FLAGS.filename)), os.path.join(self.gvf_dir, cat_id, obj, view_id, self.FLAGS.unitype, "gt_pnt.txt")

    def get_obj_file_filenm(self, cat_id, obj):
        return os.path.join(self.norm_mesh_dir, cat_id, obj, "pc_norm.obj")

    def pc_normalize(self, pc, centroid=None):

        """ pc: NxC, return NxC """
        l = pc.shape[0]

        if centroid is None:
            centroid = np.mean(pc, axis=0)

        pc = pc - centroid
        # m = np.max(pc, axis=0)
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

        pc = pc / m

        return pc, centroid, m

    def __len__(self):
        return self.epoch_amount

    def memory(self):
        """
        Get node total memory and memory usage
        """
        with open('/proc/meminfo', 'r') as mem:
            ret = {}
            tmp = 0
            for i in mem:
                sline = i.split()
                if str(sline[0]) == 'MemTotal:':
                    ret['total'] = int(sline[1])
                elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                    tmp += int(sline[1])
            ret['free'] = tmp
            ret['used'] = int(ret['total']) - int(ret['free'])
        return ret

    def getitem(self, index):
        locs, norms, dists, gt_pnts =None,None,None,None
        cat_id, obj, num = self.listinfo[index]
        if self.FLAGS.filename is not None:
            # print("self.FLAGS.filename",self.FLAGS.filename)
            # gvf_file, pntfile = self.get_gvf_h5_filenm(cat_id, obj, str(num))
            # locs, norms, dists = self.get_gvf_h5(gvf_file)
            gvf_file, pntfile = self.get_gvf_init_filenm(cat_id, obj, str(num))
            locs, norms, dists = data_util.read_pcnorm_ply(gvf_file)
            gt_pnts = np.loadtxt(pntfile, delimiter=";")
        else:
            obj_file = self.get_obj_file_filenm(cat_id, obj)
            gt_pnts = self.get_gt_pnts(obj_file)
        img_dir, img_file_lst = self.get_img_dir(cat_id, obj)
        return locs, norms, dists, img_dir, img_file_lst, cat_id, obj, num, gt_pnts

    def get_gt_pnts(self, obj_file):
        mesh_list = trimesh.load_mesh(obj_file)
        if not isinstance(mesh_list, list):
            mesh_list = [mesh_list]
        points_all = np.zeros((0, 3), dtype=np.float32)
        for i in range(len(mesh_list)):
            mesh = mesh_list[i]
            # print("start sample surface of ", mesh.faces.shape[0])
            points, index = trimesh.sample.sample_surface(mesh, mesh.faces.shape[0] * 3)
            points_all = np.concatenate([points_all, points], axis=0)
        return points_all

    def get_gvf_h5(self, gvf_h5_file):
        # print(gvf_h5_file)
        try:
            h5_f = h5py.File(gvf_h5_file, 'r')
            locs = h5_f['locs'][:].astype(np.float32)
            norms = h5_f['norms'][:].astype(np.float32)
            dists = h5_f['dists'][:].astype(np.float32)
        except:
            print("h5py wrong:", gvf_h5_file)
            sys.stdout.flush()
        finally:
            h5_f.close()
        return locs, norms, dists

    # def get_img_old(self, img_dir, num, file_lst):
    #     params = np.loadtxt(img_dir + "/rendering_metadata.txt")
    #     img_file = os.path.join(img_dir, file_lst[num])
    #     # azimuth, elevation, in-plane rotation, distance, the field of view.
    #     param = params[num, :].astype(np.float32)
    #     # cam_mat, cam_pos = self.camera_info(self.degree2rad(param))
    #     img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)[:,:,:3].astype(np.float32) / 255.
    #     return img_arr, cam_mat, cam_pos

    def get_img(self, img_dir, num):
        img_h5 = os.path.join(img_dir, "%02d.h5"%num)
        trans_mat, obj_rot_mat = None, None
        with h5py.File(img_h5, 'r') as h5_f:
            trans_mat = h5_f["trans_mat"][:].astype(np.float32)
            obj_rot_mat = h5_f["obj_rot_mat"][:].astype(np.float32)
            if self.FLAGS.alpha:
                img_arr = h5_f["img_arr"][:].astype(np.float32)
                img_arr[:, :, :4] = img_arr[:,:,:4] / 255.
            else:
                img_raw = h5_f["img_arr"][:]
                img_arr = img_raw[:, :, :3]
                img_arr = np.clip(img_arr, 0, 255)
                img_arr = img_arr.astype(np.float32) / 255.

            return img_arr, trans_mat, obj_rot_mat

    def degree2rad(self, params):
        params[0] = np.deg2rad(params[0] + 180.0)
        params[1] = np.deg2rad(params[1])
        params[2] = np.deg2rad(params[2])
        return params

    def unit(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def camera_info(self, param):
        az_mat = self.get_az(param[0])
        el_mat = self.get_el(param[1])
        inl_mat = self.get_inl(param[2])
        cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
        cam_pos = self.get_cam_pos(param)
        return cam_mat, cam_pos

    def get_cam_pos(self, param):
        camX = 0
        camY = 0
        camZ = param[3]
        cam_pos = np.array([camX, camY, camZ])
        return -1 * cam_pos

    def get_az(self, az):
        cos = np.cos(az)
        sin = np.sin(az)
        mat = np.asarray([cos, 0.0, sin, 0.0, 1.0, 0.0, -1.0*sin, 0.0, cos], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat
    #
    def get_el(self, el):
        cos = np.cos(el)
        sin = np.sin(el)
        mat = np.asarray([1.0, 0.0, 0.0, 0.0, cos, -1.0*sin, 0.0, sin, cos], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat
    #
    def get_inl(self, inl):
        cos = np.cos(inl)
        sin = np.sin(inl)
        # zeros = np.zeros_like(inl)
        # ones = np.ones_like(inl)
        mat = np.asarray([cos, -1.0*sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        mat = np.reshape(mat, [3,3])
        return mat

    def get_batch(self, index):
        if index + self.FLAGS.batch_size > self.epoch_amount:
            index = index + self.FLAGS.batch_size - self.epoch_amount
        batch_locs, batch_norms, batch_dists = [],[],[]
        if self.FLAGS.alpha:
            batch_img = np.zeros((self.FLAGS.batch_size, self.FLAGS.img_h, self.FLAGS.img_w, 4), dtype=np.float32)
        else:
            batch_img = np.zeros((self.FLAGS.batch_size, self.FLAGS.img_h, self.FLAGS.img_w, 3), dtype=np.float32)
        batch_obj_rot_mat = np.zeros((self.FLAGS.batch_size, 3, 3), dtype=np.float32)
        batch_trans_mat = np.zeros((self.FLAGS.batch_size, 4, 3), dtype=np.float32)
        batch_cat_id = []
        batch_obj_nm = []
        batch_view_id = []
        batch_gt_pnts = []
        cnt = 0
        for i in range(index, index + self.FLAGS.batch_size):
            # print(index, self.order,i)
            single_obj = self.getitem(self.order[i])
            if single_obj == None:
                raise Exception("single mesh is None!")
            locs, norms, dists, img_dir, img_file_lst, cat_id, obj, num, gt_pnts = single_obj
            img, trans_mat, obj_rot_mat = self.get_img(img_dir, num)
            if self.FLAGS.filename is not None:
                if self.FLAGS.surf_num < locs.shape[0]:
                    choice = np.asarray(random.sample(range(locs.shape[0]), self.FLAGS.surf_num), dtype=np.int32)
                    locs, norms, dists = locs[choice], norms[choice], dists[choice]
                batch_locs.append(locs)
                batch_norms.append(norms)
                batch_dists.append(dists)
            batch_gt_pnts.append(gt_pnts)
            batch_img[cnt, ...] = img.astype(np.float32)
            batch_obj_rot_mat[cnt, ...] = obj_rot_mat
            batch_trans_mat[cnt, ...] = trans_mat
            batch_cat_id.append(cat_id)
            batch_obj_nm.append(obj)
            batch_view_id.append(num)
            cnt += 1
        batch_data = {'imgs': batch_img, 'obj_rot_mats': batch_obj_rot_mat, 'trans_mats': batch_trans_mat, 'cat_id': batch_cat_id, 'obj_nm': batch_obj_nm, 'view_id': batch_view_id}
        if self.FLAGS.filename is not None:
            batch_data['locs'] = np.array(batch_locs)
            batch_data['norms'] = np.array(batch_norms)
            batch_data['dists'] = np.array(batch_dists)
        batch_data["gt_pnts"] = batch_gt_pnts
        return batch_data

    def refill_data_order(self):
        temp_order = copy.deepcopy(self.data_order)
        cats_quota = {key: value for key, value in self.cats_limit.items()}
        np.random.shuffle(temp_order)
        pointer = 0
        epoch_order=[]
        while len(epoch_order) < self.epoch_amount:
            cat_id, _, _ = self.listinfo[temp_order[pointer]]
            if cats_quota[cat_id] > 0:
                epoch_order.append(temp_order[pointer])
                cats_quota[cat_id]-=1
            pointer+=1
        return epoch_order


    def work(self, epoch, index):
        if index == 0 and self.shuffle:
            self.order = self.refill_data_order()
            print("data order reordered!")
        return self.get_batch(index)

    def run(self):
        print("start running")
        while (self.bno // (self.num_batches* self.FLAGS.batch_size)) < self.FLAGS.max_epoch and not self.stopped:
            self.queue.put(self.work(self.bno // (self.num_batches* self.FLAGS.batch_size),
                                     self.bno % (self.num_batches * self.FLAGS.batch_size)))
            self.bno += self.FLAGS.batch_size

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()

if __name__ == '__main__':

    sys.path.append('../preprocessing/')
    import create_file_lst as create

    data = Pt_sdf_img(res=256, expr=1.5,
                      listinfo=[["03001627", "ff3581996365bdddc3bd24f986301745"],
                                ["03001627", "ff3581996365bdddc3bd24f986301745"]],
                      info=create.get_all_info(), maxnverts=6000, maxntris=50000,
                      minsurbinvox=4096, num_points=2048, batch_size=2, normalize=False, norm_color=True)
    batch1 = data.get_batch(0)
    print(batch1.keys())
    print(batch1["verts"].shape)
    print(batch1["nverts"])
    print(batch1["tris"].shape)
    print(batch1["ntris"])
    print(batch1["surfacebinvoxpc"].shape)
    print(batch1["sdf"].shape)
    print(batch1["sdf_params"])
    print(batch1["img"].shape, batch1["img"][0, 64, 64, :])
    print(batch1["img_cam"])

    # (2048, 3)
    cloud1 = batch1["surfacebinvoxpc"][0, ...]
    trans1 = batch1["img_cam"][0, ...]
    az1 = float(trans1[0] + 180) * math.pi / 180.0
    el1 = float(trans1[1]) * math.pi / 180.0
    in1 = float(trans1[2]) * math.pi / 180.0
    transmatrix_az1 = [[math.cos(az1), 0, math.sin(az1)],
                       [0, 1, 0],
                       [-math.sin(az1), 0, math.cos(az1)]]
    transmatrix_az1 = np.asarray(transmatrix_az1).astype(np.float32)
    transmatrix_el1 = [[1, 0, 0],
                       [0, math.cos(el1), -math.sin(el1)],
                       [0, math.sin(el1), math.cos(el1)]]
    transmatrix_el1 = np.asarray(transmatrix_el1).astype(np.float32)
    transmatrix_in1 = [[math.cos(in1), -math.sin(in1), 0],
                       [math.sin(in1), math.cos(in1), 0],
                       [0, 0, 1]]
    transmatrix_in1 = np.asarray(transmatrix_in1).astype(np.float32)

    trans = np.matmul(np.matmul(transmatrix_in1, transmatrix_el1), transmatrix_az1)
    translate1 = np.tile(np.expand_dims(np.asarray([-trans1[2], 0, 0]).astype(np.float32), axis=0), (2048, 1))
    points = np.matmul(cloud1, trans.T)
    np.savetxt("ff_rotate.xyz", points)
    np.savetxt("ff.xyz", cloud1)
