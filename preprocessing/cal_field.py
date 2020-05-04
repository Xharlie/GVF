import os
import numpy as np
from numpy import dot
from math import sqrt
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pycuda.driver as drv



def cal_field(pnts, gt_pnts, gpu=0):
    # print("gpu",gpu)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    if gpu < 0:
        import pycuda.autoinit
    else:
        drv.init()
        dev1 = drv.Device(gpu)
        ctx1 = dev1.make_context()

    mod = SourceModule("""
    include <math.h>
    #define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

    __device__ void compute_force_scalar(float dist)
    {
	    return 1/(pow(dist*1000,8)+1E-6)
    }

    __global__ void p2g(float *gvfs, float *pnts, float *gt_pnts, int pnt_num, int gt_num)
    {
        int p_id = blockIdx.x * blockDim.x + threadIdx.x;
        float px = pnts[p_id*3];
        float py = pnts[p_id*3+1];
        float pz = pnts[p_id*3+2];
        float x_dist, y_dist, z_dist, dist, force_sum=0, x_sum=0, y_sum=0, z_sum=0;
        for (int gt_id=0; gt_id<gt_num; gt_id++){
            x_dist = gt_pnts[gt_id*3] - px;
            y_dist = gt_pnts[gt_id*3+1] - py;
            z_dist = gt_pnts[gt_id*3+2] - pz;
            dist = sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);
            force = compute_force_scalar(dist);
            force_sum = force_sum + force;
            x_sum = x_sum + x_dist * force;
            y_sum = y_sum + y_dist * force;
            z_sum = z_sum + z_dist * force;
        }
        gvfs[p_id*3] = x_sum / force_sum;
        gvfs[p_id*3+1] = y_sum / force_sum;
        gvfs[p_id*3+2] = z_sum / force_sum;
    }
    """)

    kMaxThreadsPerBlock = 1024
    pnt_num = pnts.shape[0]
    gt_num = gt_pnts.shape[0]

    print("start to cal gvf gt field pnt num: ", gt_num)
    gvfs = np.zeros((pnt_num, gt_num, 3)).astype(np.float32)
    gridSize = int((pnt_num + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
    pnts_tries_ivt = mod.get_function("p2g")
    pnts_tries_ivt(drv.Out(gvfs), drv.In(np.float32(pnts)), drv.In(np.float32(gt_pnts)), np.int32(pnt_num), np.int32(gt_num), block=(kMaxThreadsPerBlock,1,1), grid=(gridSize,1))
    # print("ivt[0,0,:]", ivt[0,0,:])
    if gpu >= 0: 
        ctx1.pop()
    return gvfs