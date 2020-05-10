import torch
import torch.optim as optim
# from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
# import matplotlib; matplotlib.use('Agg')
# from im2mesh import config, data
# from im2mesh.checkpoints import CheckpointIO

device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--encoder', type=str, default='vgg_16', help='encoder model: vgg_16, resnet_v1_50, resnet_v1_101, resnet_v2_50, resnet_v2_101')
parser.add_argument('--category', type=str, default="all", help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint', help='Log dir [default: log]')
parser.add_argument('--num_pnts', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--res', type=int, default=16, help='Point Number [default: 2048]')
parser.add_argument('--uni_num', type=int, default=1024, help='Point Number [default: 2048]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=28, help='Batch Size during training [default: 32]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Initial learning rate [default: 0.001]')
parser.add_argument('--wd', type=float, default=1e-6, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--restore_model', default='', help='restore_model') #checkpoint/sdf_2d3d_sdfbasic2_nowd
parser.add_argument('--restore_modelcnn', default='', help='restore_model')#../models/CNN/pretrained_model/vgg_16.ckpt

parser.add_argument('--train_lst_dir', default=lst_dir, help='train mesh data list')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--decay_step', type=int, default=5, help='Decay step for lr decay [default: 1000000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--weight_type', type=str, default="ntanh")
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--binary', action='store_true')
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--act', type=str, default="relu")
parser.add_argument('--source', type=str, default="saved")
parser.add_argument('--edgeweight', type=float, default=1.0)
parser.add_argument('--rot', action='store_true')
parser.add_argument('--XYZ', action='store_true')
parser.add_argument('--decoder', type=str, default='norm')
parser.add_argument('--cam_est', action='store_true')
parser.add_argument('--cat_limit', type=int, default=1168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--multi_view', action='store_true')
parser.add_argument('--bn', action='store_true')
parser.add_argument('--manifold', action='store_true')
parser.add_argument('--lossw', nargs='+', action='store', default=[0.0, 1.0, 0.0, 0.0, 1.0, 0.0], help="xyz, locnorm, locsqrnorm, dist, dirct, drct_abs")
parser.add_argument('--distlimit', nargs='+', action='store', type=str, default=[1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.18, 0.18, 0.16, 0.16, 0.14, 0.14, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, -0.01])
parser.add_argument('--surfrange', nargs='+', action='store', default=[0.0, 0.15], help="lower bound, upperbound")


FLAGS = parser.parse_args()
FLAGS.lossw = [float(i) for i in FLAGS.lossw]
FLAGS.distlimit = [float(i) for i in FLAGS.distlimit]
FLAGS.surfrange = [float(i) for i in FLAGS.surfrange]

print(FLAGS)