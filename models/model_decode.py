import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg, resnet_v1, resnet_v2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'data'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
print(os.path.join(BASE_DIR, 'models'))
import gvfnet
import unet_model_3d

def placeholder_inputs(scope='', FLAGS=None, num_pnts=None):
    if num_pnts is None:
        num_pnts = FLAGS.num_pnts
    with tf.compat.v1.variable_scope(scope) as sc:
        pnts_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, num_pnts, 3))
        if FLAGS.alpha:
            imgs_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.img_h, FLAGS.img_w, 4))
        else:
            imgs_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.img_h, FLAGS.img_w, 3))
        gvfs_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, num_pnts, 3))
        obj_rot_mat_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, 3, 3))
        trans_mat_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, 4, 3))
        onedge_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, num_pnts, 1))
    gvf = {}
    gvf['pnts'] = pnts_pl
    gvf['gvfs'] = gvfs_pl
    gvf['imgs'] = imgs_pl
    gvf['obj_rot_mats'] = obj_rot_mat_pl
    gvf['trans_mats'] = trans_mat_pl
    gvf['onedge'] = onedge_pl
    return gvf

def get_model(input_pls, is_training, bn=False, bn_decay=None, img_size = 224, FLAGS=None):

    if FLAGS.act == "relu":
        activation_fn = tf.nn.relu
    elif FLAGS.act == "elu":
        activation_fn = tf.nn.elu


    input_imgs = input_pls['imgs']
    input_pnts = input_pls['pnts']
    input_gvfs = input_pls['gvfs']
    input_onedge = input_pls['onedge']
    input_trans_mat = input_pls['trans_mats']
    input_obj_rot_mats = input_pls['obj_rot_mats']

    batch_size = input_imgs.get_shape()[0].value

    # endpoints
    end_points = {}
    end_points['pnts'] = input_pnts
    if FLAGS.rot:
        end_points['gt_gvfs_xyz'] = tf.matmul(input_gvfs, input_obj_rot_mats)
        end_points['pnts_rot'] = tf.matmul(input_pnts, input_obj_rot_mats)
    else:
        end_points['gt_gvfs_xyz'] = input_gvfs #* 10
        end_points['pnts_rot'] = input_pnts
    if FLAGS.edgeweight != 1.0:
        end_points['onedge'] = input_onedge
    end_points['imgs'] = input_imgs # B*H*W*3|4

    # Image extract features
    if input_imgs.shape[1] != img_size or input_imgs.shape[2] != img_size:
        if FLAGS.alpha:
            ref_img_rgb = tf.compat.v1.image.resize_bilinear(input_imgs[:,:,:,:3], [img_size, img_size])
            ref_img_alpha = tf.image.resize_nearest_neighbor(
                tf.expand_dims(input_imgs[:,:,:,3], axis=-1), [img_size, img_size])
            ref_img = tf.concat([ref_img_rgb, ref_img_alpha], axis = -1)
        else:
            ref_img = tf.compat.v1.image.resize_bilinear(input_imgs, [img_size, img_size])
    else:
        ref_img = input_imgs
    end_points['resized_ref_img'] = ref_img
    if FLAGS.encoder[:6] == "vgg_16":
        vgg.vgg_16.default_image_size = img_size
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(FLAGS.wd)):
            ref_feats_embedding, encdr_end_points = vgg.vgg_16(ref_img, num_classes=FLAGS.num_classes, is_training=False, scope='vgg_16', spatial_squeeze=False)
    end_points['img_embedding'] = ref_feats_embedding
    point_img_feat=None
    gvfs_feat=None
    # sample_img_points = get_img_points(input_pnts, input_trans_mat)  # B * N * 2
    dec3d = unet_model_3d(ref_feats_embedding, channel_size=(512, 256, 128, 64, 64), pool_size=(2, 2, 2), deconvolution=False, depth=4,
                  batch_normalization=False, activation_lst=["relu","relu","relu","relu","sigmoid"], res=True)

    # if FLAGS.img_feat_onestream:
    #     with tf.compat.v1.variable_scope("sdfimgfeat") as scope:
    #         if FLAGS.encoder[:3] == "vgg":
    #             conv1 = tf.compat.v1.image.resize_bilinear(encdr_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
    #             point_conv1 = tf.contrib.resampler.resampler(conv1, sample_img_points)
    #             conv2 = tf.compat.v1.image.resize_bilinear(encdr_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
    #             point_conv2 = tf.contrib.resampler.resampler(conv2, sample_img_points)
    #             conv3 = tf.compat.v1.image.resize_bilinear(encdr_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
    #             point_conv3 = tf.contrib.resampler.resampler(conv3, sample_img_points)
    #             conv4 = tf.compat.v1.image.resize_bilinear(encdr_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
    #             point_conv4 = tf.contrib.resampler.resampler(conv4, sample_img_points)
    #             point_img_feat = tf.concat(axis=2, values=[point_conv1, point_conv2, point_conv3, point_conv4]) # small
    #
    #         print("point_img_feat.shape", point_img_feat.get_shape())
    #         point_img_feat = tf.expand_dims(point_img_feat, axis=2)

    end_points['pred_gvfs_xyz'], end_points['pred_gvfs_dist'], end_points['pred_gvfs_direction'] = None, None, None
    if FLAGS.XYZ:
        end_points['pred_gvfs_xyz'] = gvfnet.xyz_gvfhead(gvfs_feat, batch_size, wd=FLAGS.wd, activation_fn=activation_fn)
        end_points['pred_gvfs_dist'] = tf.sqrt(tf.reduce_sum(tf.square(end_points['pred_gvfs_xyz']), axis=2, keepdims=True))
        end_points['pred_gvfs_direction'] = end_points['pred_gvfs_xyz'] / tf.maximum(end_points['pred_gvfs_dist'], 1e-6)
    else:
        end_points['pred_gvfs_dist'], end_points['pred_gvfs_direction'] = gvfnet.dist_direct_gvfhead(gvfs_feat, batch_size, wd=FLAGS.wd, activation_fn=activation_fn)
        end_points['pred_gvfs_xyz'] = end_points['pred_gvfs_direction'] * end_points['pred_gvfs_dist']

    end_points["sample_img_points"] = sample_img_points
    # end_points["ref_feats_embedding"] = ref_feats_embedding
    end_points["point_img_feat"] = point_img_feat

    return end_points

def get_img_points(sample_pc, trans_mat_right):
    # sample_pc B*N*3
    size_lst = sample_pc.get_shape().as_list()
    homo_pc = tf.concat((sample_pc, tf.ones((size_lst[0], size_lst[1], 1),dtype=np.float32)),axis= -1)
    print("homo_pc.get_shape()", homo_pc.get_shape())
    pc_xyz = tf.matmul(homo_pc, trans_mat_right)
    print("pc_xyz.get_shape()", pc_xyz.get_shape()) # B * N * 3
    pc_xy = tf.divide(pc_xyz[:,:,:2], tf.expand_dims(pc_xyz[:,:,2], axis = 2))
    mintensor = tf.constant([0.0,0.0], dtype=tf.float32)
    maxtensor = tf.constant([136.0,136.0], dtype=tf.float32)
    return tf.minimum(maxtensor, tf.maximum(mintensor, pc_xy))


def get_loss(end_points, regularization=True, FLAGS=None):

    gt_gvfs_xyz = end_points['gt_gvfs_xyz']
    gt_gvfs_dist = tf.sqrt(tf.reduce_sum(tf.square(gt_gvfs_xyz), axis=2, keepdims=True))
    print("gt_gvfs_dist.get_shape().as_list(): ", gt_gvfs_dist.get_shape().as_list())
    gt_gvfs_direction = gt_gvfs_xyz / tf.maximum(gt_gvfs_dist, 1e-6)
    end_points['gt_gvfs_dist'] = gt_gvfs_dist
    end_points['gt_gvfs_direction'] = gt_gvfs_direction

    if FLAGS.weight_type == 'propor':
        weight_mask = 1 / tf.maximum(gt_gvfs_dist, 1e-6)
    elif FLAGS.weight_type == 'ntanh':
        thresh = tf.constant(0.05, dtype=tf.float32)
        weight_mask = tf.cast(tf.less_equal(gt_gvfs_dist, thresh),dtype=tf.float32) \
              + tf.cast(tf.greater(gt_gvfs_dist, thresh),dtype=tf.float32) * (tf.tanh(thresh - gt_gvfs_dist) + 1)
    else:
        weight_mask = tf.ones([FLAGS.batch_size, FLAGS.num_pnts, 1], dtype=tf.float32)
    end_points['weighed_mask'] = weight_mask

    ###########gvf
    print('gt_gvfs_xyz, gt_dist, gt_direction, weight_mask shapes:', gt_gvfs_xyz.get_shape().as_list(), gt_gvfs_dist.get_shape().as_list(), gt_gvfs_direction.get_shape().as_list(), weight_mask.get_shape().as_list())
    ################
    # Compute loss #
    ################
    end_points['losses'] = {}

    ############### accuracy
    gvfs_xyz_diff = tf.abs(gt_gvfs_xyz - end_points['pred_gvfs_xyz'])
    gvfs_xyz_avg_diff = tf.reduce_mean(gvfs_xyz_diff)

    gvfs_locnorm_diff = tf.norm(gvfs_xyz_diff, ord='euclidean', axis=2, keepdims=True)
    gvfs_locsqrnorm_diff = tf.reduce_sum(tf.square(gvfs_xyz_diff), axis=2, keepdims=True)
    gvfs_locnorm_avg_diff = tf.reduce_mean(gvfs_locnorm_diff)
    gvfs_locsqrnorm_avg_diff = tf.reduce_mean(gvfs_locsqrnorm_diff)
    if FLAGS.edgeweight != 1.0:
        onedge = end_points['onedge']
        onedge_count = tf.reduce_sum(onedge)
        ontri= 1.0 - onedge
        ontri_count = tf.reduce_sum(ontri)
        gvfs_locnorm_onedge_sum_diff = tf.reduce_sum(tf.multiply(onedge, gvfs_locnorm_diff))
        gvfs_locnorm_ontri_sum_diff = tf.reduce_sum(tf.multiply(ontri, gvfs_locnorm_diff))

    gvfs_dist_diff = tf.abs(gt_gvfs_dist - end_points['pred_gvfs_dist'])
    gvfs_dist_avg_diff = tf.reduce_mean(gvfs_dist_diff)
    gvfs_direction_diff = tf.reduce_sum(tf.multiply(gt_gvfs_direction, end_points['pred_gvfs_direction']), axis=2, keepdims=True)
    gvfs_direction_avg_diff = tf.reduce_mean(gvfs_direction_diff)
    gvfs_direction_abs_diff = tf.abs(gvfs_direction_diff)
    gvfs_direction_abs_avg_diff = tf.reduce_mean(gvfs_direction_abs_diff)
    end_points['lvl'] = {}
    if FLAGS.distlimit is not None:
        end_points['lvl']["xyz_lvl_diff"], end_points['lvl']["locnorm_lvl_diff"], end_points['lvl']["locsqrnorm_lvl_diff"], end_points['lvl']["dist_lvl_diff"], end_points['lvl']["direction_lvl_diff"], end_points['lvl']["direction_abs_lvl_diff"], end_points['lvl']["num"] \
            = gen_lvl_diff(gvfs_xyz_diff, gvfs_locnorm_diff, gvfs_locsqrnorm_diff, gvfs_dist_diff, gvfs_direction_diff, gvfs_direction_abs_diff, FLAGS.distlimit, gt_gvfs_dist)

    gvfs_xyz_loss, gvfs_dist_loss, gvfs_direction_loss, gvfs_direction_abs_loss, gvfs_locnorm_loss, gvfs_locsqrnorm_loss = 0., 0., 0., 0., 0., 0.
    
    if FLAGS.lossw[0] !=0:
        if FLAGS.weight_type == "non":
            gvfs_xyz_loss = gvfs_xyz_avg_diff
            if FLAGS.edgeweight != 1.0:
                gvfs_xyz_loss = tf.reduce_mean(tf.multiply(onedge, gvfs_xyz_avg_diff) * FLAGS.edgeweight + tf.multiply(ontri, gvfs_xyz_avg_diff))
        else:
            gvfs_xyz_loss = tf.reduce_mean(gvfs_xyz_diff * weight_mask)
        end_points['losses']['gvfs_xyz_loss'] = gvfs_xyz_loss
    if FLAGS.lossw[1] != 0:
        if FLAGS.weight_type == "non":
            gvfs_locnorm_loss = gvfs_locnorm_avg_diff
            if FLAGS.edgeweight != 1.0:
                gvfs_locnorm_loss = tf.reduce_mean(tf.multiply(onedge, gvfs_locnorm_diff) * FLAGS.edgeweight + tf.multiply(ontri, gvfs_locnorm_diff))
        else:
            gvfs_locnorm_loss = tf.reduce_mean(gvfs_locnorm_diff * weight_mask)
        end_points['losses']['gvfs_locnorm_loss'] = gvfs_locnorm_loss
    if FLAGS.lossw[2] != 0:
        if FLAGS.weight_type == "non":
            gvfs_locsqrnorm_loss = gvfs_locsqrnorm_avg_diff
            if FLAGS.edgeweight != 1.0:
                gvfs_locsqrnorm_loss = tf.reduce_mean(tf.multiply(onedge, gvfs_locsqrnorm_diff) * FLAGS.edgeweight + tf.multiply(ontri, gvfs_locsqrnorm_diff))
        else:
            gvfs_locsqrnorm_loss = tf.reduce_mean(gvfs_locsqrnorm_diff * weight_mask)
        end_points['losses']['gvfs_locsqrnorm_loss'] = gvfs_locsqrnorm_loss
    if FLAGS.lossw[3] != 0:
        if FLAGS.weight_type == "non":
            gvfs_dist_loss = gvfs_dist_avg_diff
            if FLAGS.edgeweight != 1.0:
                gvfs_dist_loss = tf.reduce_mean(tf.multiply(onedge, gvfs_dist_diff) * FLAGS.edgeweight + tf.multiply(ontri, gvfs_dist_diff))
        else:
            gvfs_dist_loss = tf.reduce_mean(gvfs_dist_diff * weight_mask)
        end_points['losses']['gvfs_dist_loss'] = gvfs_dist_loss
    if FLAGS.lossw[4] != 0:
        if FLAGS.edgeweight != 1.0:
            gvfs_direction_loss = tf.reduce_mean(tf.multiply(onedge, (1.0-gvfs_direction_diff)) * FLAGS.edgeweight + tf.multiply(ontri, (1.0-gvfs_direction_diff)))
        else:
            gvfs_direction_loss = tf.reduce_mean((1.0-gvfs_direction_diff))
        end_points['losses']['gvfs_direction_loss'] = gvfs_direction_loss
    if FLAGS.lossw[5] != 0:
        if FLAGS.edgeweight != 1.0:
            gvfs_direction_abs_loss = tf.reduce_mean(tf.multiply(onedge, (1.0-gvfs_direction_abs_diff)) * FLAGS.edgeweight + tf.multiply(ontri, (1.0-gvfs_direction_abs_diff)))
        else:
            gvfs_direction_abs_loss = tf.reduce_mean((1.0-gvfs_direction_abs_diff))
        end_points['losses']['gvfs_direction_abs_loss'] = gvfs_direction_abs_loss

    loss = FLAGS.lossw[0] * gvfs_xyz_loss + FLAGS.lossw[1] * gvfs_locnorm_loss + FLAGS.lossw[2] * gvfs_locsqrnorm_loss + FLAGS.lossw[3] * gvfs_dist_loss + FLAGS.lossw[4] * gvfs_direction_loss + FLAGS.lossw[5] * gvfs_direction_abs_loss
    end_points['losses']['gvfs_xyz_avg_diff'] = gvfs_xyz_avg_diff
    end_points['losses']['gvfs_dist_avg_diff'] = gvfs_dist_avg_diff
    end_points['losses']['gvfs_direction_avg_diff'] = gvfs_direction_avg_diff
    end_points['losses']['gvfs_direction_abs_avg_diff'] = gvfs_direction_abs_avg_diff
    end_points['losses']['gvfs_locnorm_avg_diff'] = gvfs_locnorm_avg_diff
    end_points['losses']['gvfs_locsqrnorm_avg_diff'] = gvfs_locsqrnorm_avg_diff
    end_points['losses']['loss'] = loss
    if FLAGS.edgeweight != 1.0:
        end_points['losses']['gvfs_locnorm_onedge_sum_diff'] = gvfs_locnorm_onedge_sum_diff
        end_points['losses']['gvfs_locnorm_ontri_sum_diff'] = gvfs_locnorm_ontri_sum_diff
        end_points['losses']['onedge_count'] = onedge_count
        end_points['losses']['ontri_count'] = ontri_count
    ############### weight decay
    if regularization and FLAGS.wd>0:
        vgg_regularization_loss = tf.add_n(tf.compat.v1.losses.get_regularization_losses())
        decoder_regularization_loss = tf.add_n(tf.compat.v1.get_collection('regularizer'))
        end_points['losses']['regularization'] = (vgg_regularization_loss + decoder_regularization_loss)
        loss += (vgg_regularization_loss + decoder_regularization_loss)
    end_points['losses']['overall_loss'] = loss
    return loss, end_points


def gen_lvl_diff(gvfs_xyz_diff, gvfs_locnorm_diff, gvfs_locsqrnorm_diff, gvfs_dist_diff, gvfs_direction_diff, gvfs_direction_abs_diff, distlimit, gt_gvfs_dist):
    xyz_lvl_diff, locnorm_lvl_diff, locsqrnorm_lvl_diff, dist_lvl_diff, direction_lvl_diff, direction_abs_lvl_diff = [],[],[],[],[],[]
    lvl_num = []
    ones = tf.ones_like(gt_gvfs_dist, dtype=tf.float32)
    for i in range(len(distlimit)//2):
        upper = distlimit[i*2] * ones
        lower = distlimit[i*2+1] * ones
        floatmask = tf.cast(tf.compat.v1.logical_and(tf.compat.v1.less_equal(gt_gvfs_dist, upper), tf.compat.v1.greater(gt_gvfs_dist,lower)), dtype=tf.float32)
        lvl_num.append(tf.reduce_sum(floatmask))
        xyz_lvl_diff.append(tf.reduce_sum(floatmask*gvfs_xyz_diff))
        locnorm_lvl_diff.append(tf.reduce_sum(floatmask*gvfs_locnorm_diff))
        locsqrnorm_lvl_diff.append(tf.reduce_sum(floatmask*gvfs_locsqrnorm_diff))
        dist_lvl_diff.append(tf.reduce_sum(floatmask*gvfs_dist_diff))
        direction_lvl_diff.append(tf.reduce_sum(floatmask*gvfs_direction_diff))
        direction_abs_lvl_diff.append(tf.reduce_sum(floatmask*gvfs_direction_abs_diff))
    lvl_num = tf.convert_to_tensor(lvl_num)
    xyz_lvl_diff = tf.convert_to_tensor(xyz_lvl_diff)
    print("xyz_lvl_diff tensorf .get_shape().as_list()", xyz_lvl_diff.get_shape().as_list())
    locnorm_lvl_diff = tf.convert_to_tensor(locnorm_lvl_diff)
    locsqrnorm_lvl_diff = tf.convert_to_tensor(locsqrnorm_lvl_diff)
    dist_lvl_diff = tf.convert_to_tensor(dist_lvl_diff)
    direction_lvl_diff = tf.convert_to_tensor(direction_lvl_diff)
    direction_abs_lvl_diff = tf.convert_to_tensor(direction_abs_lvl_diff)
    return xyz_lvl_diff, locnorm_lvl_diff, locsqrnorm_lvl_diff, dist_lvl_diff, direction_lvl_diff, direction_abs_lvl_diff, lvl_num