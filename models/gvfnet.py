import tensorflow as tf
import tf_util



def get_grid_loc(span, dim, bsize):
    vlength = span * 2 / dim
    pos = (tf.range(0.0, dim, delta=1.0) + 0.5) * vlength
    x, y, z = tf.meshgrid(pos, pos, pos)
    xyz = tf.reshape(tf.concat([x, y, z], axis = -1), (1,-1,3))
    return tf.tile(xyz, (bsize,1,1))

def get_decoder_feat(src_pc, dec3d, dim):
    vlength = 2.14 / dim
    print("vlength", vlength)
    bottom = tf.reshape(tf.constant([-1.07,-1.07,-1.07]),(1,1,3))
    pc_rebase = src_pc - bottom
    pc_ind = tf.maximum(tf.minimum(dim-1.0, tf.math.floordiv(pc_rebase, vlength)), 0.0)
    pc_relative = pc_rebase - (pc_ind+0.5) * vlength
    dec_feats_pnts = tf.expand_dims(tf.gather_nd(dec3d, tf.cast(pc_ind,tf.int32), batch_dims=1),axis=-2)
    print("dec3d.shape, dec_feats_pnts.shape", dec3d.get_shape().as_list(), dec_feats_pnts.get_shape().as_list())
    return dec_feats_pnts, pc_relative, pc_ind

def get_gvf_decoderfeat(src_pc, decoderfeats, is_training, batch_size, bn, bn_decay, wd=None, activation_fn=tf.nn.relu):

    net = tf_util.conv2d(tf.expand_dims(src_pc,2), min(64,decoderfeats.get_shape()[-1].value//2), [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv1')
    net = tf_util.conv2d(net, min(128,decoderfeats.get_shape()[-1].value), [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn,bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2')
    # net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv3')

    print( 'net', net.shape)
    print( 'decoderfeats', decoderfeats.shape)
    concat = tf.concat(axis=3, values=[net, decoderfeats])

    net = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv2')

    return net


def get_gvf_basic(src_pc, globalfeats, is_training, batch_size, bn, bn_decay, wd=None, activation_fn=tf.nn.relu):

    net = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn,bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2')
    net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'net', net.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net, globalfeats_expand])

    net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv2')

    return net


def get_gvf_basic_imgfeat_onestream(src_pc, globalfeats, point_feat, is_training, batch_size, bn, bn_decay, wd=None, activation_fn=tf.nn.relu):

    net = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2')
    net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print('net', net.shape)
    print('globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net, globalfeats_expand, point_feat])

    net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv2')

    return net


def get_gvf_att_imgfeat(src_pc, globalfeats, point_feat, is_training, batch_size, bn, bn_decay, wd=None, activation_fn=tf.nn.relu):

    # globalfeats: b X 1024
    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])

    gw = tf_util.conv2d(globalfeats, 256 * 8, [1, 1], padding='VALID', stride=[1, 1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='gwfold/conv1')
    gw = tf_util.conv2d(gw, 256 * 64, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='gwfold/conv2') # b*1*1*  256 * 64
    gw = tf.reshape(gw, [batch_size, 64, 256]) # b * 64 * 256


    gb = tf_util.conv2d(globalfeats, 256, [1, 1], padding='VALID', stride=[1, 1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='bwfold/conv1')
    gb = tf_util.conv2d(gb, 256, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='bwfold/conv2') # b*1*1* 256
    gb = tf.tile(tf.reshape(gb, [batch_size, 1, 256]), [1, src_pc.get_shape()[1], 1]) # b* 2048 * 256


    pc_f = tf_util.conv2d(tf.expand_dims(src_pc,2), 32, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv1')
    pc_f = tf_util.conv2d(pc_f, 64, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2') # b * 2048 * 1 * 64
    pc_f = tf.reshape(pc_f, [batch_size, src_pc.get_shape()[1], 64])  # b * 2048 * 64


    pc_f = tf.linalg.matmul(pc_f, gw) + gb # b* 2048 * 256
    pc_f = tf.expand_dims(pc_f, axis = 2) # b* 2048 * 1 * 128
    concat = tf.concat(axis=3, values=[pc_f, point_feat])


    net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv2')

    return net


def get_gvf_basic_imgfeat_onestream_skip(src_pc, globalfeats, point_feat, is_training, batch_size, bn, bn_decay, wd=None, activation_fn=tf.nn.relu):

    net1 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net1, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2')
    net3 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print('skip net3', net3.shape)
    print('globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net3, globalfeats_expand])

    net4 = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv1')

    concat = tf.concat(axis=3, values=[point_feat, net3])
    print('skip point_feat', point_feat.shape)

    net5 = tf_util.conv2d(concat, 512, [1, 1], padding='VALID', stride=[1, 1], activation_fn=activation_fn,
                          bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv2')

    net6 = tf_util.conv2d(net5+net4, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv3')

    return net6


def get_gvf_basic_imgfeat_twostream(src_pc, point_feat, is_training, batch_size, bn, bn_decay, wd=None, activation_fn=tf.nn.relu):

    net = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2')
    net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[net, point_feat])

    net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv2')

    return net

def xyz_gvfhead(net, batch_size, wd=None, activation_fn=tf.nn.relu):

    pred = tf_util.conv2d(net, 3, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='head/xyz_conv')

    pred = tf.reshape(pred, [batch_size, -1, 3])

    return pred

def dist_direct_gvfhead(net, batch_size, wd=None, activation_fn=tf.nn.relu):

    direction = tf_util.conv2d(net, 3, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='head/direction_conv')

    norm = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(direction), axis=3, keepdims=True)), 1e-6)
    print("norm.get_shape().as_list(): ", norm.get_shape().as_list())
    direction = tf.reshape(direction/norm, [batch_size, -1, 3])

    pred_dist = tf.math.abs(tf_util.conv2d(net, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='head/dist_conv'))

    pred_dist = tf.reshape(pred_dist, [batch_size, -1, 1])

    return pred_dist, direction

