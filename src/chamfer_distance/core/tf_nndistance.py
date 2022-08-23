import os

import tensorflow as tf
from tensorflow.python.framework import ops

module_path = os.path.abspath(os.path.dirname(__file__))
nn_distance_module = tf.load_op_library(os.path.join(module_path, 'tf_nndistance_so.so'))


def nn_distance(xyz1, xyz2):
    """
    Computes the distance of nearest neighbors for a pair of point clouds
    input: xyz1: (batch_size,#points_1,3)  the first point cloud
    input: xyz2: (batch_size,#points_2,3)  the second point cloud
    output: dist1: (batch_size,#point_1)   distance from first to second
    output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second
    output: dist2: (batch_size,#point_2)   distance from second to first
    output: idx2:  (batch_size,#point_2)   nearest neighbor from second to first
    """
    return nn_distance_module.nn_distance(xyz1, xyz2)


# @tf.RegisterShape('NnDistance')
def _nn_distance_shape(op):
    shape1 = op.inputs[0].get_shape().with_rank(3)
    shape2 = op.inputs[1].get_shape().with_rank(3)
    return [tf.TensorShape([shape1.dims[0], shape1.dims[1]]), tf.TensorShape([shape1.dims[0], shape1.dims[1]]),
            tf.TensorShape([shape2.dims[0], shape2.dims[1]]), tf.TensorShape([shape2.dims[0], shape2.dims[1]])]


@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op, grad_dist1, grad_idx1, grad_dist2, grad_idx2):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    idx1 = op.outputs[1]
    idx2 = op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1, xyz2, grad_dist1, idx1, grad_dist2, idx2)


def _demo():
    import numpy as np
    import random
    import time
    random.seed(100)
    np.random.seed(100)
    with tf.compat.v1.Session('') as sess:
        xyz1 = np.random.randn(32, 16384, 3).astype('float32')
        xyz2 = np.random.randn(32, 1024, 3).astype('float32')
        with tf.device('/gpu:0'):
            inp1 = tf.Variable(xyz1)
            inp2 = tf.constant(xyz2)
            reta, retb, retc, retd = nn_distance(inp1, inp2)
            loss = tf.reduce_sum(input_tensor=reta) + tf.reduce_sum(input_tensor=retc)
            train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
        sess.run(tf.compat.v1.initialize_all_variables())
        t0 = time.time()
        t1 = t0
        best = 1e100
        for i in range(200):
            trainloss, _ = sess.run([loss, train])
            newt = time.time()
            best = min(best, newt - t1)
            print(i, trainloss, (newt - t0) / (i + 1), best)
            t1 = newt


if __name__ == '__main__':
    _demo()
