import os
import tensorflow as tf
import numpy as np


def img_alexnet_layers(img, batch_size, output_dim, stage, model_weights, with_tanh=True, val_batch_size=32):
    deep_param_img = {}
    train_layers = []
    train_last_layer = []
    print("loading img model from %s" % model_weights)
    net_data = dict(np.load(model_weights, encoding='bytes').item())
    print(list(net_data.keys()))

    # swap(2,1,0), bgr -> rgb
    reshaped_image = tf.cast(img, tf.float32)[:, :, :, ::-1]

    height = 227
    width = 227

    # Randomly crop a [height, width] section of each image
    with tf.name_scope('preprocess'):
        def train_fn():
            return tf.stack([tf.random_crop(tf.image.random_flip_left_right(each), [height, width, 3])
                             for each in tf.unstack(reshaped_image, batch_size)])

        def val_fn():
            unstacked = tf.unstack(reshaped_image, val_batch_size)

            def crop(img, x, y): return tf.image.crop_to_bounding_box(
                img, x, y, width, height)

            def distort(f, x, y): return tf.stack(
                [crop(f(each), x, y) for each in unstacked])

            def distort_raw(x, y): return distort(lambda x: x, x, y)

            def distort_fliped(x, y): return distort(
                tf.image.flip_left_right, x, y)
            distorted = tf.concat([distort_fliped(0, 0), distort_fliped(28, 0),
                                   distort_fliped(
                                       0, 28), distort_fliped(28, 28),
                                   distort_fliped(14, 14), distort_raw(0, 0),
                                   distort_raw(28, 0), distort_raw(0, 28),
                                   distort_raw(28, 28), distort_raw(14, 14)], 0)

            return distorted
        distorted = tf.cond(stage > 0, val_fn, train_fn)

        # Zero-mean input
        mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[
                           1, 1, 1, 3], name='img-mean')
        distorted = distorted - mean

    # Conv1
    # Output 96, kernel 11, stride 4
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(net_data['conv1'][0], name='weights')
        conv = tf.nn.conv2d(distorted, kernel, [1, 4, 4, 1], padding='VALID')
        biases = tf.Variable(net_data['conv1'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(out, name=scope)
        deep_param_img['conv1'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pool1
    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # LRN1
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(pool1,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # Conv2
    # Output 256, pad 2, kernel 5, group 2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(net_data['conv2'][0], name='weights')
        group = 2

        def convolve(i, k): return tf.nn.conv2d(
            i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(lrn1, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k)
                         for i, k in zip(input_groups, kernel_groups)]
        # Concatenate the groups
        conv = tf.concat(output_groups, 3)

        biases = tf.Variable(net_data['conv2'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(out, name=scope)
        deep_param_img['conv2'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pool2
    pool2 = tf.nn.max_pool(conv2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')

    # LRN2
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(pool2,
                                              depth_radius=radius,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias)

    # Conv3
    # Output 384, pad 1, kernel 3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(net_data['conv3'][0], name='weights')
        conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(net_data['conv3'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(out, name=scope)
        deep_param_img['conv3'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Conv4
    # Output 384, pad 1, kernel 3, group 2
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(net_data['conv4'][0], name='weights')
        group = 2

        def convolve(i, k): return tf.nn.conv2d(
            i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(conv3, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k)
                         for i, k in zip(input_groups, kernel_groups)]
        # Concatenate the groups
        conv = tf.concat(output_groups, 3)
        biases = tf.Variable(net_data['conv4'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(out, name=scope)
        deep_param_img['conv4'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Conv5
    # Output 256, pad 1, kernel 3, group 2
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(net_data['conv5'][0], name='weights')
        group = 2

        def convolve(i, k): return tf.nn.conv2d(
            i, k, [1, 1, 1, 1], padding='SAME')
        input_groups = tf.split(conv4, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k)
                         for i, k in zip(input_groups, kernel_groups)]
        # Concatenate the groups
        conv = tf.concat(output_groups, 3)
        biases = tf.Variable(net_data['conv5'][1], name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(out, name=scope)
        deep_param_img['conv5'] = [kernel, biases]
        train_layers += [kernel, biases]

    # Pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')

    # FC6
    # Output 4096
    with tf.name_scope('fc6'):
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.Variable(net_data['fc6'][0], name='weights')
        fc6b = tf.Variable(net_data['fc6'][1], name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        fc6 = tf.cond(stage > 0, lambda: fc6, lambda: tf.nn.dropout(fc6, 0.5))
        fc6o = tf.nn.relu(fc6l)
        deep_param_img['fc6'] = [fc6w, fc6b]
        train_layers += [fc6w, fc6b]

    # FC7
    # Output 4096
    with tf.name_scope('fc7'):
        fc7w = tf.Variable(net_data['fc7'][0], name='weights')
        fc7b = tf.Variable(net_data['fc7'][1], name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)
        fc7 = tf.cond(stage > 0, lambda: fc7, lambda: tf.nn.dropout(fc7, 0.5))
        deep_param_img['fc7'] = [fc7w, fc7b]
        train_layers += [fc7w, fc7b]

    # FC8
    # Output output_dim
    with tf.name_scope('fc8'):
        # Differ train and val stage by 'fc8' as key
        if 'fc8' in net_data:
            fc8w = tf.Variable(net_data['fc8'][0], name='weights')
            fc8b = tf.Variable(net_data['fc8'][1], name='biases')
        else:
            fc8w = tf.Variable(tf.random_normal([4096, output_dim],
                                                dtype=tf.float32,
                                                stddev=1e-2), name='weights')
            fc8b = tf.Variable(tf.constant(0.0, shape=[output_dim],
                                           dtype=tf.float32), name='biases')
        fc8l = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b)
        if with_tanh:
            fc8_t = tf.nn.tanh(fc8l)
        else:
            fc8_t = fc8l

        def val_fn1():
            concated = tf.concat([tf.expand_dims(i, 0)
                                  for i in tf.split(fc8_t, 10, 0)], 0)
            return tf.reduce_mean(concated, 0)
        fc8 = tf.cond(stage > 0, val_fn1, lambda: fc8_t)
        deep_param_img['fc8'] = [fc8w, fc8b]
        train_last_layer += [fc8w, fc8b]

    print("img model loading finished")
    # Return outputs
    return fc8, deep_param_img, train_layers, train_last_layer


def txt_mlp_layers(txt, txt_dim, output_dim, stage, model_weights=None, with_tanh=True):
    deep_param_txt = {}
    train_layers = []
    train_last_layer = []

    if model_weights is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_weights = os.path.join(
            dir_path, "pretrained_model/reference_pretrain.npy")

    net_data = dict(np.load(model_weights, encoding='bytes').item())

    # txt_fc1
    with tf.name_scope('txt_fc1'):
        if 'txt_fc1' not in net_data:
            txt_fc1w = tf.Variable(tf.truncated_normal([txt_dim, 4096],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
            txt_fc1b = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                   trainable=True, name='biases')
        else:
            txt_fc1w = tf.Variable(net_data['txt_fc1'][0], name='weights')
            txt_fc1b = tf.Variable(net_data['txt_fc1'][1], name='biases')
        txt_fc1l = tf.nn.bias_add(tf.matmul(txt, txt_fc1w), txt_fc1b)

        txt_fc1 = tf.cond(stage > 0, lambda: tf.nn.relu(
            txt_fc1l), lambda: tf.nn.dropout(tf.nn.relu(txt_fc1l), 0.5))

        train_layers += [txt_fc1w, txt_fc1b]
        deep_param_txt['txt_fc1'] = [txt_fc1w, txt_fc1b]

    # txt_fc2
    with tf.name_scope('txt_fc2'):
        if 'txt_fc2' not in net_data:
            txt_fc2w = tf.Variable(tf.truncated_normal([4096, output_dim],
                                                       dtype=tf.float32,
                                                       stddev=1e-2), name='weights')
            txt_fc2b = tf.Variable(tf.constant(0.0, shape=[output_dim], dtype=tf.float32),
                                   trainable=True, name='biases')
        else:
            txt_fc2w = tf.Variable(net_data['txt_fc2'][0], name='weights')
            txt_fc2b = tf.Variable(net_data['txt_fc2'][1], name='biases')

        txt_fc2l = tf.nn.bias_add(tf.matmul(txt_fc1, txt_fc2w), txt_fc2b)
        if with_tanh:
            txt_fc2 = tf.nn.tanh(txt_fc2l)
        else:
            txt_fc2 = txt_fc2l

        train_layers += [txt_fc2w, txt_fc2b]
        train_last_layer += [txt_fc2w, txt_fc2b]
        deep_param_txt['txt_fc2'] = [txt_fc2w, txt_fc2b]

    # return the output of text layer
    return txt_fc2, deep_param_txt, train_layers, train_last_layer
