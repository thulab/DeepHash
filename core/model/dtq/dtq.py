import os
import random
import shutil
import time
from datetime import datetime
from math import ceil

import numpy as np
import tensorflow as tf
from sklearn.cluster import MiniBatchKMeans

from architecture.single_model import img_alexnet_layers
from distance.tfversion import distance
from evaluation import MAPs_CQ


class DTQ(object):
    def __init__(self, config):
        # Initialize setting
        np.set_printoptions(precision=4)

        with tf.name_scope('stage'):
            # 0 for training, 1 for validation
            self.stage = tf.placeholder_with_default(tf.constant(0), [])

        self.output_dim = config.output_dim
        self.n_class = config.label_dim

        self.subspace_num = config.subspace
        self.subcenter_num = config.subcenter
        self.code_batch_size = config.code_batch_size
        self.cq_lambda = config.cq_lambda
        self.max_iter_update_Cb = config.max_iter_update_Cb
        self.max_iter_update_b = config.max_iter_update_b

        self.batch_size = 3 * config.batch_size
        self.val_batch_size = config.val_batch_size
        self.max_epoch = config.epochs
        self.img_model = config.img_model
        self.with_tanh = config.with_tanh
        self.dist_type = config.dist_type
        self.learning_rate = config.lr
        self.decay_factor = config.decay_factor
        self.decay_step = config.decay_step
        self.val_freq = config.val_freq

        self.triplet_margin = config.triplet_margin
        self.select_strategy = config.select_strategy
        self.n_part = config.n_part
        self.triplet_thresold = config.triplet_thresold

        self.finetune_all = config.finetune_all

        self.file_name = 'lr_{}_margin_{}_{}_dim_{}_subspace_{}_subcenter_{}_{}_n_part_{}_{}_ds_{}'.format(
                self.learning_rate,
                self.triplet_margin,
                self.select_strategy,
                self.output_dim,
                self.subspace_num,
                self.subcenter_num,
                self.dist_type,
                self.n_part,
                self.triplet_thresold,
                config.dataset)
        self.save_dir = os.path.join(config.save_dir, self.file_name + '.npy')
        self.log_dir = config.log_dir

        # Setup session
        config_proto = tf.ConfigProto()
        config_proto.gpu_options.allow_growth = True
        config_proto.allow_soft_placement = True
        self.sess = tf.Session(config=config_proto)

        # Create variables and placeholders
        self.img = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.model_weights = config.model_weights
        self.img_last_layer, self.deep_param_img, self.train_layers, self.train_last_layer = self.load_model()

        with tf.name_scope('quantization'):
            self.C = tf.Variable(tf.random_uniform(
                                    [self.subspace_num * self.subcenter_num, self.output_dim],
                                    minval=-1, maxval=1, dtype=tf.float32, name='centers'))
            self.deep_param_img['C'] = self.C

            self.img_output_all = tf.placeholder(tf.float32, [None, self.output_dim])
            self.img_b_all = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])

            self.b_img = tf.placeholder(tf.float32, [None, self.subspace_num * self.subcenter_num])
            self.ICM_m = tf.placeholder(tf.int32, [])
            self.ICM_b_m = tf.placeholder(tf.float32, [None, self.subcenter_num])
            self.ICM_b_all = tf.placeholder(tf.float32, [None, self.subcenter_num * self.subspace_num])
            self.ICM_X = tf.placeholder(tf.float32, [self.code_batch_size, self.output_dim])
            self.ICM_C_m = tf.slice(self.C, [self.ICM_m * self.subcenter_num, 0], [self.subcenter_num, self.output_dim])
            self.ICM_X_residual = self.ICM_X - tf.matmul(self.ICM_b_all, self.C) + tf.matmul(self.ICM_b_m, self.ICM_C_m)
            ICM_X_expand = tf.expand_dims(self.ICM_X_residual, 1)
            ICM_C_m_expand = tf.expand_dims(self.ICM_C_m, 0)

            ICM_Loss = tf.reduce_sum(tf.square(ICM_X_expand - ICM_C_m_expand), 2)  # N * M * D -> N * M
            ICM_best_centers = tf.argmin(ICM_Loss, 1)
            self.ICM_best_centers_one_hot = tf.one_hot(ICM_best_centers, self.subcenter_num, dtype=tf.float32)

        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = self.apply_loss_function(self.global_step)
        self.sess.run(tf.global_variables_initializer())
        return

    def load_model(self):
        if self.img_model == 'alexnet':
            img_output = img_alexnet_layers(
                    self.img,
                    self.batch_size,
                    self.output_dim,
                    self.stage,
                    self.model_weights,
                    self.with_tanh,
                    self.val_batch_size)
        else:
            raise Exception('cannot use such CNN model as ' + self.img_model)
        return img_output

    def save_codes(self, database, query, C, model_file=None):
        if model_file is None:
            model_file = self.save_dir + "_codes.npy"

        model = {
            'db_features': database.output,
            'db_reconstr': np.dot(database.codes, C),
            'db_label': database.label,
            'val_features': query.output,
            'val_reconstr': np.dot(query.codes, C),
            'val_label': query.label,
        }
        print("saving codes to %s" % model_file)
        folder = os.path.dirname(model_file)
        if os.path.exists(folder) is False:
            os.makedirs(folder)
        np.save(model_file, np.array(model))
        return

    def save_model(self, model_file=None):
        if model_file is None:
            model_file = self.save_dir

        model = {}
        for layer in self.deep_param_img:
            model[layer] = self.sess.run(self.deep_param_img[layer])

        print("saving model to %s" % model_file)
        folder = os.path.dirname(model_file)
        if os.path.exists(folder) is False:
            os.makedirs(folder)

        np.save(model_file, np.array(model))
        return

    def triplet_loss(self, anchor, pos, neg, margin):
        with tf.variable_scope('triplet_loss'):
            pos_dist = distance(anchor, pos, pair=False, dist_type=self.dist_type)
            neg_dist = distance(anchor, neg, pair=False, dist_type=self.dist_type)
            basic_loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
            loss = tf.reduce_mean(basic_loss, 0)

            tf.summary.histogram('pos_dist', pos_dist)
            tf.summary.histogram('neg_dist', neg_dist)
            tf.summary.histogram('pos_dist - neg_dist', pos_dist - neg_dist)

        return loss

    def quantization_loss(self, z, h):
        with tf.name_scope('quantization_loss'):
            q_loss = tf.reduce_mean(tf.reduce_sum(z - tf.matmul(h, self.C), -1))
        return q_loss

    def apply_loss_function(self, global_step):
        anchor, pos, neg = tf.split(self.img_last_layer, 3, axis=0)
        triplet_loss = self.triplet_loss(anchor, pos, neg, self.triplet_margin)
        cq_loss = self.quantization_loss(self.img_last_layer, self.b_img)
        self.loss = triplet_loss + cq_loss * self.cq_lambda

        self.lr = tf.train.exponential_decay(
                self.learning_rate,
                global_step,
                self.decay_step,
                self.decay_factor,
                staircase=True)
        opt = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
        grads_and_vars = opt.compute_gradients(self.loss, self.train_layers+self.train_last_layer)
        fcgrad, _ = grads_and_vars[-2]
        fbgrad, _ = grads_and_vars[-1]

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('triplet_loss', triplet_loss)
        tf.summary.scalar('cq_loss', cq_loss)
        tf.summary.scalar('lr', self.lr)
        self.merged = tf.summary.merge_all()

        # Last layer has a 10 times learning rate
        if self.finetune_all:
            return opt.apply_gradients([(grads_and_vars[0][0], self.train_layers[0]),
                                        (grads_and_vars[1][0]*2, self.train_layers[1]),
                                        (grads_and_vars[2][0], self.train_layers[2]),
                                        (grads_and_vars[3][0]*2, self.train_layers[3]),
                                        (grads_and_vars[4][0], self.train_layers[4]),
                                        (grads_and_vars[5][0]*2, self.train_layers[5]),
                                        (grads_and_vars[6][0], self.train_layers[6]),
                                        (grads_and_vars[7][0]*2, self.train_layers[7]),
                                        (grads_and_vars[8][0], self.train_layers[8]),
                                        (grads_and_vars[9][0]*2, self.train_layers[9]),
                                        (grads_and_vars[10][0], self.train_layers[10]),
                                        (grads_and_vars[11][0]*2, self.train_layers[11]),
                                        (grads_and_vars[12][0], self.train_layers[12]),
                                        (grads_and_vars[13][0]*2, self.train_layers[13]),
                                        (fcgrad*10, self.train_last_layer[0]),
                                        (fbgrad*20, self.train_last_layer[1])], global_step=global_step)
        else:
            return opt.apply_gradients([(fcgrad*10, self.train_last_layer[0]),
                                        (fbgrad*20, self.train_last_layer[1])], global_step=global_step)

    def initial_centers(self, img_output):
        C_init = np.zeros([self.subspace_num * self.subcenter_num, self.output_dim])
        all_output = img_output
        for i in range(self.subspace_num):
            start = i*int(self.output_dim/self.subspace_num)
            end = (i+1)*int(self.output_dim/self.subspace_num)
            to_fit = all_output[:, start:end]
            kmeans = MiniBatchKMeans(n_clusters=self.subcenter_num).fit(to_fit)
            C_init[i * self.subcenter_num: (i + 1) * self.subcenter_num, start:end] = kmeans.cluster_centers_
        return C_init

    def update_centers(self, img_dataset):
        '''
        Optimize:
            self.C = (U * hu^T + V * hv^T) (hu * hu^T + hv * hv^T)^{-1}
            self.C^T = (hu * hu^T + hv * hv^T)^{-1} (hu * U^T + hv * V^T)
            but all the C need to be replace with C^T :
            self.C = (hu * hu^T + hv * hv^T)^{-1} (hu^T * U + hv^T * V)
        '''
        old_C_value = self.sess.run(self.C)

        h = self.img_b_all
        U = self.img_output_all
        smallResidual = tf.constant(np.eye(self.subcenter_num * self.subspace_num, dtype=np.float32) * 0.001)
        Uh = tf.matmul(tf.transpose(h), U)
        hh = tf.add(tf.matmul(tf.transpose(h), h), smallResidual)
        compute_centers = tf.matmul(tf.matrix_inverse(hh), Uh)

        update_C = self.C.assign(compute_centers)
        C_value = self.sess.run(update_C, feed_dict={
            self.img_output_all: img_dataset.output,
            self.img_b_all: img_dataset.codes,
            })

        C_sums = np.sum(np.square(C_value), axis=1)
        C_zeros_ids = np.where(C_sums < 1e-8)
        C_value[C_zeros_ids, :] = old_C_value[C_zeros_ids, :]
        self.sess.run(self.C.assign(C_value))

    def update_codes_ICM(self, output, code):
        '''
        Optimize:
            min || output - self.C * codes ||
            min || output - codes * self.C ||
        args:
            output: [n_train, n_output]
            self.C: [n_subspace * n_subcenter, n_output]
                [C_1, C_2, ... C_M]
            codes: [n_train, n_subspace * n_subcenter]
        '''

        code = np.zeros(code.shape)

        for iterate in range(self.max_iter_update_b):
            sub_list = list(range(self.subspace_num))
            random.shuffle(sub_list)
            for m in sub_list:
                best_centers_one_hot_val = self.sess.run(self.ICM_best_centers_one_hot, feed_dict={
                    self.ICM_b_m: code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num],
                    self.ICM_b_all: code,
                    self.ICM_m: m,
                    self.ICM_X: output,
                })

                code[:, m * self.subcenter_num: (m + 1) * self.subcenter_num] = best_centers_one_hot_val
        return code

    def update_codes_batch(self, dataset, batch_size):
        '''
        update codes in batch size
        '''
        total_batch = int(ceil(dataset.n_samples / float(batch_size)))
        dataset.finish_epoch()

        for i in range(total_batch):
            output_val, code_val = dataset.next_batch_output_codes(batch_size)
            codes_val = self.update_codes_ICM(output_val, code_val)
            dataset.feed_batch_codes(batch_size, codes_val)

        dataset.finish_epoch()

    def update_codes_and_centers(self, img_dataset):
        for i in range(self.max_iter_update_Cb):
            self.update_codes_batch(img_dataset, self.code_batch_size)
            self.update_centers(img_dataset)

    def update_embedding_and_triplets(self, img_dataset):
        epoch_iter = int(img_dataset.n_samples / self.batch_size)
        for i in range(epoch_iter):
            images, labels, codes = img_dataset.next_batch(self.batch_size)
            output = self.sess.run(self.img_last_layer,
                                   feed_dict={self.img: images, self.b_img: codes})

            img_dataset.feed_batch_output(self.batch_size, output)
        img_dataset.update_triplets(self.triplet_margin, n_part=self.n_part, select_strategy=self.select_strategy)

        n_triplets = img_dataset.triplets.shape[0]
        if n_triplets < self.triplet_thresold and self.n_part > 1:
            print('Halve n_part, num_triplets(%s) < triplet thresold(%d)' % (n_triplets, self.triplet_thresold))
            self.n_part = int(self.n_part / 2)

    def train_cq(self, img_dataset, img_query, img_database, R):
        print("%s #train# start training" % datetime.now())

        # tensorboard
        tflog_path = os.path.join(self.log_dir, self.file_name)
        if os.path.exists(tflog_path):
            shutil.rmtree(tflog_path)
        train_writer = tf.summary.FileWriter(tflog_path, self.sess.graph)

        print("Get initial embedding and select triplets...")
        start_time = time.time()
        self.update_embedding_and_triplets(img_dataset)
        duration = time.time() - start_time
        print('Embedding Done: Time %.3fs' % duration)

        print('Initialize centers and update codes and centers')
        self.sess.run(self.C.assign(self.initial_centers(img_dataset.output)))
        self.update_codes_and_centers(img_dataset)

        train_iter = 0
        for epoch in range(self.max_epoch):
            triplet_batch_size = int(self.batch_size / 3)
            epoch_iter = int(img_dataset.triplets.shape[0] / triplet_batch_size)
            img_dataset.finish_epoch()
            for i in range(epoch_iter):
                start_time = time.time()
                images, labels, codes = img_dataset.next_batch_triplet(triplet_batch_size)
                _, output, loss, summary = self.sess.run(
                    [self.train_op, self.img_last_layer, self.loss, self.merged],
                    feed_dict={self.img: images,
                               self.b_img: codes})
                img_dataset.feed_batch_triplet_output(triplet_batch_size, output)
                if train_iter < 100 or i % 100 == 0:
                    print('%s Epoch: [%d/%d][%d/%d]\tTime %.3fs\tLoss %.3f'
                          % (datetime.now(), epoch, self.max_epoch, i+1, epoch_iter, time.time() - start_time, loss))
                train_writer.add_summary(summary, train_iter)
                train_iter += 1

            # every epoch: update embedding, codes and centers
            self.update_codes_and_centers(img_dataset)

            # update triplets
            self.update_embedding_and_triplets(img_dataset)
            # img_dataset.update_triplets(self.triplet_margin, n_part=self.n_part, select_strategy=self.select_strategy)

            val_summary = tf.Summary()
            val_summary.value.add(tag='num_triplets', simple_value=img_dataset.triplets.shape[0])
            # validation
            if (epoch+1) % self.val_freq == 0 or (epoch+1) == self.max_epoch:
                maps = self.validation(img_query, img_database, R)
                for key in maps:
                    print("{}\t{}".format(key, maps[key]))
                    val_summary.value.add(tag=key, simple_value=maps[key])
            train_writer.add_summary(val_summary, epoch+1)
            train_writer.flush()

        print("%s #traing# finish training" % datetime.now())
        self.save_model()
        print("model saved")

        self.sess.close()

    def val_forward(self, img_dataset, val_print_freq=100):
        batch = int(ceil(img_dataset.n_samples / float(self.val_batch_size)))
        img_dataset.finish_epoch()
        for i in range(batch):
            images, labels, codes = img_dataset.next_batch(self.val_batch_size)
            output = self.sess.run([self.img_last_layer],
                                   feed_dict={self.img: images,
                                              self.stage: 1})
            img_dataset.feed_batch_output(self.val_batch_size, output)
            if i % val_print_freq == 0:
                print("%s #validation# batch %d/%d" % (datetime.now(), i, batch))

    def validation(self, img_query, img_database, R=100):
        print("%s #validation# start validation" % (datetime.now()))

        # Forward to get output
        self.val_forward(img_query)
        self.val_forward(img_database)

        # Initialize centers
        self.sess.run(self.C.assign(self.initial_centers(img_database.output)))

        # Get codes of database && Update centers
        self.update_codes_and_centers(img_database)

        # Get codes of query
        self.update_codes_batch(img_query, self.code_batch_size)

        # Evaluation
        print("%s #validation# calculating MAP@%d" % (datetime.now(), R))
        C_tmp = self.sess.run(self.C)
        mAPs = MAPs_CQ(C_tmp, self.subspace_num, self.subcenter_num, R)
        self.save_codes(img_database, img_query, C_tmp)
        return {
            'map_feature_ip': mAPs.get_mAPs_by_feature(img_database, img_query),
            'map_AQD_ip':  mAPs.get_mAPs_AQD(img_database, img_query),
            'map_SQD_ip': mAPs.get_mAPs_SQD(img_database, img_query)
        }

