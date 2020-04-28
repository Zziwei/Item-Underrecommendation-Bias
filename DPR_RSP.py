import tensorflow as tf
import time
import numpy as np
import os
import copy
import pickle
import argparse
import utility
import pandas as pd
from sklearn.metrics import *


class DPR_RSP:

    def __init__(self, sess, args, train_df, vali_df, item_genre, genre_error_weight,
                 key_genre, item_genre_list, user_genre_count):
        self.dataname = args.dataname

        self.layers = eval(args.layers)

        self.key_genre = key_genre
        self.item_genre_list = item_genre_list
        self.user_genre_count = user_genre_count

        self.sess = sess
        self.args = args

        self.num_cols = len(train_df['item_id'].unique())
        self.num_rows = len(train_df['user_id'].unique())

        self.hidden_neuron = args.hidden_neuron
        self.neg = args.neg
        self.batch_size = args.batch_size

        self.train_df = train_df
        self.vali_df = vali_df
        self.num_train = len(self.train_df)
        self.num_vali = len(self.vali_df)

        self.train_epoch = args.train_epoch
        self.train_epoch_a = args.train_epoch_a
        self.display_step = args.display_step

        self.lr_r = args.lr_r  # learning rate
        self.lr_a = args.lr_a  # learning rate

        self.reg = args.reg  # regularization term trade-off
        self.reg_s = args.reg_s

        self.num_genre = args.num_genre
        self.alpha = args.alpha
        self.item_genre = item_genre
        self.genre_error_weight = genre_error_weight

        self.genre_count_list = []
        for k in range(self.num_genre):
            self.genre_count_list.append(np.sum(item_genre[:, k]))

        print('**********DPR_RSP**********')
        print(self.args)
        self._prepare_model()

    def loadmodel(self, saver, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def run(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

        saver = tf.train.Saver([self.P, self.Q])
        self.loadmodel(saver, "./"+self.dataname+"/BPR_check_points")

        for epoch_itr in range(1, self.train_epoch + 1 + self.train_epoch_a):
            self.train_model(epoch_itr)
            if epoch_itr % self.display_step == 0:
                self.test_model(epoch_itr)
        return self.make_records()

    def _prepare_model(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

            self.input_item_genre = tf.placeholder(dtype=tf.float32, shape=[None, self.num_genre]
                                                   , name="input_item_genre")
            self.input_item_error_weight = tf.placeholder(dtype=tf.float32, shape=[None, 1]
                                                          , name="input_item_error_weight")

        with tf.variable_scope("BPR", reuse=tf.AUTO_REUSE):
            self.P = tf.get_variable(name="P",
                                     initializer=tf.truncated_normal(shape=[self.num_rows, self.hidden_neuron], mean=0,
                                                                     stddev=0.03), dtype=tf.float32)
            self.Q = tf.get_variable(name="Q",
                                     initializer=tf.truncated_normal(shape=[self.num_cols, self.hidden_neuron], mean=0,
                                                                     stddev=0.03), dtype=tf.float32)
        para_r = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="BPR")

        with tf.variable_scope("Adversarial", reuse=tf.AUTO_REUSE):
            num_layer = len(self.layers)
            adv_W = []
            adv_b = []
            for l in range(num_layer):
                if l == 0:
                    in_shape = 1
                else:
                    in_shape = self.layers[l - 1]
                adv_W.append(tf.get_variable(name="adv_W" + str(l),
                                             initializer=tf.truncated_normal(shape=[in_shape, self.layers[l]],
                                                                             mean=0, stddev=0.03), dtype=tf.float32))
                adv_b.append(tf.get_variable(name="adv_b" + str(l),
                                             initializer=tf.truncated_normal(shape=[1, self.layers[l]],
                                                                             mean=0, stddev=0.03), dtype=tf.float32))
            adv_W_out = tf.get_variable(name="adv_W_out",
                                        initializer=tf.truncated_normal(shape=[self.layers[-1], self.num_genre],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)

            adv_b_out = tf.get_variable(name="adv_b_out",
                                        initializer=tf.truncated_normal(shape=[1, self.num_genre],
                                                                        mean=0, stddev=0.03), dtype=tf.float32)
        para_a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adversarial")

        p = tf.reduce_sum(tf.nn.embedding_lookup(self.P, self.user_input), 1)
        q_neg = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input_neg), 1)
        q_pos = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input_pos), 1)

        predict_pos = tf.reduce_sum(p * q_pos, 1)
        predict_neg = tf.reduce_sum(p * q_neg, 1)

        r_cost1 = tf.reduce_sum(tf.nn.softplus(-(predict_pos - predict_neg)))
        r_cost2 = self.reg * 0.5 * (self.l2_norm(self.P) + self.l2_norm(self.Q))  # regularization term
        pred = tf.matmul(self.P, tf.transpose(self.Q))
        self.s_mean = tf.reduce_mean(pred, axis=1)
        self.s_std = tf.keras.backend.std(pred, axis=1)
        self.s_cost = tf.reduce_sum(tf.square(self.s_mean) + tf.square(self.s_std) - 2 * tf.log(self.s_std) - 1)
        self.r_cost = r_cost1 + r_cost2 + self.reg_s * 0.5 * self.s_cost

        adv_last = tf.reshape(tf.concat([predict_pos, predict_neg], axis=0), [tf.shape(self.input_item_genre)[0], 1])
        for l in range(num_layer):
            adv = tf.nn.relu(tf.matmul(adv_last, adv_W[l]) + adv_b[l])
            adv_last = adv
        self.adv_output = tf.nn.sigmoid(tf.matmul(adv_last, adv_W_out) + adv_b_out)
        self.a_cost = tf.reduce_sum(tf.square(self.adv_output - self.input_item_genre) * self.input_item_error_weight)

        self.all_cost = self.r_cost - self.alpha * self.a_cost  # the loss function

        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            self.r_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_r).minimize(self.r_cost, var_list=para_r)
            self.a_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_a).minimize(self.a_cost, var_list=para_a)
            self.all_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_r).minimize(self.all_cost, var_list=para_r)

    def train_model(self, itr):
        NS_start_time = time.time() * 1000.0
        epoch_r_cost = 0.0
        epoch_s_cost = 0.0
        epoch_s_mean = 0.0
        epoch_s_std = 0.0
        epoch_a_cost = 0.0
        num_sample, user_list, item_pos_list, item_neg_list = utility.negative_sample(self.train_df, self.num_rows,
                                                                                      self.num_cols, self.neg)
        NS_end_time = time.time() * 1000.0

        start_time = time.time() * 1000.0
        num_batch = int(num_sample / float(self.batch_size)) + 1
        random_idx = np.random.permutation(num_sample)
        for i in range(num_batch):
            # get the indices of the current batch
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]

            if itr > self.train_epoch:
                random_idx_a = np.random.permutation(num_sample)
                for j in range(num_batch):
                    if j == num_batch - 1:
                        batch_idx_a = random_idx_a[j * self.batch_size:]
                    elif j < num_batch - 1:
                        batch_idx_a = random_idx_a[(j * self.batch_size):((j + 1) * self.batch_size)]
                    item_idx_list = ((item_pos_list[batch_idx_a, :]).reshape((len(batch_idx_a)))).tolist() \
                                    + ((item_neg_list[batch_idx_a, :]).reshape((len(batch_idx_a)))).tolist()
                    _, tmp_a_cost = self.sess.run(  # do the optimization by the minibatch
                        [self.a_optimizer, self.a_cost],
                        feed_dict={self.user_input: user_list[batch_idx_a, :],
                                   self.item_input_pos: item_pos_list[batch_idx_a, :],
                                   self.item_input_neg: item_neg_list[batch_idx_a, :],
                                   self.input_item_genre: self.item_genre[item_idx_list, :],
                                   self.input_item_error_weight: self.genre_error_weight[item_idx_list, :]})
                    epoch_a_cost += tmp_a_cost

                item_idx_list = ((item_pos_list[batch_idx, :]).reshape((len(batch_idx)))).tolist() \
                                + ((item_neg_list[batch_idx, :]).reshape((len(batch_idx)))).tolist()
                _, tmp_r_cost, tmp_s_cost, tmp_s_mean, tmp_s_std = self.sess.run(  # do the optimization by the minibatch
                    [self.all_optimizer, self.all_cost, self.s_cost, self.s_mean, self.s_std],
                    feed_dict={self.user_input: user_list[batch_idx, :],
                               self.item_input_pos: item_pos_list[batch_idx, :],
                               self.item_input_neg: item_neg_list[batch_idx, :],
                               self.input_item_genre: self.item_genre[item_idx_list, :],
                               self.input_item_error_weight: self.genre_error_weight[item_idx_list, :]})
                epoch_r_cost += tmp_r_cost
                epoch_s_mean += np.mean(tmp_s_mean)
                epoch_s_std += np.mean(tmp_s_std)
                epoch_s_cost += tmp_s_cost
            else:
                item_idx_list = ((item_pos_list[batch_idx, :]).reshape((len(batch_idx)))).tolist() \
                                + ((item_neg_list[batch_idx, :]).reshape((len(batch_idx)))).tolist()
                _, tmp_r_cost, tmp_s_cost, tmp_s_mean, tmp_s_std = self.sess.run(  # do the optimization by the minibatch
                    [self.r_optimizer, self.r_cost, self.s_cost, self.s_mean, self.s_std],
                    feed_dict={self.user_input: user_list[batch_idx, :],
                               self.item_input_pos: item_pos_list[batch_idx, :],
                               self.item_input_neg: item_neg_list[batch_idx, :],
                               self.input_item_genre: self.item_genre[item_idx_list, :],
                               self.input_item_error_weight: self.genre_error_weight[item_idx_list, :]})
                epoch_r_cost += tmp_r_cost
                epoch_s_mean += np.mean(tmp_s_mean)
                epoch_s_std += np.mean(tmp_s_std)
                epoch_s_cost += tmp_s_cost
        epoch_a_cost /= num_batch
        if itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % itr, " Total r_cost = %.5f" % epoch_r_cost,
                   " Total s_cost = %.5f" % epoch_s_cost,
                   " Total s_mean = %.5f" % epoch_s_mean,
                   " Total s_std = %.5f" % epoch_s_std,
                   " Total a_cost = %.5f" % epoch_a_cost,
                   "Training time : %d ms" % (time.time() * 1000.0 - start_time),
                   "negative Sampling time : %d ms" % (NS_end_time - NS_start_time),
                   "negative samples : %d" % (num_sample))

    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        if itr % self.display_step == 0:
            start_time = time.time() * 1000.0
            P, Q = self.sess.run([self.P, self.Q])
            Rec = np.matmul(P, Q.T)

            utility.test_model_all(Rec, self.vali_df, self.train_df)
            utility.ranking_analysis(Rec, self.vali_df, self.train_df, self.key_genre, self.item_genre_list,
                                     self.user_genre_count)
            print(
                "Testing //", "Epoch %d //" % itr,
                "Testing time : %d ms" % (time.time() * 1000.0 - start_time))
            print("=" * 200)

    def make_records(self):  # record all the results' details into files
        P, Q = self.sess.run([self.P, self.Q])
        Rec = np.matmul(P, Q.T)
        [precision, recall, f_score, NDCG] = utility.test_model_all(Rec, self.vali_df, self.train_df)
        return precision, recall, f_score, NDCG, Rec

    @staticmethod
    def l2_norm(tensor):
        return tf.reduce_sum(tf.square(tensor))


parser = argparse.ArgumentParser(description='DPR_RSP')
parser.add_argument('--train_epoch', type=int, default=0)
parser.add_argument('--train_epoch_a', type=int, default=20)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--lr_r', type=float, default=0.01)
parser.add_argument('--lr_a', type=float, default=0.005)
parser.add_argument('--reg', type=float, default=0.1)
parser.add_argument('--reg_s', type=float, default=30)
parser.add_argument('--hidden_neuron', type=int, default=20)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--neg', type=int, default=5)
parser.add_argument('--alpha', type=float, default=5000.0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--layers', nargs='?', default='[50, 50, 50, 50]')
parser.add_argument('--dataname', nargs='?', default='ml1m-4')
args = parser.parse_args()

dataname = args.dataname


train_df = pickle.load(open('./' + dataname + '/training_df.pkl'))
vali_df = pickle.load(open('./' + dataname + '/valiing_df.pkl'))  # for validation
# vali_df = pickle.load(open('./' + dataname + '/testing_df.pkl'))  # for testing
key_genre = pickle.load(open('./' + dataname + '/key_genre.pkl'))
item_idd_genre_list = pickle.load(open('./' + dataname + '/item_idd_genre_list.pkl'))
genre_item_vector = pickle.load(open('./' + dataname + '/genre_item_vector.pkl'))
genre_count = pickle.load(open('./' + dataname + '/genre_count.pkl'))
user_genre_count = pickle.load(open('./' + dataname + '/user_genre_count.pkl'))

num_item = len(train_df['item_id'].unique())
num_user = len(train_df['user_id'].unique())
num_genre = len(key_genre)

item_genre_list = []
for u in range(num_item):
    gl = item_idd_genre_list[u]
    tmp = []
    for g in gl:
        if g in key_genre:
            tmp.append(g)
    item_genre_list.append(tmp)

print('!' * 100)
print('number of positive feedback: ' + str(len(train_df)))
print('estimated number of training samples: ' + str(args.neg * len(train_df)))
print('!' * 100)

# genreate item_genre matrix
item_genre = np.zeros((num_item, num_genre))
for i in range(num_item):
    gl = item_genre_list[i]
    for k in range(num_genre):
        if key_genre[k] in gl:
            item_genre[i, k] = 1.0

genre_count_mean_reciprocal = []
for k in key_genre:
    genre_count_mean_reciprocal.append(1.0 / genre_count[k])
genre_count_mean_reciprocal = (np.array(genre_count_mean_reciprocal)).reshape((num_genre, 1))
genre_error_weight = np.dot(item_genre, genre_count_mean_reciprocal)

args.num_genre = num_genre

precision = np.zeros(4)
recall = np.zeros(4)
f1 = np.zeros(4)
ndcg = np.zeros(4)
RSP = np.zeros(4)
REO = np.zeros(4)

n = args.n
for i in range(n):
    with tf.Session() as sess:
        dpr = DPR_RSP(sess, args, train_df, vali_df, item_genre, genre_error_weight,
                      key_genre, item_genre_list, user_genre_count)
        [prec_one, rec_one, f_one, ndcg_one, Rec] = dpr.run()
        [RSP_one, REO_one] = utility.ranking_analysis(Rec, vali_df, train_df, key_genre,
                                                      item_genre_list, user_genre_count)
        precision += prec_one
        recall += rec_one
        f1 += f_one
        ndcg += ndcg_one
        RSP += RSP_one
        REO += REO_one

with open('Rec_' + dataname + '_DPR_RSP.mat', "wb") as f:
    np.save(f, Rec)


precision /= n
recall /= n
f1 /= n
ndcg /= n
RSP /= n
REO /= n

print('')
print('*' * 100)
print('Averaged precision@1\t%.7f,\t||\tprecision@5\t%.7f,\t||\tprecision@10\t%.7f,\t||\tprecision@15\t%.7f' \
      % (precision[0], precision[1], precision[2], precision[3]))
print('Averaged recall@1\t%.7f,\t||\trecall@5\t%.7f,\t||\trecall@10\t%.7f,\t||\trecall@15\t%.7f' \
      % (recall[0], recall[1], recall[2], recall[3]))
print('Averaged f1@1\t\t%.7f,\t||\tf1@5\t\t%.7f,\t||\tf1@10\t\t%.7f,\t||\tf1@15\t\t%.7f' \
      % (f1[0], f1[1], f1[2], f1[3]))
print('Averaged NDCG@1\t\t%.7f,\t||\tNDCG@5\t\t%.7f,\t||\tNDCG@10\t\t%.7f,\t||\tNDCG@15\t\t%.7f' \
      % (ndcg[0], ndcg[1], ndcg[2], ndcg[3]))
print('*' * 100)
print('Averaged RSP    @1\t%.7f\t||\t@5\t%.7f\t||\t@10\t%.7f\t||\t@15\t%.7f' \
      % (RSP[0], RSP[1], RSP[2], RSP[3]))
print('Averaged REO @1\t%.7f\t||\t@5\t%.7f\t||\t@10\t%.7f\t||\t@15\t%.7f' \
      % (REO[0], REO[1], REO[2], REO[3]))
print('*' * 100)

