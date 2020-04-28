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


class BPR:

    def __init__(self, sess, args, train_df, vali_df
                 , key_genre, item_genre_list, user_genre_count):
        self.dataname = args.dataname

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

        self.lr = args.lr  # learning rate
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step

        self.num_genre = args.num_genre

        self.reg = args.reg  # regularization term trade-off

        print('**********BPR**********')
        print(self.args)
        self._prepare_model()

    def run(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        for epoch_itr in range(1, self.train_epoch + 1):
            self.train_model(epoch_itr)
            if epoch_itr % self.display_step == 0:
                self.test_model(epoch_itr)
        return self.make_records()

    def _prepare_model(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")

        with tf.variable_scope("BPR", reuse=tf.AUTO_REUSE):
            self.P = tf.get_variable(name="P", initializer=tf.truncated_normal(shape=[self.num_rows, self.hidden_neuron],
                                                                          mean=0, stddev=0.03), dtype=tf.float32)
            self.Q = tf.get_variable(name="Q", initializer=tf.truncated_normal(shape=[self.num_cols, self.hidden_neuron],
                                                                          mean=0, stddev=0.03), dtype=tf.float32)

        self.saver = tf.train.Saver([self.P, self.Q])

        p = tf.reduce_sum(tf.nn.embedding_lookup(self.P, self.user_input), 1)
        q_neg = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input_neg), 1)
        q_pos = tf.reduce_sum(tf.nn.embedding_lookup(self.Q, self.item_input_pos), 1)

        predict_pos = (tf.reduce_sum(p * q_pos, 1))
        predict_neg = (tf.reduce_sum(p * q_neg, 1))

        cost1 = tf.reduce_sum(tf.nn.softplus(-(predict_pos - predict_neg)))
        cost2 = self.reg * 0.5 * (self.l2_norm(self.P) + self.l2_norm(self.Q))  # regularization term

        self.cost = cost1 + cost2  # the loss function

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "Adadelta":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "GradientDescent":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif self.optimizer_method == "Momentum":
            optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
        else:
            raise ValueError("Optimizer Key ERROR")

        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            self.optimizer = optimizer.minimize(self.cost)

    def train_model(self, itr):
        NS_start_time = time.time() * 1000.0
        epoch_cost = 0
        num_sample, user_list, item_pos_list, item_neg_list = utility.negative_sample(self.train_df, self.num_rows,
                                                                                      self.num_cols, self.neg)
        NS_end_time = time.time() * 1000.0

        start_time = time.time() * 1000.0
        num_batch = int(len(user_list) / float(self.batch_size)) + 1
        random_idx = np.random.permutation(len(user_list))
        for i in range(num_batch):

            # get the indices of the current batch
            if i == num_batch - 1:
                batch_idx = random_idx[i * self.batch_size:]
            elif i < num_batch - 1:
                batch_idx = random_idx[(i * self.batch_size):((i + 1) * self.batch_size)]
            _, tmp_cost = self.sess.run(  # do the optimization by the minibatch
                [self.optimizer, self.cost],
                feed_dict={self.user_input: user_list[batch_idx, :],
                           self.item_input_pos: item_pos_list[batch_idx, :],
                           self.item_input_neg: item_neg_list[batch_idx, :]})
            epoch_cost += tmp_cost

        if itr % self.display_step == 0:
            print ("Training //", "Epoch %d //" % itr, " Total cost = {:.5f}".format(epoch_cost),
                   "Training time : %d ms" % (time.time() * 1000.0 - start_time),
                   "negative Sampling time : %d ms" % (NS_end_time - NS_start_time),
                   "negative samples : %d" % (num_sample))

        ckpt_save_path = "./"+self.dataname+"/BPR_check_points"
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        self.saver.save(sess, ckpt_save_path + "/check_point.ckpt", global_step=itr)

    def test_model(self, itr):  # calculate the cost and rmse of testing set in each epoch
        if itr % self.display_step == 0:
            start_time = time.time() * 1000.0
            P, Q = self.sess.run([self.P, self.Q])
            Rec = np.matmul(P, Q.T)

            [precision, recall, f_score, NDCG] = utility.test_model_all(Rec, self.vali_df, self.train_df)
            utility.ranking_analysis(Rec, self.vali_df, self.train_df, self.key_genre, self.item_genre_list,
                                     self.user_genre_count)

            print (
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


parser = argparse.ArgumentParser(description='BPR')
parser.add_argument('--train_epoch', type=int, default=20)
parser.add_argument('--display_step', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--reg', type=float, default=0.1)
parser.add_argument('--optimizer_method', choices=['Adam', 'Adadelta', 'Adagrad', 'RMSProp', 'GradientDescent',
                                                   'Momentum'], default='Adam')
parser.add_argument('--hidden_neuron', type=int, default=20)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--neg', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dataname', nargs='?', default='ml1m-6')
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

args.num_genre = num_genre

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
genre_item_indicator = np.zeros((num_genre, num_item))

for k in range(num_genre):
    genre_item_indicator[k, :] = genre_item_vector[key_genre[k]]

precision = np.zeros(4)
recall = np.zeros(4)
f1 = np.zeros(4)
ndcg = np.zeros(4)
RSP = np.zeros(4)
REO = np.zeros(4)

n = args.n
for i in range(n):
    with tf.Session() as sess:
        bpr = BPR(sess, args, train_df, vali_df, key_genre, item_genre_list, user_genre_count)
        [prec_one, rec_one, f_one, ndcg_one, Rec] = bpr.run()
        [RSP_one, REO_one] = utility.ranking_analysis(Rec, vali_df, train_df, key_genre, item_genre_list,
                                                      user_genre_count)
        precision += prec_one
        recall += rec_one
        f1 += f_one
        ndcg += ndcg_one
        RSP += RSP_one
        REO += REO_one

with open('Rec_' + dataname + '_BPR.mat', "wb") as f:
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

