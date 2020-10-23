import numpy as np
import tensorflow as tf
import os
import pandas as pd
import scipy.sparse
import time
import sys
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run ENMF")
    parser.add_argument('--dataset', nargs='?', default='ml-srns',
                        help='Choose a dataset: ml-srns')
    parser.add_argument('--verbose', type=int, default=10,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='dropout keep_prob')
    parser.add_argument('--negative_weight', type=float, default=0.3,   #important 
                        help='weight of non-observed data')
    parser.add_argument('--topK', nargs='?', type=int, default=3,
                        help='topK for hr/ndcg')

    return parser.parse_args()



def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep='\t')
    return tp


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


class ENMF:
    def __init__(self, user_num, item_num, embedding_size, max_item_pu, pre_item_num,args):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = embedding_size
        self.max_item_pu = max_item_pu
        self.weight1 = args.negative_weight
        self.lambda_bilinear = [0, 0]

    def _create_placeholders(self):
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_ur = tf.placeholder(tf.int32, [None, self.max_item_pu], name="input_ur")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #for pre
        self.input_pre = tf.placeholder(tf.int32, [None, pre_item_num], name="input_ur")

    def _create_variables(self):
        self.uidW = tf.Variable(tf.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidWg")
        self.iidW = tf.Variable(tf.truncated_normal(shape=[self.item_num + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="iidW")

        # item domain
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hi")

    def _create_inference(self):
        self.uid = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.uid = tf.reshape(self.uid, [-1, self.embedding_size])

        self.uid = tf.nn.dropout(self.uid, self.dropout_keep_prob)

        self.pos_item = tf.nn.embedding_lookup(self.iidW, self.input_ur)
        self.pos_num_r = tf.cast(tf.not_equal(self.input_ur, self.item_num), 'float32')
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
        self.pos_r = tf.einsum('ac,abc->abc', self.uid, self.pos_item)
        self.pos_r = tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)
        self.pos_r = tf.reshape(self.pos_r, [-1, max_item_pu])

    def _pre(self):
        pre_item = tf.nn.embedding_lookup(self.iidW, self.input_pre)
        dot = tf.einsum('ac,abc->abc', self.uid, pre_item)
        pre = tf.einsum('ajk,kl->ajl', dot, self.H_i)
        pre = tf.reshape(pre, [-1, pre_item_num])
        return pre

    def _create_loss(self):
        self.loss1 = self.weight1 * tf.reduce_sum(
            tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc', self.iidW, self.iidW), 0)
                          * tf.reduce_sum(tf.einsum('ab,ac->abc', self.uid, self.uid), 0)
                          * tf.matmul(self.H_i, self.H_i, transpose_b=True), 0), 0)
        self.loss1 += tf.reduce_sum((1.0 - self.weight1) * tf.square(self.pos_r) - 2.0 * self.pos_r)
        self.l2_loss0 = tf.nn.l2_loss(self.uidW)
        self.l2_loss1 = tf.nn.l2_loss(self.iidW)
        self.loss = self.loss1 \
                    + self.lambda_bilinear[0] * self.l2_loss0 \
                    + self.lambda_bilinear[1] * self.l2_loss1

        self.reg_loss = self.lambda_bilinear[0] * self.l2_loss0 \
                        + self.lambda_bilinear[1] * self.l2_loss1

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self.pre = self._pre()


def train_step1(u_batch, y_batch,args):
    """
    A single training step
    """

    feed_dict = {
        deep.input_u: u_batch,
        deep.input_ur: y_batch,
        deep.dropout_keep_prob: args.dropout,
    }
    _, loss, loss1, loss2 = sess.run(
        [train_op1, deep.loss, deep.loss1, deep.reg_loss],
        feed_dict)
    return loss, loss1, loss2

def dev_leave_one_out(u_test,item_test,args):



    user_test = u_test[:, np.newaxis]

    ll = int(len(u_test) / 128) + 1

    hit=0.0
    ndcg=0.0

    for batch_num in range(ll):

        start_index = batch_num * 128
        end_index = min((batch_num + 1) * 128, len(u_test))
        u_batch = user_test[start_index:end_index]
        i_batch = item_test[start_index:end_index]
        batch_users = end_index - start_index

        feed_dict = {
            deep.input_u: u_batch,
            deep.input_pre:i_batch,
            deep.dropout_keep_prob: 1.0,
        }

        pre = sess.run(
            deep.pre, feed_dict)

        pre = np.array(pre)

        recall = []

        sort_idx=np.argsort(-pre, axis=1)
        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1

        hit_batch=(gt_rank <= args.topK)

        ndcg_batch = hit_batch / np.log2(gt_rank + 1)

        hit +=np.sum(hit_batch)
        ndcg+=np.sum(ndcg_batch)

    hit=hit/len(u_test)
    ndcg=ndcg/len(u_test)

    print hit, ndcg 

def get_train_instances1(train_set):
    user_train, item_train = [], []
    for i in train_set.keys():
        user_train.append(i)
        item_train.append(train_set[i])

    user_train = np.array(user_train)
    item_train = np.array(item_train)
    user_train = user_train[:, np.newaxis]
    return user_train, item_train


if __name__ == '__main__':
    np.random.seed(2019)
    random_seed = 2019
    args = parse_args()

    if args.dataset == 'ml-srns':
        print('load ml-srns data')
        DATA_ROOT = '../data/ml-srns'
        f1 = open(os.path.join(DATA_ROOT, 'ENMF_user.txt'), 'w')
    

    tp_test = load_data(os.path.join(DATA_ROOT, 'test.csv'))
    tp_train = load_data(os.path.join(DATA_ROOT, 'train.csv'))

    tp_all = tp_train.append(tp_test)

    tp_dul = pd.merge(tp_train, tp_test)

    usercount, itemcount = get_count(tp_all, 'user_id'), get_count(tp_all, 'item_id')

    n_users, n_items = usercount.shape[0]+1, itemcount.shape[0]+2

    print n_users, n_items

    batch_size = args.batch_size
    lr=args.lr
    embedding_size=args.embed_size
    epochs=args.epochs

    u_train = np.array(tp_train['user_id'], dtype=np.int32)
    i_train = np.array(tp_train['item_id'], dtype=np.int32)
    u_test = np.array(tp_test['user_id'], dtype=np.int32)
    i_test = np.array(tp_test['item_id'], dtype=np.int32)

    i_test_neg = tp_test['neg_items']

    item_test= []
    
    for i in range(len(u_test)):
        temp=[i_test[i]]
        for j in json.loads(i_test_neg[i]):
            temp.append(j)
        item_test.append(temp)

    pre_item_num = len(item_test[0])

    train_set = {}
    max_item_pu = 0
    for i in range(len(u_train)):
        if u_train[i] in train_set:
            train_set[u_train[i]].append(i_train[i])
        else:
            train_set[u_train[i]] = [i_train[i]]
    for i in train_set:
        if len(train_set[i]) > max_item_pu:
            max_item_pu = len(train_set[i])
    print max_item_pu
    for i in train_set:
        while len(train_set[i]) < max_item_pu:
            train_set[i].append(n_items)

    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            deep = ENMF(n_users, n_items, embedding_size, max_item_pu, pre_item_num, args)
            deep._build_graph()
            optimizer1 = tf.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=1e-8).minimize(
                deep.loss)
            train_op1 = optimizer1

            sess.run(tf.global_variables_initializer())

            user_train1, item_train1 = get_train_instances1(train_set)

            for epoch in range(epochs):
                print epoch
                start_t = _writeline_and_time('\tUpdating...')

                shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
                user_train1 = user_train1[shuffle_indices]
                item_train1 = item_train1[shuffle_indices]

                ll = int(len(user_train1) / batch_size)
                loss = [0.0, 0.0, 0.0]

                for batch_num in range(ll):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, len(user_train1))

                    u_batch = user_train1[start_index:end_index]
                    i_batch = item_train1[start_index:end_index]

                    loss1, loss2, loss3 = train_step1(u_batch, i_batch,args)
                    loss[0] += loss1
                    loss[1] += loss2
                    loss[2] += loss3
                print('\r\tUpdating: time=%.2f'
                      % (time.time() - start_t))
                print 'loss,loss_no_reg,loss_reg ', loss[0] / ll, loss[1] / ll, loss[2] / ll

                if epoch < epochs:
                    if epoch % args.verbose == 0:
                        dev_leave_one_out(u_test,item_test,args)

                if epoch >= epochs:
                    dev_leave_one_out(u_test,item_test,args)
























