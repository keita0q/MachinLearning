# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import random


num_of_input_nodes          = 1
num_of_hidden_nodes         = 80
num_of_output_nodes         = 1
length_of_sequences         = 10
num_of_training_epochs      = 5000
size_of_mini_batch          = 100
num_of_prediction_epochs    = 1000
learning_rate               = 0.01
forget_bias                 = 0.8
num_of_sample               = 1000

# batchデータ生成
def get_batch(batch_size, X, t):
    rnum = [random.randint(0,len(X)-1) for x in range(batch_size)]
    xs = np.array([[[y] for y in list(X[r])] for r in rnum])
    ts = np.array([[t[r]] for r in rnum])
    return xs,ts

def create_data(nb_of_samples, sequence_len):
    X = np.zeros((nb_of_samples, sequence_len))
    for row_idx in range(nb_of_samples):
        X[row_idx,:] = np.around(np.random.rand(sequence_len)).astype(int)
    # Create the targets for each sequence
    t = np.sum(X, axis=1)
    return X, t

def make_prediction(nb_of_samples):
    sequence_len = length_of_sequences
    xs, ts = create_data(nb_of_samples, sequence_len)
    return np.array([[[y] for y in x] for x in xs]), np.array([[x] for x in ts])


def inference(input_ph):
     with tf.name_scope("inference") as scope:
        weight_in = tf.Variable(tf.truncated_normal([num_of_input_nodes, num_of_hidden_nodes], stddev=0.1), name="weight_in")
        weight_out = tf.Variable(tf.truncated_normal([num_of_hidden_nodes, num_of_output_nodes], stddev=0.1), name="weight_out")
        bias_in   = tf.Variable(tf.truncated_normal([num_of_hidden_nodes], stddev=0.1), name="bias_in")
        bias_out   = tf.Variable(tf.truncated_normal([num_of_output_nodes], stddev=0.1), name="bias_out")

        # 10*100*1
        in1 = tf.transpose(input_ph, [1, 0, 2])
        u_in = tf.reshape(in1, [-1, num_of_input_nodes])
        z_in = tf.matmul(u_in, weight_in) + bias_in
        # length_of_sequences 個に入力を分割する
        z_in_seq = tf.split(0, length_of_sequences, z_in)

        # make Basic RNN cells
        # cell = rnn_cell.BasicRNNCell(num_of_hidden_nodes)
        # make LSTN cells
        cell = rnn_cell.BasicLSTMCell(num_of_hidden_nodes, forget_bias=forget_bias)

        # modeling rnn Layer
        # rnn_outputs -> 全状態繊維家庭における出力結果
        # states_op -> 最終出力の一つ前の状態(出力)
        rnn_outputs, states_op = rnn.rnn(cell, z_in_seq, dtype=tf.float32)

        # modelの最終出力
        # rnn_outputs[-1] -> 最終状態での出力
        output_op = tf.matmul(rnn_outputs[-1], weight_out) + bias_out

        # Add summary ops to collect data
        w1_hist = tf.histogram_summary("weights_in", weight_in)
        w2_hist = tf.histogram_summary("weights_out", weight_out)
        b1_hist = tf.histogram_summary("biases_in", bias_in)
        b2_hist = tf.histogram_summary("biases_out", bias_out)
        output_hist = tf.histogram_summary("output",  output_op)
        # save 用
        results = [weight_in, weight_out, bias_in,  bias_out]
        return output_op, states_op, results

def loss(output_op, supervisor_ph):
    with tf.name_scope("loss") as scope:
        error = tf.reduce_mean(tf.square(output_op - supervisor_ph))
        tf.scalar_summary("loss", error)
        return error

def training(loss_op, optimizer):
    with tf.name_scope("training") as scope:
        training_op = optimizer.minimize(loss_op)
        return training_op

def calc_accuracy(output_op, prints=False):
        inputs, ts = make_prediction(num_of_prediction_epochs)
        pred_dict = {
                input_ph:  inputs,
                supervisor_ph: ts,
        }
        output= sess.run([output_op], feed_dict=pred_dict)

        def print_result (i, p, q):
            print ([list(x)[0] for x in i])
            print("output: %f, correct: %d" % (p , q))
        if prints:
            [print_result(i, p, q)  for i, p, q in zip(inputs, output[0], ts)]

        opt = abs(output - ts)[0]
        total = sum([1 if x[0] < 0.05 else 0 for x in opt])
        print("accuracy %f" % (total/float(len(ts))))
        return output

# -------------------------------------------------------------------------------------------------------
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

X, t = create_data(num_of_sample, length_of_sequences)

with tf.Graph().as_default():
    # 入力データ流し込み口
    input_ph      = tf.placeholder(tf.float32, [None, length_of_sequences, num_of_input_nodes], name="input")
    # ラベルデータ流し込み口
    supervisor_ph = tf.placeholder(tf.float32, [None, num_of_output_nodes], name="supervisor")

    # 毎回 placeholder に状態は初期値を流し込む。
    output_op, states_op, datas_op = inference(input_ph)
    loss_op = loss(output_op, supervisor_ph)
    training_op = training(loss_op, optimizer)

    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        saver = tf.train.Saver()
        summary_writer = tf.train.SummaryWriter("./tmp/LSTM_log", graph=sess.graph)
        sess.run(init)

        for epoch in range(num_of_training_epochs):
            inputs, label = get_batch(size_of_mini_batch, X, t)
            train_dict = {
                input_ph:      inputs,
                supervisor_ph: label,
            }
            _, summary_str = sess.run([training_op,summary_op], feed_dict=train_dict)
            summary_writer.add_summary(summary_str, epoch)

            if (epoch ) % 100 == 0:
                train_loss = sess.run(loss_op, feed_dict=train_dict)
                print("train#%d, train loss: %e" % (epoch, train_loss))
                if (epoch ) % 500 == 0:
                    calc_accuracy(output_op)

        calc_accuracy(output_op, prints=True)
        datas = sess.run(datas_op)
        saver.save(sess, "model.ckpt")
