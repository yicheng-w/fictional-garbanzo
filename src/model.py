import random
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core

from tf_utils.utils import get_keep_prob, sparse_cross_entropy_with_probs
from tf_utils.ops import create_multi_rnn, bidaf_attention,\
    tri_linear_attention, bi_rnn_encoder, concat_with_product,\
    self_attention_encoder, bi_cudnn_rnn_encoder

from tf_utils.pointer_wrapper import AttnPointerWrapper

from data import encode_data, generate_oov_vocab

from prototypes import BaseModel

from multiprocessing import Queue

class BiRNNClf(BaseModel):
    '''
    multi hop bidaf + residual self attn + ELMo
    '''
    def __init__(self, opts, vocab, classes): # opts = tf.app.flags.FLAGS object
        self.opt = opts
        self.vocab = vocab
        self.classes = classes
        batch_size = self.opt.batch_size
        dropout_rate = self.opt.dropout_rate

        glove_embeddings = tf.constant(self.vocab.embeddings(),
                dtype=tf.float32)

        self.emb = tf.get_variable("embeddings", initializer=glove_embeddings)

        # create placeholders
        self.encoder_inputs = tf.placeholder(shape=[batch_size, None],
                dtype=tf.int32, name='encoder_inputs')
        self.encoder_input_lengths = tf.placeholder(shape=[batch_size],
                dtype=tf.int32, name='encoder_input_lengths')

        self.is_training = tf.placeholder(shape=[], dtype=tf.bool,
                name='is_training')

        self.target_labels = tf.placeholder(shape=[batch_size], dtype=tf.int32,
                name='target_labels')

        self.keep_prob = get_keep_prob(dropout_rate, self.is_training)

        self.graph_built = False
        self.train_op_added = False


    def build_graph(self):

        with tf.device("/cpu:0"):
            with tf.variable_scope("embedding"):
                embedded_input_seq = tf.nn.embedding_lookup(self.emb,
                        self.encoder_inputs)

        enc_hidden_sz = self.opt.hidden_size_encoder
        enc_num_layers = self.opt.num_layers_encoder

        with tf.variable_scope("encoder"):
            _, state = bi_rnn_encoder("lstm", enc_hidden_sz, enc_num_layers,
                    self.opt.dropout_rate, embedded_input_seq,
                    self.encoder_input_lengths)
            state = tf.concat([state[0].c, state[0].h, state[1].c, state[1].h], axis=-1)

        print(state.get_shape().as_list())

        with tf.variable_scope("MLP"):
            hidden = tf.layers.dense(
                    inputs=state,
                    units=self.opt.hidden_size,
                    activation=tf.nn.relu,
                    name='hidden_projection')

            logits = tf.layers.dense(
                    inputs=hidden,
                    units=self.classes,
                    activation=None,
                    name='output_projection')
        
        self.preds = tf.argmax(logits, axis=1)

        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.target_labels))
        self.eval_loss = self.loss
        
        self.graph_built = True

    def add_train_op(self):
        opt = tf.train.AdamOptimizer(self.opt.learning_rate)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.loss, tvars)
        clipped_grads, self.norm = tf.clip_by_global_norm(
                grads, self.opt.clipping_threshold)

        self._train_op = opt.apply_gradients(
                zip(clipped_grads, tvars))

        self.saver = tf.train.Saver(
                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
                max_to_keep=None)

        self.train_op_added = True

    def encode(self, batch, is_train):
        opt = self.opt

        inputs = map(lambda item: item[0], batch)
        labels = map(lambda item: item[1], batch)

        max_input_len = min(opt.max_iterations, max(map(len, inputs)))

        #print(max_input_len)

        encoded_inputs, encoded_inputs_len = encode_data(
                inputs, max_input_len, self.vocab)


        feed_dict = {
                self.encoder_inputs : encoded_inputs,
                self.encoder_input_lengths : encoded_inputs_len,
                self.is_training: is_train
                }
        
        if is_train:
            feed_dict[self.target_labels] = labels

        return feed_dict

    def train_step(self, sess, fd):
        if not self.graph_built:
            raise ValueError("Graph is not built yet!")
        if not self.train_op_added:
            raise ValueError("Train ops have not been added yet!")

        _, loss, grad_norm = sess.run([self._train_op, self.loss, self.norm], feed_dict=fd)
        return loss, grad_norm

    def eval(self, sess, fd):
        if not self.graph_built:
            raise ValueError("Graph is not built yet!")
        if not self.train_op_added:
            raise ValueError("Train ops have not been added yet!")

        loss, preds = sess.run([self.eval_loss, self.preds], feed_dict=fd)
        return loss, preds

    def save_to(self, sess, path):
        if not self.graph_built:
            raise ValueError("Graph is not built yet!")
        if not self.train_op_added:
            raise ValueError("Train ops have not been added yet!")

        self.saver.save(sess, path)

    def restore_from(self, sess, path):
        if not self.graph_built:
            raise ValueError("Graph is not built yet!")
        if not self.train_op_added:
            raise ValueError("Train ops have not been added yet!")

        self.saver.restore(sess, path)
