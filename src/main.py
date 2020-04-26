import os
import tensorflow as tf
import numpy as np
from math import exp
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from summary_handler import SummaryHandler
from read_data import *
from data import GenModelVocab, translate, save_vocab, restore_vocab,\
    translate_spans, GloVEVocab
import time

from tensorflow.python.client import timeline

from utils import *

import random

from model import BiRNNClf as UsedModel

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def main(config):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # disable cpp error msgs
    if config.mode == 'train':
        _train(config)
    elif config.mode == 'test':
        _test(config)
    else:
        raise ValueError("Invalid Mode!")

def _train(config):
    if config.dataset == "imdb":
        train_data, valid_data, test_data = create_imdb_data(config)
    else:
        train_data, valid_data, test_data = create_twenty_newsgroup_data(config)

    vocab_freq = train_data.get_word_lists()

    print("Data loaded!")
    vocab = GloVEVocab(vocab_freq, config.embedding_file,
            threshold=config.min_occurence)

    print("Vocab built! Size (%d)" % vocab.size())

    model = UsedModel(config, vocab, 2 if config.dataset == 'imdb' else 20)

    #create session 
    gpu_configuration = gpu_config()
    sess = tf.Session(config=gpu_configuration)
    with sess.as_default():
        model.build_graph()
        print("Graph built!")
        model.add_train_op()
        print("Train op added!")
    sess.run(tf.global_variables_initializer())
    print("Variables initialized")

    if config.continue_training:
        start_e, steps, out_dir, ckpt_dir = restore_from_last_ckpt(
                config, model, sess)

        # backup new argv
        with open(os.path.join(out_dir, 'argv.txt'), 'a') as f:
            f.write("\n")
            f.write(" ".join(sys.argv))

        print("Continue training after epoch %d, step %d" % (start_e, steps))

    else:
        if config.model_name == 'default':
            c_time = time.strftime("%m_%d_%H_%M_%S", time.localtime())
            config.model_name = UsedModel.__name__ + "_%s" % c_time

        if config.debug:
            config.checkpoint_size = 10

        if not config.debug:
            out_dir = os.path.join(config.out_root, config.model_name)
            if os.path.exists(out_dir):
                raise ValueError("Output directory already exists!")
            else:
                os.makedirs(out_dir)

            # back up src file
            os.system("cp -r src %s" % os.path.join(out_dir, 'src'))
            # back up argv
            with open(os.path.join(out_dir, "argv.txt"), 'w') as f:
                f.write(" ".join(sys.argv))

            # back up environ
            with open(os.path.join(out_dir, 'recreate_environ.sh'), 'w') as f:
                for var, val in os.environ.items():
                    f.write("export %s=\"%s\"\n" % (var, val))

            os.system("chmod +x %s" % os.path.join(out_dir, 'recreate_environ.sh'))

            ckpt_dir = os.path.join(out_dir, "ckpts")

            vocab_loc = os.path.join(out_dir, "vocab.pkl")
            save_vocab(vocab, vocab_loc)

            print("Initialized output at %s" % out_dir)

        steps = 0
        start_e = -1

        print("Started training!")

    #construct graph handler
    summary_handler = SummaryHandler(
        os.path.join(config.summary_save_path, config.model_name),
        ['LOSS', 'ACCURACY'])

    for e in range(config.num_epochs):
        total_loss = []
        grad_norms = []
        for batches in tqdm(train_data.get_batches(config.batch_size)):

            if steps != 0 or not config.start_eval:
                steps += 1

            if steps > 10 and config.debug:
                exit(0)

            is_training = True

            fd = model.encode(batches, is_training)
            loss, grad_norm = model.train_step(sess, fd)
            total_loss.append(loss)
            grad_norms.append(grad_norm)

            if steps % config.checkpoint_size == 0:
                accuracy = eval_model(
                        config, valid_data, vocab, model, sess)
                print("Result at step %d: %f" % (steps, accuracy))
                print("avg lost: %f" % (sum(total_loss) / len(total_loss)))
                print("avg grad norm: %f"%(sum(grad_norms) / len(grad_norms)))

                if not config.debug:
                    summary_handler.write_summaries(sess,
                            {
                                'ITERATION': steps,
                                'LOSS': avg(total_loss),
                                'ACCURACY': accuracy
                            })

                    if start_e > 0:
                        epoch = e + start_e
                    else:
                        epoch = e

                    model.save_to(sess, os.path.join(ckpt_dir,
                        'epoch_%04d_step%08d_acc(%f)' % (epoch, steps,
                            accuracy)))

    summary_handler.close_writer()   

def _test(config):
    print("Evaluating!")

    out_dir = os.path.join(config.out_root, config.model_name)

    vocab_loc = os.path.join(out_dir, "vocab.pkl")
    if os.path.exists(vocab_loc): # vocab exists!
        vocab = restore_vocab(vocab_loc)
    else:
        raise Exception("Not valid output directory! No vocab found!")

    print("Vocab built! Size (%d)" % vocab.size())

    if config.dataset == "imdb":
        train_data, valid_data, test_data = create_imdb_data(config)
    else:
        train_data, valid_data, test_data = create_twenty_newsgroup_data(config)

    if not config.use_dev:
        valid_data = test_data

    print("Data loaded!")

    #construct model
    model = UsedModel(config, vocab, 2 if config.dataset == 'imdb' else 20)

    gpu_configuration = gpu_config()
    sess = tf.Session(config=gpu_configuration)
    with sess.as_default():
        model.build_graph()
        print("Graph built!")
        model.add_train_op()
        print("Train op added!")

    sess.run(tf.global_variables_initializer())

    if config.use_ckpt is not None:
        model.restore_from(sess, os.path.join(out_dir, 'ckpts', config.use_ckpt))
    elif config.at_step is not None:
        restore_from_step(config, model, sess, config.at_step)
    else:
        raise ValueError("Must specify a ckpt to restore from!")

    accuracy = eval_model(
            config, valid_data, vocab, model, sess)
    print("Results: %f" % (accuracy))

    print("Done!")

