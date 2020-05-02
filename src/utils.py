import os
import tensorflow as tf
import numpy as np
from math import exp
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from summary_handler import SummaryHandler
from data import GenModelVocab, translate, translate_spans
import time
import copy
import json

import sys

import random

import json

def eval_model(config, valid_data, vocab, model, sess):
    model_preds, eval_losses = generate_predictions(config, valid_data, vocab,
            model, sess)

    accuracy = []

    for i, p in enumerate(model_preds):
        accuracy.append(1 if valid_data.data[i][1] == p else 0)

    accuracy = float(sum(accuracy)) / len(accuracy)

    return accuracy

def generate_predictions(config, valid_data, vocab, model, sess):
    model_preds = []
    eval_losses = []

    dev_batch_obj = tqdm(enumerate(valid_data.get_batches(
        config.batch_size, shuffle=False, pad_to_full_batch=True)))

    for i, dev_batch in dev_batch_obj:
        is_training = True
        fd = model.encode(dev_batch, is_training)
        eval_loss, preds = model.eval(sess, fd)
        eval_losses.append(eval_loss)
        model_preds.extend(preds)

    model_preds = model_preds[:valid_data.num_examples]
    eval_losses = avg(eval_losses[:valid_data.num_examples])

    return model_preds, eval_losses


def write_summaries(sess, summary_handler, loss, eval_loss, bleu_1, bleu_4,
        meteor, rouge, cider, iteration):
    scores = {}
    scores['ITERATION'] = iteration
    scores['LOSS'] = loss
    scores['PERPLEXITY'] = exp(loss) 
    scores['EVAL_PERPLEXITY'] = exp(eval_loss) 
    scores['BLEU_1'] = bleu_1
    scores['BLEU_4'] = bleu_4
    scores['METEOR'] = meteor
    scores['ROUGE'] = rouge
    scores['CIDER'] = cider

    summary_handler.write_summaries(sess, scores) 

def avg(lst):
    avg = sum(lst) / len(lst)
    return avg
 
def gpu_config():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    return config

def restore_from_last_ckpt(config, model, sess):
    out_dir = os.path.join(config.out_root, config.model_name)
    ckpt_dir = os.path.join(out_dir, 'ckpts')

    ckpts = []
    for file in os.listdir(ckpt_dir):
        if file.endswith("index"):
            ckpts.append(file[:file.rfind(".")])

    ckpts.sort()

    last_ckpt = os.path.join(ckpt_dir, ckpts[-1])

    steps = last_ckpt[last_ckpt.find("step") + 4:]
    steps = int(steps[:steps.find("_")])

    epochs = last_ckpt[last_ckpt.find("epoch") + 6:]
    epochs = int(epochs[:epochs.find("_")])

    print("Restoring from %s" % last_ckpt)
    print("At epoch %d, step %d" % (epochs, steps))

    model.restore_from(sess, last_ckpt)

    print("Done")

    return epochs, steps, out_dir, ckpt_dir

def restore_from_step(config, model, sess, at_step):
    out_dir = os.path.join(config.out_root, config.model_name)
    ckpt_dir = os.path.join(out_dir, 'ckpts')

    ckpts = []
    for file in os.listdir(ckpt_dir):
        if file.endswith("index"):
            ckpts.append(file[:file.rfind(".")])

    for ckpt in ckpts:
        steps = ckpt[ckpt.find("step") + 4:]
        steps = int(steps[:steps.find("_")])

        if steps == at_step:
            print("Restoring from %s" % ckpt)
            ckpt_path = os.path.join(ckpt_dir, ckpt)
            model.restore_from(sess, ckpt_path)
            print("Done")

            return

    raise ValueError("No step found!")

def debug(config, *msg):
    if config.debug:
        print(msg)
