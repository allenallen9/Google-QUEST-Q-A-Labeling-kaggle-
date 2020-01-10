import os
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image, ImageFont, ImageDraw, ImageChops
import tensorflow as tf
# import tensorflow.python.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import math
import random
import cv2
import pandas as pd
import tensorflow_hub as hub
import tokenization
from math import floor, ceil
from tqdm import tqdm
np.random.seed(8888)

def set_gpu_config(device = 0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[device], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[device], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)


PATH = 'qal_1/'
DATA_PATH = 'dataset/'
BERT_PATH = 'pretrain/bert_en_uncased_L-12_H-768_A-12/'
MAX_SEQUENCE_LENGTH = 512
tokenizer = tokenization.FullTokenizer(BERT_PATH + 'assets/vocab.txt', do_lower_case=True)
#result

if not os.path.exists(PATH):
    os.mkdir(PATH)


def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))
def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))
def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
    return input_ids
def _trim_input(question, answer, max_sequence_length):
    """Trims tokenized input to max_sequence_length,
    while keeping the same ratio of Q and A length"""
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)

    q_len = len(q)
    a_len = len(a)

    if (q_len + a_len + 3) > max_sequence_length:

        new_q_len = q_len / (a_len + q_len) * (max_sequence_length - 3)
        new_a_len = a_len / (q_len + a_len) * (max_sequence_length - 3)
        new_q_len, new_a_len = int(ceil(new_q_len)), int(floor(new_a_len))

        if new_a_len + new_q_len + 3 != max_sequence_length:
            raise ValueError("too small %s" % str(new_a_len + new_q_len + 3))

        q = q[:new_q_len]
        a = a[:new_a_len]

    return q, a
def _convert_to_bert_inputs(question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""

    stoken = ["[CLS]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]
def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm(df[columns].iterrows(),total=df.shape[0]):
        q, a = instance.question_body, instance.answer

        q, a = _trim_input(q, a, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(
            q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)]
def compute_output_arrays(df, columns):
    return np.asarray(df[columns])
def bert_model():
    input_word_ids  = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_word_ids')
    input_masks     = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    input_segments  = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')

    bert_layer = hub.KerasLayer(BERT_PATH, trainable=True)
    # bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)

    pooled_output, _ = bert_layer([input_word_ids, input_masks, input_segments])

    x = tf.keras.layers.Dropout(0.3)(pooled_output)
    x = tf.keras.layers.Dense(768, activation='elu', name='dense_penultimate')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(30, activation="sigmoid", name="dense_output")(x)

    model = tf.keras.models.Model(
        inputs=[input_word_ids, input_masks, input_segments], outputs=out)

    return model

def trainFunc():

    df_train = pd.read_csv(DATA_PATH + 'train.csv')
    df_test = pd.read_csv(DATA_PATH + 'test.csv')
    df_sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

    # df_train = df_train[0:5]
    # df_test = df_test[0:5]
    # df_sub = df_sub[0:5]

    print('train shape =', df_train.shape)
    print('test shape =', df_test.shape)
    print('sub shape =', df_sub.shape)

    output_categories = list(df_train.columns[11:])
    input_categories = list(df_train.columns[[2, 5]])
    print('\noutput categories:\n\t', output_categories)
    print('\ninput categories:\n\t', input_categories)

    outputs = compute_output_arrays(df_train, output_categories)
    inputs = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    test_inputs = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

    print(len(inputs))
    print(inputs[0].shape)
    print(len(outputs))
    print(len(test_inputs))

    ##data
    train_inputs = inputs
    train_outputs = outputs

    ####
    set_gpu_config(1)
    model = bert_model()
    model.summary()

    class trainLossCallback(tf.keras.callbacks.Callback):

        def __init__(self):
            self.train_loss_hist=[]
        def on_batch_end(self, batch, logs=None):
            self.train_loss_hist.append(logs.get('loss'))
        def on_epoch_end(self, epoch, logs=None):
            plt.plot(self.train_loss_hist)
            plt.title('train_loss')
            plt.ylabel('loss')
            plt.xlabel('iter')
            plt.legend(['train'], loc='upper right')
            plt.savefig(PATH + "train_loss.png")
            plt.close()
    train_loss_callback = trainLossCallback()
    callback=[train_loss_callback]

    learning_rate = 3e-5
    epochs = 10
    batch_size = 8
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=optimizer)
    model.fit(train_inputs, train_outputs, epochs=epochs, batch_size=batch_size, callback=callback)

    # model.save(PATH + 'model.h5')
    model.save_weights(PATH + 'model_weight.h5')
def testFunc():
    df_train = pd.read_csv(DATA_PATH + 'train.csv')
    df_test = pd.read_csv(DATA_PATH + 'test.csv')
    df_sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

    output_categories = list(df_train.columns[11:])
    input_categories = list(df_train.columns[[2, 5]])
    print('\noutput categories:\n\t', output_categories)
    print('\ninput categories:\n\t', input_categories)

    # outputs = compute_output_arrays(df_train, output_categories)
    # inputs = compute_input_arays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    test_inputs = compute_input_arays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

    ####
    set_gpu_config(1)
    model = bert_model()
    model.summary()
    model.load_weights(PATH+'model_weight.h5')

    test_pred = model.predict(test_inputs, batch_size=8)

    # test_predictions = [histories[i].test_predictions for i in range(len(histories))]
    # test_predictions = [np.average(test_predictions[i], axis=0) for i in range(len(test_predictions))]
    # test_predictions = np.mean(test_predictions, axis=0)

    df_sub.iloc[:, 1:] = test_pred

    df_sub.to_csv(PATH + 'submission.csv', index=False)

if __name__ == '__main__':
    trainFunc()
    # testFunc()
