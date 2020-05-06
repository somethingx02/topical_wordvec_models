# -*- coding: utf8 -*-

# torch is a file named torch.py
import torch
# torch here is a folder named torch
from torch.autograd import Variable
# this is filename, once imported, you can use the classes in it

from models import topicalWordEmbedding

# equal to from models.topicalAttentionGRU import TopicalAttentionGRU
from settings import HALF_WINDOW_SIZE
from settings import HIDDEN_LAYER_SIZE
from settings import VOCABULARY_SIZE
from settings import TRAINING_INSTANCES
from settings import TOPIC_COUNT
from settings import DIM_ENCODER

from settings import DefaultConfig

from utils import *

import json

import os

import math
from scipy import spatial

import scipy.stats as stats

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# cuda device id
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

'''
compute the representations of all the words in vocabulary
and output them in a txt file with word\tdim1 dim2 ... dimN
It should be an iteration over the whole dataset
'''


def aggr_and_output_all_word_rep(
        model_dir,
        param_fpathin_index2vocab,
        param_fpathout_aggrd_all_wordrep,
        mtype='TopicalWordEmbedding'):
    '''
    The dataset take the trainset path
    ====================
    params:
    ----------
    model_dir: saved model dir
    param_fpathin_index2vocab: vocabfile
    mtype: model name

    return:
    ----------
    None
    '''
    # ----------load the index2vocabulary
    fpointerInIndex2Vocabulary = open(
        param_fpathin_index2vocab,
        'rt',
        encoding='utf8')
    dictIndex2Vocab = \
        json.load(fpointerInIndex2Vocabulary)
    fpointerInIndex2Vocabulary.close()
    config = DefaultConfig()

    batch_size = config.batch_size

    # ----------Compute the wordrep
    dictIndex2Wordvec = dict()
    for i in range(VOCABULARY_SIZE):
        dictIndex2Wordvec[i] = numpy.zeros(shape=HIDDEN_LAYER_SIZE,
                                           dtype=numpy.float32)

    # determine whether to run on cuda
    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if not config.on_cuda:
            logger.info('Cuda is unavailable,\
                Although wants to run on cuda,\
                Model still run on CPU')

    model_path = '%s/model' % model_dir

    if config.model == 'TopicalWordEmbedding':
        model = topicalWordEmbedding.TopicalWordEmbedding(
            param_on_cuda=config.on_cuda,
            param_half_window_size=HALF_WINDOW_SIZE,
            param_vocabulary_size=VOCABULARY_SIZE,
            param_hidden_layer_size=HIDDEN_LAYER_SIZE,
            param_encoder_pi_size=DIM_ENCODER,
            param_topic_count=TOPIC_COUNT)

    print('Loading trained model')
    if config.on_cuda:
        model.load(model_path)
        model = model.cuda()
    else:
        model.load_cpu_from_gputrained(model_path)
        model = model.cpu()

    train_data_manager = DataManager(batch_size, TRAINING_INSTANCES)
    train_data_manager.load_dataframe_from_file(TRAIN_SET_PATH)
    n_batch = train_data_manager.n_batches()
    batch_index = 0
    for batch_index in range(0, n_batch - 1):
        # this operation is time consuming
        xn, wc = train_data_manager.next_batch()
        idx = numpy.argmax(xn, axis=1)
        if config.on_cuda:
            var_xn = Variable(torch.from_numpy(xn).float()).cuda()
            # print( x.size() )
            var_wc = Variable(torch.from_numpy(wc).float(),
                              requires_grad=False).cuda()
        else:
            var_xn = Variable(torch.from_numpy(xn).float()).cpu()
            # print( x.size() )
            var_wc = Variable(torch.from_numpy(wc).float(),
                              requires_grad=False).cpu()
        var_rep = model.forward_obtain_xn_rep(var_xn, var_wc)
        arr_rep = var_rep.data.cpu().numpy()
        for row_idx, pivot_idx in enumerate(idx):
            pivot_rep = arr_rep[row_idx]
            dictIndex2Wordvec[pivot_idx] += pivot_rep
            # += pivot_rep
        # y = y - 1
        # print(y.size())

    if TRAINING_INSTANCES % batch_size == 0:
        train_data_manager.set_current_cursor_in_dataframe_zero()
    else:
        xn, wc = train_data_manager.tail_batch_nobatchpadding()
        idx = numpy.argmax(xn, axis=1)
        if config.on_cuda:
            var_xn = Variable(torch.from_numpy(xn).float()).cuda()
            # print( x.size() )
            var_wc = Variable(torch.from_numpy(wc).float(),
                              requires_grad=False).cuda()
        else:
            var_xn = Variable(torch.from_numpy(xn).float()).cpu()
            # print( x.size() )
            var_wc = Variable(torch.from_numpy(wc).float(),
                              requires_grad=False).cpu()
        var_rep = model.forward_obtain_xn_rep(var_xn, var_wc)
        arr_rep = var_rep.data.cpu().numpy()
        for row_idx, pivot_idx in enumerate(idx):
            pivot_rep = arr_rep[row_idx]
            dictIndex2Wordvec[pivot_idx] += pivot_rep
        # train_data_manager.set_current_cursor_in_dataframe_zero()

    # ----------Output the dict
    fpointerOutWordRep = open(param_fpathout_aggrd_all_wordrep,
                              'wt',
                              encoding='utf8')
    for an_word_idx in dictIndex2Wordvec:
        arr_word_rep = dictIndex2Wordvec[an_word_idx]
        arr_word_rep = arr_word_rep.astype(dtype=str)
        str_word_rep = ' '.join(arr_word_rep)
        str_vocab = dictIndex2Vocab[str(an_word_idx)]
        str4output = str_vocab + ' ' + str_word_rep + '\n'
        fpointerOutWordRep.write(str4output)
    fpointerOutWordRep.close()


def aggr_and_output_topic_all_word_rep(
        model_dir,
        param_fpathin_index2vocab,
        param_topic_index,
        param_fpathout_aggrd_topic_all_wordrep,
        mtype='TopicalWordEmbedding'):
    '''
    The dataset take the trainset path,
    compute the topic-dependent wordrep, highest in \zeta
    in a particular dimension
    ====================
    params:
    ----------
    model_dir: saved model dir
    param_fpathin_index2vocab: vocabfile
    param_topic_index: topic index specified
    param_fpathout_aggrd_topic_all_wordrep: the outputed all wordrep
    mtype: model name

    return:
    ----------
    None
    '''
    # ----------load the index2vocabulary
    fpointerInIndex2Vocabulary = open(
        param_fpathin_index2vocab,
        'rt',
        encoding='utf8')
    dictIndex2Vocab = \
        json.load(fpointerInIndex2Vocabulary)
    fpointerInIndex2Vocabulary.close()
    config = DefaultConfig()

    batch_size = config.batch_size

    # ----------Compute the wordrep
    dictIndex2Wordvec = dict()
    dictIndex2Zeta = dict()
    # for i in range(VOCABULARY_SIZE):
    #     dictIndex2Wordvec[i] = numpy.zeros(shape=HIDDEN_LAYER_SIZE,
    #                                        dtype=numpy.float32)
    #     dictIndex2Zeta[i] = numpy.zeros(shape=TOPIC_COUNT,
    #                                     dtype=numpy.float32)

    # determine whether to run on cuda
    if config.on_cuda:
        config.on_cuda = torch.cuda.is_available()
        if not config.on_cuda:
            logger.info('Cuda is unavailable,\
                Although wants to run on cuda,\
                Model still run on CPU')

    model_path = '%s/model' % model_dir

    if config.model == 'TopicalWordEmbedding':
        model = topicalWordEmbedding.TopicalWordEmbedding(
            param_on_cuda=config.on_cuda,
            param_half_window_size=HALF_WINDOW_SIZE,
            param_vocabulary_size=VOCABULARY_SIZE,
            param_hidden_layer_size=HIDDEN_LAYER_SIZE,
            param_encoder_pi_size=DIM_ENCODER,
            param_topic_count=TOPIC_COUNT)

    print('Loading trained model')
    if config.on_cuda:
        model.load(model_path)
        model = model.cuda()
    else:
        model.load_cpu_from_gputrained(model_path)
        model = model.cpu()

    train_data_manager = DataManager(batch_size, TRAINING_INSTANCES)
    train_data_manager.load_dataframe_from_file(TRAIN_SET_PATH)
    n_batch = train_data_manager.n_batches()
    batch_index = 0
    for batch_index in range(0, n_batch - 1):
        # this operation is time consuming
        xn, wc = train_data_manager.next_batch()
        idx = numpy.argmax(xn, axis=1)
        if config.on_cuda:
            var_xn = Variable(torch.from_numpy(xn).float()).cuda()
            # print( x.size() )
            var_wc = Variable(torch.from_numpy(wc).float(),
                              requires_grad=False).cuda()
        else:
            var_xn = Variable(torch.from_numpy(xn).float()).cpu()
            # print( x.size() )
            var_wc = Variable(torch.from_numpy(wc).float(),
                              requires_grad=False).cpu()
        var_zn = model.forward_obtain_xn_rep(var_xn, var_wc)
        var_zeta = model.forward_obtain_xn_zeta(var_xn, var_wc)
        # var_zeta_softmaxd = softmax(var_zeta, dim=1)
        arr_zn = var_zn.data.cpu().numpy()
        arr_zeta = var_zeta.data.cpu().numpy()
        for row_idx, pivot_idx in enumerate(idx):
            if pivot_idx not in dictIndex2Zeta:
                dictIndex2Zeta[pivot_idx] = arr_zeta[row_idx]
                dictIndex2Wordvec[pivot_idx] = arr_zn[row_idx]
            else:
                the_zeta = arr_zeta[row_idx]
                # if topic probability is higher than another
                if (the_zeta[param_topic_index] >
                        dictIndex2Zeta[pivot_idx][param_topic_index]):
                    dictIndex2Zeta[pivot_idx] = the_zeta
                    dictIndex2Wordvec[pivot_idx] = arr_zn[row_idx]

        # since it's no longer an aggregation, no softmax is required
        # for pivot_idx in idx:
        #     dictIndex2Wordvec[pivot_idx] = softmax_np(
        #         dictIndex2Wordvec[pivot_idx])

        # y = y - 1
        # print(y.size())

    if TRAINING_INSTANCES % batch_size == 0:
        train_data_manager.set_current_cursor_in_dataframe_zero()
    else:
        train_data_manager.set_current_cursor_in_dataframe_zero()

    # ----------Output the dict
    fpointerOutWordRep = open(param_fpathout_aggrd_topic_all_wordrep,
                              'wt',
                              encoding='utf8')
    for an_word_idx in dictIndex2Wordvec:
        arr_word_rep = dictIndex2Wordvec[an_word_idx]
        arr_word_rep = arr_word_rep.astype(dtype=str)
        str_word_rep = ' '.join(arr_word_rep)
        str_vocab = dictIndex2Vocab[str(an_word_idx)]
        str4output = str_vocab + ' ' + str_word_rep + '\n'
        fpointerOutWordRep.write(str4output)
    fpointerOutWordRep.close()


def dot_product(v1, v2):
    return sum(map(lambda x: x[0] * x[1], zip(v1, v2)))


def cosine_measure(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)


def compute_and_output_wordcossim(
        param_fpathin_vocab2vec,
        param_fpathin_gold_vocad,
        mtype='TopicalWordEmbedding'):
    '''
    The dataset take the trainset path
    ====================
    params:
    ----------
    param_fpathin_vocab2vec: vocabfile
    param_fpathin_gold_vocad: gold wordsim file
    param_fpathout_wordsim: outputed wordsim

    return:
    ----------
    None
    '''
    # ----------load the voca2vec
    fpointerInVoca2Vec = open(
        param_fpathin_vocab2vec,
        'rt',
        encoding='utf8')
    dictVoca2Vec = dict()
    for aline in fpointerInVoca2Vec:
        segs = aline.strip().split(' ')
        wordvoca = segs[0]
        arr_rep = numpy.array(segs[1:], dtype=str)
        arr_rep = arr_rep.astype(dtype=numpy.float32)
        dictVoca2Vec[wordvoca] = arr_rep
    fpointerInVoca2Vec.close()

    # ----------load the golden labels
    fpointerInGoldVocad = open(
        param_fpathin_gold_vocad,
        'rt',
        encoding='utf8')
    lst_score_predict = list()
    lst_score_gold = list()
    for aline in fpointerInGoldVocad:
        (word1, word2, score) = aline.strip().split(' ')
        score = float(score)
        lst_score_gold.append(score)
        word1rep = dictVoca2Vec[word1]
        word2rep = dictVoca2Vec[word2]
        # by cosine simlarity
        # lst_score_predict.append(cosine_measure(word1rep, word2rep))
        lst_score_predict.append(1 - spatial.distance.cosine(
            word1rep, word2rep))
    fpointerInGoldVocad.close()

    print(str(stats.stats.spearmanr(lst_score_predict, lst_score_gold)[0]))


if __name__ == '__main__':
    # # ==========Compute the word representations
    # aggr_and_output_all_word_rep(
    #     model_dir='%s/47' % SAVE_DIR,
    #     param_fpathin_index2vocab='../datasets/train_index_to_voca.txt',
    #     param_fpathout_aggrd_all_wordrep=('%s/47/aggrd_all_wordrep.txt'
    #                                       % SAVE_DIR))

    # # ==========Compute the topic-dependent word representations
    # aggr_and_output_topic_all_word_rep(
    #     model_dir='%s/1' % SAVE_DIR,
    #     param_fpathin_index2vocab='../datasets/train_index_to_voca.txt',
    #     param_topic_index=0,
    #     param_fpathout_aggrd_topic_all_wordrep=(
    #         '%s/1/aggrd_topic_all_wordrep.txt'
    #         % SAVE_DIR))

    # ==========Compute the word similarities for vocabularized benchmarks
    compute_and_output_wordcossim(
        param_fpathin_vocab2vec=('%s/99/aggrd_all_wordrep.txt'
                                 % SAVE_DIR),
        param_fpathin_gold_vocad='../datasets/wordsim353_agreed_vocad.txt',
        mtype='TopicalWordEmbedding')
