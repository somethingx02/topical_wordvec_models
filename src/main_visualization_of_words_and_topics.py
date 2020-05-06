# -*- coding: utf8 -*-

# torch is a file named torch.py
import torch
# torch here is a folder named torch
from torch.autograd import Variable
# from torchnet import meter
# this is filename, once imported, you can use the classes in it

from models import topicalWordEmbedding

# equal to from models.topicalAttentionGRU import TopicalAttentionGRU
from settings import HALF_WINDOW_SIZE
from settings import HIDDEN_LAYER_SIZE
from settings import VOCABULARY_SIZE
# from settings import TRAINING_INSTANCES
from settings import TOPIC_COUNT
from settings import DIM_ENCODER

from settings import DefaultConfig

from utils import *

import numpy

import os
import operator

import json

from main_yelp_preprocessor import YelpPreprocessor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# cuda device id
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def softmax_np(
        x):
    '''
    a softmax function for numpy
    '''
    return numpy.divide(
        numpy.exp(x), numpy.sum(numpy.exp(x), axis=0))


def display_sorted_topic_matrix(
        model_dir,
        param_fpathin_index2vocab,
        mtype='TopicalWordEmbedding'):
    '''
    get the topic matrix, for each topic, concatenate, sort and
    map the top 10 words
    ====================
    params:
    ----------
    model_dir: saved model dir
    param_fpathin_voca2index: input dict dir
    mtype: model name

    return:
    ----------
    None, output to the console
    '''
    # ----------load the voca2index
    fpointerInIndex2Vocabulary = open(
        param_fpathin_index2vocab,
        'rt',
        encoding='utf8')
    dictIndex2Vocabulary = \
        json.load(fpointerInIndex2Vocabulary)
    fpointerInIndex2Vocabulary.close()

    # ----------load the trained model
    config = DefaultConfig()
    # config.set_attrs({'batch_size': len(list_pivot)})
    model_path = '%s/model' % model_dir

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

    # ----------get the topic matrix
    var_topic_matrix = model.vae_decoder.MATRIX_decoder_beta
    arr_topic_matrix = var_topic_matrix.data.cpu().numpy()
    itemgetter_1 = operator.itemgetter(1)
    for topic_index in range(TOPIC_COUNT):
        list_voca = list(range(VOCABULARY_SIZE))
        list_topicvoca = arr_topic_matrix[topic_index, :].tolist()
        # concatenate
        list_voca_topicvoca = list(zip(list_voca, list_topicvoca))
        list_voca_topicvoca.sort(key=itemgetter_1, reverse=True)
        (list_voca, list_topicvoca) = zip(*list_voca_topicvoca)
        top_list_voca = list_voca[0: 50]
        top_list_voca_mapped = [
            dictIndex2Vocabulary[str(i)] for i in top_list_voca]
        print(top_list_voca_mapped)

        list_voca = None
        list_topicvoca = None
        list_voca_topicvoca = None

    return None


def output_sorted_topic_matrix(
        model_dir,
        param_fpathin_index2vocab,
        param_fpathout_topic_matrix,
        mtype='TopicalWordEmbedding'):
    '''
    get the topic matrix, for each topic, concatenate, sort and
    map the top 10 words
    ====================
    params:
    ----------
    model_dir: saved model dir
    param_fpathin_voca2index: input dict dir
    mtype: model name
    param_fpathout_topic_matrix: output_topic_matrix_path

    return:
    ----------
    None, output to the console
    '''
    # ----------load the voca2index
    fpointerInIndex2Vocabulary = open(
        param_fpathin_index2vocab,
        'rt',
        encoding='utf8')
    dictIndex2Vocabulary = \
        json.load(fpointerInIndex2Vocabulary)
    fpointerInIndex2Vocabulary.close()

    # ----------load the trained model
    config = DefaultConfig()
    # config.set_attrs({'batch_size': len(list_pivot)})
    model_path = '%s/model' % model_dir

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

    # ----------get and output the topic matrix
    fpointerOutTopicMatrix = open(param_fpathout_topic_matrix,
                                  'wt',
                                  encoding='utf8')

    var_topic_matrix = model.vae_decoder.MATRIX_decoder_beta
    arr_topic_matrix = var_topic_matrix.data.cpu().numpy()
    itemgetter_1 = operator.itemgetter(1)
    for topic_index in range(TOPIC_COUNT):
        list_voca = list(range(VOCABULARY_SIZE))
        list_topicvoca = arr_topic_matrix[topic_index, :].tolist()
        # concatenate
        list_voca_topicvoca = list(zip(list_voca, list_topicvoca))
        list_voca_topicvoca.sort(key=itemgetter_1, reverse=True)
        (list_voca, list_topicvoca) = zip(*list_voca_topicvoca)
        top_list_voca = list_voca[0: 50]
        top_list_voca_mapped = [
            dictIndex2Vocabulary[str(i)] for i in top_list_voca]
        top_list_voca_cleaned = [
            vocastr for vocastr in top_list_voca_mapped
            if (vocastr.find('~') == -1 and
                vocastr.find('!') == -1 and
                vocastr.find('@') == -1 and
                vocastr.find('#') == -1 and
                vocastr.find('$') == -1 and
                vocastr.find('%') == -1 and
                vocastr.find('^') == -1 and
                vocastr.find('&') == -1 and
                vocastr.find('*') == -1 and
                vocastr.find('(') == -1 and
                vocastr.find(')') == -1 and
                vocastr.find('0') == -1 and
                vocastr.find('1') == -1 and
                vocastr.find('2') == -1 and
                vocastr.find('3') == -1 and
                vocastr.find('4') == -1 and
                vocastr.find('5') == -1 and
                vocastr.find('6') == -1 and
                vocastr.find('7') == -1 and
                vocastr.find('8') == -1 and
                vocastr.find('9') == -1 and
                vocastr.find('-') == -1 and
                vocastr.find('+') == -1 and
                vocastr.find('_') == -1 and
                vocastr.find('=') == -1 and
                vocastr.find('.') == -1 and
                vocastr.find(',') == -1 and
                vocastr.find('/') == -1 and
                vocastr.find('?') == -1 and
                vocastr.find('\\') == -1 and
                vocastr.find('"') == -1 and
                vocastr.find(':') == -1 and
                vocastr.find('\'') == -1 and
                vocastr.find(';') == -1 and
                vocastr.find('|') == -1 and
                vocastr.find('<') == -1 and
                vocastr.find('>') == -1 and
                vocastr.find('[') == -1 and
                vocastr.find(']') == -1)]
        top_list_voca_top10 = top_list_voca_cleaned[:10]
        fpointerOutTopicMatrix.write(
            'topic %03d ' % topic_index + ' '.join(top_list_voca_top10) + '\n')

        list_voca = None
        list_topicvoca = None
        list_voca_topicvoca = None
    fpointerOutTopicMatrix.close()
    return None



def compute_pivot_rep(
        model_dir,
        input_doc_list,
        param_fpathin_voca2index,
        mtype='TopicalWordEmbedding'):
    '''
    given a list of documents, transfer the documents into instances,
    enumerate the instances and compute the pivot representations.
    ====================
    params:
    ----------
    model_dir: saved model dir
    input_list: input documents, unparsed
    mtype: model name

    return:
    ----------
    (pivot word list, rep list, topic rep list)
    '''

    # ----------load the voca2index
    # fpointerInVocabulary2Index = open(
    #     param_fpathin_voca2index,
    #     'rt',
    #     encoding='utf8')
    # dictVocabulary2Index = \
    #     json.load(fpointerInVocabulary2Index)
    # fpointerInVocabulary2Index.close()

    # ----------get a list of (pivot word, xn, wc)
    oYelpPreprocessor = YelpPreprocessor()
    parsed_list = oYelpPreprocessor.yelpDoclist2Parsedlist(
        paramDocList=input_doc_list,
        paramFpathInVocabulary2Index=param_fpathin_voca2index)

    (list_pivot, list_xn, list_wc) = zip(*parsed_list)
    # ----------load the trained model
    config = DefaultConfig()
    config.set_attrs({'batch_size': len(list_pivot)})
    model_path = '%s/model' % model_dir

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

    # ----------compute the representation list
    arr_xn = numpy.zeros((len(list_xn),
                          VOCABULARY_SIZE),
                         dtype=numpy.int32)
    for list_xn_linenum, list_xn_vocabindex in enumerate(list_xn):
        arr_xn[list_xn_linenum, list_xn_vocabindex] += 1
    arr_xn = arr_xn.astype(numpy.float32)
    arr_wc = numpy.array(list_wc).astype(numpy.float32)
    if config.on_cuda:
        var_xn = Variable(torch.from_numpy(arr_xn)).cuda()
        var_wc = Variable(torch.from_numpy(arr_wc)).cuda()
    else:
        var_xn = Variable(torch.from_numpy(arr_xn)).cpu()
        var_wc = Variable(torch.from_numpy(arr_wc)).cpu()
    var_rep = model.forward_obtain_xn_rep(var_xn, var_wc)
    var_zeta = model.forward_obtain_xn_zeta(var_xn, var_wc)
    arr_rep = var_rep.data.cpu().numpy()
    arr_zeta = var_zeta.data.cpu().numpy()

    return list_pivot, arr_rep, arr_zeta


if __name__ == '__main__':
    # ----------display topics
    display_sorted_topic_matrix(
        model_dir='%s/27' % SAVE_DIR,
        param_fpathin_index2vocab='../datasets/train_index_to_voca.txt',
        mtype='TopicalWordEmbedding')

    # ----------display topics to files
    output_sorted_topic_matrix(
        model_dir='%s/40' % SAVE_DIR,
        param_fpathin_index2vocab='../datasets/train_index_to_voca.txt',
        param_fpathout_topic_matrix='../datasets/jtwTopicMatrix.txt',
        mtype='TopicalWordEmbedding')

    # ----------display topical distribution for the selected pivot word
    doc_list = [
        # ('If any one in the UK is thinking of '
        #     'plastic surgery abroad this doctor and clinic '
        #     'I would most certainly recommend.'),
        # ('Online reviews of surgeons who perform '
        #     'plastic surgery may be unreliable, researchers say.'),
        ('Effective patient care requires clinical knowledge'
         ' and understanding of physical therapy'),
        ('Restaurant servers require patient temperament'),
        ('Here are 4 ways to reduce the plastic bags.')]

    # ('We have recently moved in to the area. When I '
    #     'saw this place, I thought it would be nice to '
    #     'have a hairstylist close by so one Saturday morning '
    #     'I popped in to look around. While looking around '
    #     'the store I have noticed on one of the shelves plastic '
    #     'brushes that resembled exactly a brush I have for '
    #     'detangling my hair. I wanted to buy this brush for '
    #     'my daughter. I couldn\'t find the price on the '
    #     'package so I went to the lady (short, medium long dark hair) '
    #     'at the cash register to ask how much it was. "The price is '
    #     'there!", she said rudely and went to the shelve to show me.'
    #     ' She grabbed the brush and pointed her finger at the price'
    #     ' label I was guilty of not noticing earlier. Wow, I\'m sorry '
    #     'lady I have offended you in any way on this beautiful'
    #     ' Saturday morning! I don\'t think I will ever go back there. '
    #     'I am positive, I will not go back there ever again. If they '
    #     'treat potential customers like that right in the beginning, '
    #     'the service must suck there too. By the way, the plastic brush '
    #     'was $26 and it looked exactly like the one I bought for myself '
    #     'a few months earlier somewhere else and paid $6.')]

    # list_pivot, arr_rep, arr_zeta = compute_pivot_rep(
    #     model_dir='%s/17' % SAVE_DIR,
    #     input_doc_list=doc_list,
    #     param_fpathin_voca2index='../datasets/train_voca_to_index.txt',
    #     mtype='TopicalWordEmbedding')

    # for pivot_linenum, the_pivot in enumerate(list_pivot):
    #     if the_pivot == 'patient':
    #         softmaxed_pivot_zeta = softmax_np(arr_zeta[pivot_linenum])
    #         print(softmaxed_pivot_zeta)

    # ====================Testing Centre
    # arr = numpy.array([[1, 2, 3], [4, 5, 6]])
    # lst = arr.tolist()
    # print(lst)
    # lst_arrayelem = list(arr)
    # print(lst_arrayelem)
    # arr1 = arr[:, 1]
    # print(arr1)
