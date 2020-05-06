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

from scipy import spatial

import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# cuda device id
os.environ["CUDA_VISIBLE_DEVICES"] = "8"


def yelpDoclist2Parsedlist_noTokenize(
        paramDocList,
        paramPivotList,
        paramFpathInVocabulary2Index):
    '''
    convert a Doclist to tokenized, vocabularized list
    ====================
    '''

    # ----------tokenization
    listDocTokenized = list()
    for a_doc in paramDocList:
        tokens = a_doc.split(' ')
        listDocTokenized.append(tokens)
    paramDocList = None

    fpointerInVocabulary2Index = open(
        paramFpathInVocabulary2Index,
        'rt',
        encoding='utf8')
    dictVocabulary2Index = \
        json.load(fpointerInVocabulary2Index)
    fpointerInVocabulary2Index.close()

    listDocVocabularized = listDocTokenized
    # print(listDocVocabularized)

    # ----------construct pivot word list, xn, wc
    def __function_pivot_xn_wc(
            aVocabularizedDoc,
            aPivotWord):
        '''
        transfer a doc into several instances,
        each instance contains a pivot word and xn, wc
        '''
        tokenlist_size = len(aVocabularizedDoc)
        pivotinstances = list()
        vocabularySize = len(dictVocabulary2Index)
        for n in range(tokenlist_size):
            # ----------split aVocabularizedDoc section
            pivot_word = aVocabularizedDoc[n]
            if not pivot_word.startswith('<head>'):
                continue
            else:
                head_end_index = pivot_word.find('</head>')
                pivot_word = pivot_word[6:head_end_index]
            tokenlist_section = None
            if n - HALF_WINDOW_SIZE < 0:
                if n + HALF_WINDOW_SIZE >= tokenlist_size:
                    tokenlist_section = aVocabularizedDoc
                else:
                    tokenlist_section = \
                        aVocabularizedDoc[:n + HALF_WINDOW_SIZE]
            else:
                if n + HALF_WINDOW_SIZE >= tokenlist_size:
                    tokenlist_section = \
                        aVocabularizedDoc[n - HALF_WINDOW_SIZE:]
                else:
                    tokenlist_section = \
                        aVocabularizedDoc[n - HALF_WINDOW_SIZE:
                                          n + HALF_WINDOW_SIZE]
            # ----------calculate aVocabularizedDoc multiterm
            countlist_pivot = [dictVocabulary2Index[aPivotWord]]
            # [dictVocabulary2Index[pivot_word]]

            countlist_context = [0 for i in range(vocabularySize)]
            for atoken in tokenlist_section:
                if atoken not in dictVocabulary2Index:
                    continue
                if atoken == aVocabularizedDoc[n]:
                    pass
                else:
                    countlist_context[dictVocabulary2Index[atoken]] += 1
            # countlist_pivot.extend(countlist_context)
            pivotinstances.append((
                aPivotWord,
                # aVocabularizedDoc[n],
                countlist_pivot,
                countlist_context))
        return pivotinstances

    listPivotXnWcInstances = list()
    for idx, aDocVocabularized in enumerate(listDocVocabularized):
        thePivotWord = paramPivotList[idx]
        pivotXnWcInstances = __function_pivot_xn_wc(
            aDocVocabularized,
            thePivotWord)
        listPivotXnWcInstances.extend(pivotXnWcInstances)
    return listPivotXnWcInstances


def find_the_best_possible_pivot(
        param_list_pivot,
        param_arr_rep):
    '''
    find the closest pivot with regards to [0]
    ====================
    params:
    ----------
    param_list_pivot: input possible pivot list
    param_arr_rep: input pivot rep list

    return:
    ----------
    the_best_pivot: best match
    the_best_rep: best rep
    '''
    # original_pivot = param_list_pivot[0]
    original_rep = param_arr_rep[0]

    # list_sim = [1]
    # for idx_pivot, a_possible_pivot in enumerate(param_list_pivot[1:], 1):
    #     a_possible_rep = param_arr_rep[idx_pivot]
    #     list_sim.append(
    #         1 - spatial.distance.cosine(original_rep, a_possible_rep))
    combined_list = list(zip(param_list_pivot, param_arr_rep))

    def __key_compare(elem):
        rep_elem = elem[1]
        return 1 - spatial.distance.cosine(original_rep, rep_elem)
    combined_list.sort(key=__key_compare)
    (best_pivots, best_reps) = zip(*combined_list)
    return best_pivots[:900], best_reps[900]


def subst_compute_originalpivot_rep(
        model_dir,
        fpathIn_instances,
        fpathIn_labels,
        fpathOut_topcandidate,
        param_fpathin_voca2index,
        param_fpathin_subst_voca2index,
        mtype='TopicalWordEmbedding'):
    '''
    for each line in the fpathIn_instances,
    iterate the dict and construct a list, find the closest
    rep for the original
    ====================
    params:
    ----------
    model_dir: saved model dir
    fpathIn_instances: input filepath, parsed
    fpathOut_pivot_rep: output pivot_rep, it's pi
    param_fpathin_subst_voca2index: the dictionary for candidate
    mtype: model name

    return:
    ----------
    (pivot word list, rep list, topic rep list)
    '''

    # ----------load the subst_voca2index
    fpointerInSubstVocabulary2Index = open(
        param_fpathin_subst_voca2index,
        'rt',
        encoding='utf8')
    dictSubstVocabulary2Index = \
        json.load(fpointerInSubstVocabulary2Index)
    fpointerInSubstVocabulary2Index.close()

    # ---------- load the instance list
    fpointerIn_instances = open(fpathIn_instances, 'rt', encoding='utf8')
    list_instances = list(map(str.strip, fpointerIn_instances.readlines()))
    fpointerIn_instances.close()

    # ---------- load the pivot list
    fpointerIn_labels = open(fpathIn_labels, 'rt', encoding='utf8')
    list_pivots = list(map(str.strip, fpointerIn_labels.readlines()))
    for idx, aline_in_pivot in enumerate(list_pivots):
        list_pivots[idx] = aline_in_pivot.split(' ')[0]
    fpointerIn_labels.close()

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

    # ----------iterate over each instance to find the best, and output
    fpointerOut_topcandidate = open(
        fpathOut_topcandidate, 'wt', encoding='utf8')
    for idx_instance, a_candidate_instance in enumerate(list_instances):
        a_candidate_pivot = list_pivots[idx_instance]
        list_possible_instances = [a_candidate_instance]
        list_possible_pivots = [a_candidate_pivot]

        head_start_index = a_candidate_instance.find('<head>') + 6
        head_end_index = a_candidate_instance.find('</head>')

        for a_possible_pivot in dictSubstVocabulary2Index:
            if a_possible_pivot == a_candidate_pivot:
                continue

            a_possible_instance = a_candidate_instance[:head_start_index]\
                + a_possible_pivot + a_candidate_instance[head_end_index:]
            list_possible_instances.append(a_possible_instance)
            list_possible_pivots.append(a_possible_pivot)

        # ----------get a list of (pivot word, xn, wc)
        parsed_list = yelpDoclist2Parsedlist_noTokenize(
            paramDocList=list_possible_instances,
            paramPivotList=list_possible_pivots,
            paramFpathInVocabulary2Index=param_fpathin_voca2index)
        # print(len(parsed_list))

        (list_pivot, list_xn, list_wc) = zip(*parsed_list)
        # print(list_pivot)

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
        # var_zeta = model.forward_obtain_xn_zeta(var_xn, var_wc)
        arr_rep = var_rep.data.cpu().numpy()
        # arr_zeta = var_zeta.data.cpu().numpy()

        best_pivots, best_reps = find_the_best_possible_pivot(
            list_pivot, arr_rep)
        # print(list_possible_instances)
        # break
        # print(list_pivot[0], best_pivots)
        fpointerOut_topcandidate.write(' '.join(best_pivots) + '\n')

    fpointerOut_topcandidate.close()


if __name__ == '__main__':
    subst_compute_originalpivot_rep(
        model_dir='%s/5' % SAVE_DIR,
        fpathIn_instances='../datasets/lexsub_f2_instance_vocabed.txt',
        fpathIn_labels='../datasets/lexsub_f2_labels_vocabed.txt',
        fpathOut_topcandidate='../datasets/lex_sub_top_candidate.txt',
        param_fpathin_voca2index='../datasets/train_voca_to_index.txt',
        param_fpathin_subst_voca2index=('../datasets/lexsub_'
                                        'candidate_vocab2index.txt'),
        mtype='TopicalWordEmbedding')
