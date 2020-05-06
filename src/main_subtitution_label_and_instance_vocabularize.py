# -*- coding: utf8 -*-

import json

if __name__ == '__main__':

    # ====================read in the vocab
    paramFpathInVocabulary2Index = '../datasets/train_voca_to_index.txt'
    fpointerInVocabulary2Index = open(
        paramFpathInVocabulary2Index,
        'rt',
        encoding='utf8')
    dictVocabulary2Index = \
        json.load(fpointerInVocabulary2Index)
    fpointerInVocabulary2Index.close()
    print(len(dictVocabulary2Index))

    # ====================read in label and pivot simultaneously
    fnameIn_instance = '../datasets/lexsub_f_instance.txt'
    fnameIn_labels = '../datasets/lexsub_f_labels.txt'
    fnameOut_instance = '../datasets/lexsub_f1_instance_vocabed.txt'
    fnameOut_labels = '../datasets/lexsub_f1_labels_vocabed.txt'
    fnameOut_candidate_vocab2index = \
        '../datasets/lexsub_candidate_vocab2index.txt'
    fnameOut_candidate_index2vocab = \
        '../datasets/lexsub_candidate_index2vocab.txt'
    dict_candidate_vocab2index = dict()
    dict_candidate_index2vocab = dict()
    fpIn_instance = open(
        fnameIn_instance, 'rt', encoding='utf8')
    fpIn_labels = open(
        fnameIn_labels, 'rt', encoding='utf8')
    fpOut_instance = open(
        fnameOut_instance, 'wt', encoding='utf8')
    fpOut_labels = open(
        fnameOut_labels, 'wt', encoding='utf8')
    # ---------- filter using labels
    for aline_label in fpIn_labels:
        aline_labels = aline_label.strip().split(' ')
        aline_instance = fpIn_instance.readline()
        if aline_labels[0] not in dictVocabulary2Index:
            continue
        dict_candidate_vocab2index[aline_labels[0]] = \
            dictVocabulary2Index[aline_labels[0]]
        dict_candidate_index2vocab[dictVocabulary2Index[aline_labels[0]]] = \
            aline_labels[0]
        candidate_word_check = False
        new_labels = aline_labels[0]
        for a_candidate_word in aline_labels[1:]:
            if a_candidate_word in dictVocabulary2Index:
                dict_candidate_vocab2index[a_candidate_word] = \
                    dictVocabulary2Index[a_candidate_word]
                dict_candidate_index2vocab[
                    dictVocabulary2Index[a_candidate_word]] = \
                    a_candidate_word
                candidate_word_check = True
                new_labels = new_labels + ' ' + a_candidate_word
        if not candidate_word_check:
            continue
        fpOut_instance.write(aline_instance)
        fpOut_labels.write(new_labels + '\n')

    fpOut_instance.close()
    fpOut_labels.close()
    fpIn_instance.close()
    fpIn_labels.close()
    # ---------- filter using instances
    fnameIn_instance = '../datasets/lexsub_f1_instance_vocabed.txt'
    fnameIn_labels = '../datasets/lexsub_f1_labels_vocabed.txt'
    fnameOut_instance = '../datasets/lexsub_f2_instance_vocabed.txt'
    fnameOut_labels = '../datasets/lexsub_f2_labels_vocabed.txt'

    fpIn_instance = open(
        fnameIn_instance, 'rt', encoding='utf8')
    fpIn_labels = open(
        fnameIn_labels, 'rt', encoding='utf8')
    fpOut_instance = open(
        fnameOut_instance, 'wt', encoding='utf8')
    fpOut_labels = open(
        fnameOut_labels, 'wt', encoding='utf8')
    for aline_instance in fpIn_instance:
        aline_instances = aline_instance.strip().split(' ')
        aline_label = fpIn_labels.readline()
        new_instance = ''
        candidate_word_check = False
        for aword in aline_instances:
            if aword.startswith('<head>'):
                new_instance = new_instance + aword + ' '
                continue
            if aword not in dictVocabulary2Index:
                continue
            candidate_word_check = True
            new_instance = new_instance + aword + ' '
            dict_candidate_vocab2index[aword] = \
                dictVocabulary2Index[aword]
            dict_candidate_index2vocab[dictVocabulary2Index[aword]] = \
                aword
        if not candidate_word_check:
            continue
        fpOut_instance.write(new_instance.strip() + '\n')
        fpOut_labels.write(aline_label)

    fpOut_instance.close()
    fpOut_labels.close()
    fpIn_instance.close()
    fpIn_labels.close()

    # ==================== Output the candidate dictionary
    fp4jsonoutput = open(fnameOut_candidate_vocab2index,
                         'wt', encoding='utf8')
    json.dump(dict_candidate_vocab2index, fp4jsonoutput, ensure_ascii=False)
    fp4jsonoutput.close()

    fp4jsonoutput = open(fnameOut_candidate_index2vocab,
                         'wt', encoding='utf8')
    json.dump(dict_candidate_index2vocab, fp4jsonoutput, ensure_ascii=False)
    fp4jsonoutput.close()
