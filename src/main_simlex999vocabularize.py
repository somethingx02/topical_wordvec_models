# -*- coding: utf8 -*-

import json


if __name__ == '__main__':
    paramFpathInVocabulary2Index = '../datasets/train_voca_to_index.txt'
    fpointerInVocabulary2Index = open(
        paramFpathInVocabulary2Index,
        'rt',
        encoding='utf8')
    dictVocabulary2Index = \
        json.load(fpointerInVocabulary2Index)
    fpointerInVocabulary2Index.close()
    print(len(dictVocabulary2Index))

    fptInWS353 = open('../datasets/SimLex-999.txt',
                      'rt',
                      encoding='utf8')
    fptOutWS353vocad = open('../datasets/SimLex-999_vocad.txt', 'wt',
                            encoding='utf8')
    fptInWS353.readline()
    for aline in fptInWS353:
        aline = aline.strip().lower()
        (word1, word2, pos, score, other) = aline.split('\t', 4)
        if word1 in dictVocabulary2Index and word2 in dictVocabulary2Index:
            fptOutWS353vocad.write(word1 + ' ' + word2 + ' ' + score + '\n')
    fptInWS353.close()
    fptOutWS353vocad.close()
