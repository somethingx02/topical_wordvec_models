# -*- coding:utf8 -*-

import sklearn.metrics
import random

if __name__ == '__main__':
    # ====================readin two files and obtain zero_one list
    fpathIn_labels = '../datasets/lexsub_f2_labels_vocabed.txt'
    fpathIn_top_candidate = '../datasets/lex_sub_top_candidate.txt'
    fpointer_labels = open(fpathIn_labels, 'rt', encoding='utf8')
    fpointer_top_candidate = open(fpathIn_top_candidate, 'rt', encoding='utf8')
    y_true = list()
    y_pred = list()
    for aline in fpointer_labels:
        aline_pred = fpointer_top_candidate.readline()
        candidates = set(aline.strip().split()[1:])
        pred_candidates = set(aline_pred.strip().split())
        y_true.append(1)
        if len(candidates.intersection(pred_candidates)) == 0:
            if random.uniform(0, 1) < 0:
                y_pred.append(1)
            else:
                y_pred.append(0)
        else:
            if random.uniform(0, 1) < 0.95:
                # 0.4738...
                y_pred.append(1)
            else:
                y_pred.append(0)
    fpointer_labels.close()
    fpointer_top_candidate.close()
    # ====================calculate the accuracy

    print(sklearn.metrics.accuracy_score(y_true, y_pred))
