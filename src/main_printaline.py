# -*- coding: utf8 -*-

if __name__ == '__main__':
    fpIn = open('../datasets/train_sparseinstances.csv', 'rt', encoding='utf8')
    aline = fpIn.readline()
    print(aline)
    print(len(aline.strip().split(' ')))
    fpIn.close()
