# -*- coding: utf8 -*-

import json

import math
import random
import operator

import os


from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from settings import HALF_WINDOW_SIZE


class YelpPreprocessor:
    '''
    Convert Yelp Preprocessor to a text file consisting of lines
    ======================================================
    parameters:
    ----------
    None

    return:
    ----------
    None
    '''

    def __init__(self):
        return None

    def json_preprocess_clinical(self,
                                 paramFpathInBussiness,
                                 paramFpathInReview,
                                 paramFpathOutReview,
                                 paramFpathOutStars):
        '''
        retrieve clinical documents from business, reviews
        ==================================================
        parameters:
        -----------
        paramFpathInBussiness: business file
        paramFpathInReview: review file
        paramFpathOutReview: texted review
        paramFpathOutStars: stars file

        return:
        -----------
        None
        '''

        # dictClinicalBussinessCategories = {
        #     "Chiropractic and physical therapy":
        #     {
        #         "includeif":
        #         {"chiropractor", "physical therapy"},
        #         "excludeif": {}
        #     },
        #         "Dental":
        #         {"includeif": {"dentist"}, "excludeif": {}},
        #     "Dermatology":
        #     {"includeif": {"dermatologist"},
        #      "excludeif": {"optometrist", "veterinarians", "pets"}},
        #     "Family practice": {"includeif": {"family practice"},
        #                         "excludeif": {"psychiatrist", "chiropractor",
        #                                       "beauty", "physical therapy",
        #                                       "specialty", "dermatologists",
        #                                       "weight loss", "acupuncture",
        #                                       "cannabis clinics",
        #                                       "naturopathic",
        #                                       "optometrists"}},
        #     "Hospitals and clinics": {"includeif": {"hospital"},
        #                               "excludeif": {"physical therapy",
        #                                             "rehab",
        #                                             "retirement homes",
        #                                             "veterinarians",
        #                                             "dentist"}},
        #     "Optometry": {"includeif": {"optometrist"},
        #                   "excludeif": {"dermatologist"}},
        #     "Mental health": {"includeif": {"psychiatrist", "psychologist"},
        #                       "excludeif": {}},
        #     "Dental": {"includeif": {"speech therapy"},
        #                "excludeif": {"speech"}},
        # }
        dictClinicalBussinessCategories = {
            "Chiropractic and physical therapy":
                {"includeif": {"Chiropractor", "Physical Therapy"},
                 "excludeif": set()},
            "Dental": {"includeif": {"Dentist"}, "excludeif": set()},
            "Dermatology": {"includeif": {"Dermatologist"},
                            "excludeif": {"Optometrist",
                                          "Veterinarians",
                                          "Pets"}},
            "Family practice": {"includeif": {"Family Practice"},
                                "excludeif": {"Psychiatrist",
                                              "Chiropractor",
                                              "Beauty",
                                              "Physical Therapy",
                                              "Specialty",
                                              "Dermatologists",
                                              "Weight Loss",
                                              "Acupuncture",
                                              "Cannabis Clinics",
                                              "Naturopathic",
                                              "Optometrists"}},
            "Hospitals and clinics": {"includeif": {"Hospital"},
                                      "excludeif": {"Physical Therapy",
                                                    "Rehab",
                                                    "Retirement Homes",
                                                    "Veterinarians",
                                                    "Dentist"}},
            "Optometry": {"includeif": {"Optometrist"},
                          "excludeif": {"Dermatologist"}},
            "Mental health": {"includeif": {"Psychiatrist", "Psychologist"},
                              "excludeif": set()},
            "Dental": {"includeif": {"Speech Therapy"},
                       "excludeif": {"Speech"}},
        }
        setClinicalBussinessIds = set()
        fpointerInReview = open(paramFpathInBussiness, 'rt', encoding='utf8')
        for aline in fpointerInReview:
            aline = aline.strip()
            jo = json.loads(aline)
            bussinessCategoryAttributes = jo['categories']
            setBussinessCategory = set(bussinessCategoryAttributes)
            for akey in dictClinicalBussinessCategories:
                includedifset = \
                    dictClinicalBussinessCategories[akey]['includeif']
                excludedifset = \
                    dictClinicalBussinessCategories[akey]['excludeif']
                if len(includedifset.intersection(setBussinessCategory)) != 0:
                    if len(excludedifset.intersection(
                            setBussinessCategory)) != 0:
                        pass
                    else:
                        bussinessId = jo['business_id']
                        setClinicalBussinessIds.add(bussinessId)
                        break

        fpointerInReview.close()
        # print( setClinicalBussinessIds)

        fpointerInReview = open(paramFpathInReview, 'rt', encoding='utf8')
        fpointerOutReview = open(paramFpathOutReview, 'wt', encoding='utf8')
        fpointerOutStars = open(paramFpathOutStars, 'wt', encoding='utf8')
        for aline in fpointerInReview:
            aline = aline.strip()
            jo = json.loads(aline)

            bussinessId = jo['business_id']
            if bussinessId in setClinicalBussinessIds:
                reviewText = jo['text']
                reviewText = reviewText.replace('\r\n', ' ')
                reviewText = reviewText.replace('\r', ' ')
                # actually in Unbuntu and Windows only
                # this line went into effect
                reviewText = reviewText.replace('\n', ' ')
                reviewStar = jo['stars']
                fpointerOutReview.write(reviewText + '\n')
                fpointerOutStars.write(str(reviewStar) + '\n')

            # print(reviewText)
            # break
        fpointerInReview.close()
        fpointerOutReview.close()
        fpointerOutStars.close()

    def json_preprocess_all(self,
                            paramFpathInReview,
                            paramFpathOutReview):
        '''
        retrieve documents from all reviews
        ==================================================
        parameters:
        -----------
        paramFpathInReview: review file
        paramFpathOutReview: texted review

        return:
        -----------
        None
        '''

        fpointerInReview = open(paramFpathInReview, 'rt', encoding='utf8')
        fpointerOutReview = open(paramFpathOutReview, 'wt', encoding='utf8')
        for aline in fpointerInReview:
            aline = aline.strip()
            jo = json.loads(aline)

            reviewText = jo['text']
            reviewText = reviewText.replace('\r\n', ' ')
            reviewText = reviewText.replace('\r', ' ')
            # actually in Unbuntu and Windows only
            # this line went into effect
            reviewText = reviewText.replace('\n', ' ')
            fpointerOutReview.write(reviewText + '\n')

            # print(reviewText)
            # break
        fpointerInReview.close()
        fpointerOutReview.close()

    def yelpTrainAndTestConstructFromWhole(
            self,
            paramFpathInReview,
            paramFpathOutTrain,
            paramFpathOutTest,
            paramTrainsetPercent=0.9):
        '''
        combine reviews with stars, reshuffle reviews, and split into two sets
        ===================================================
        parameters:
        -----------
        paramFpathInReview: texted review
        paramFpathOutTrain: train set
        paramFpathOutTest: test set
        paramFpathOutParams: the parameters needed for training
        paramTrainsetPercent: train set percent

        return:
        -----------
        None
        '''

        fpointerInReview = open(paramFpathInReview, 'rt', encoding='utf8')

        def __function4map(elem4map):
            '''
            stripe elem
            ===================================================
            parameters:
            -----------
            elem4map

            return:
            -----------
            mapped elem
            '''
            # no stripe since fp.writelines do not add \n
            elemnotstriped = elem4map
            return elemnotstriped

        listReviews = list(map(__function4map, fpointerInReview.readlines()))
        fpointerInReview.close()

        random.shuffle(listReviews)

        paramTrainsetSize = math.floor(paramTrainsetPercent * len(listReviews))
        listTrainset = listReviews[:paramTrainsetSize]
        listTestset = listReviews[paramTrainsetSize:]

        fpointerOutTrain = open(
            os.path.splitext(paramFpathOutTrain)[0] + '.txt',
            'wt',
            encoding='utf8')
        fpointerOutTrain.writelines(listTrainset)
        fpointerOutTrain.close()
        fpointerOutTest = open(
            os.path.splitext(paramFpathOutTest)[0] + '.txt',
            'wt',
            encoding='utf8')
        fpointerOutTest.writelines(listTestset)
        fpointerOutTest.close()

        # release memory, Note that in pandas you will have to use fflush
        listTrainset = None
        listTestset = None
        listReviews = None

    def yelpInstanceConstructFromTrain(
            self,
            paramFpathInTrainTxt,
            paramFpathOutToken2IndexDict,
            paramFpathOutIndex2TokenDict,
            paramFpathOutTrainParams,
            paramFpathOutTrainInstance):
        '''
        combine reviews with stars, reshuffle reviews, and split into two sets
        ===================================================
        parameters:
        -----------
        paramFpathInTrainTxt: review texted train
        paramFpathOutToken2IndexDict: map token to index
        paramFpathOutIndex2TokenDict: map index to token
        paramFpathOutTest: test se
        paramFpathOutParams: the parameters needed for training
        paramTrainsetPercent: train set percent

        return:
        -----------
        None
        '''

        # read in the train.txt
        fpointerInTrainTxt = open(paramFpathInTrainTxt, 'rt', encoding='utf8')

        def __function4map(elem4map):
            '''
            stripe elem
            ===================================================
            parameters:
            -----------
            elem4map

            return:
            -----------
            mapped elem
            '''
            elemstriped = elem4map.strip()
            return elemstriped

        listTrainTxt = list(map(__function4map,
                                fpointerInTrainTxt.readlines()))
        fpointerInTrainTxt.close()

        # ----------initialize TextPreProcessor
        text_processor = TextPreProcessor(
            normailze=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time', 'date', 'number'],
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                      "emphasis", "censored"},
            fix_html=True,
            segmenter="english",
            corrector="english",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,

            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            dicts=[emoticons]
        )
        # ----------Initialize TextPreProcessor

        listTrainTxtTokenized = \
            list(text_processor.pre_process_docs(listTrainTxt))
        listTrainTxt = None
        # ----------save the vocabulary table,
        #           calculate and save the parameters
        # filter top 20,000 tokens
        dictVocabulary2Freq = dict()
        for listTokens in listTrainTxtTokenized:
            for aToken in listTokens:
                if aToken in dictVocabulary2Freq:
                    dictVocabulary2Freq[aToken] += 1
                else:
                    dictVocabulary2Freq[aToken] = 1
        itemgetter1 = operator.itemgetter(1)
        list_k_v_top_20000 = sorted(dictVocabulary2Freq.items(),
                                    key=itemgetter1,
                                    reverse=True)[0:20000]
        dict_k_v_top_20000 = {k: v for k, v in list_k_v_top_20000}
        dictVocabulary2Freq = None
        list_k_v_top_20000 = None

        # calculate maxDocumentSize and vocabularySize
        maxDocumentSize = 0
        vocabularySize = 0

        dictVocabulary2Index = dict()
        dictIndex2Vocabulary = dict()
        tokenCurrentIndex = 0
        for listTokens in listTrainTxtTokenized:
            if maxDocumentSize < len(listTokens):
                maxDocumentSize = len(listTokens)
            for aToken in listTokens:
                # filter rare words, reduce vocabulary size
                if aToken not in dict_k_v_top_20000:
                    continue
                if aToken in dictVocabulary2Index:
                    pass
                else:
                    dictVocabulary2Index[aToken] = tokenCurrentIndex
                    dictIndex2Vocabulary[tokenCurrentIndex] = aToken
                    tokenCurrentIndex += 1
        vocabularySize = tokenCurrentIndex
        assert vocabularySize == len(dictVocabulary2Index)

        # trim doc_size to 0.5 maxDocSize
        # trimmed_doc_size = maxDocumentSize * 0.5

        # json write using the fp4jsonoutput = open(,'wt', encoding='utf8')
        fp4jsonoutput = open(paramFpathOutToken2IndexDict,
                             'wt', encoding='utf8')
        json.dump(dictVocabulary2Index, fp4jsonoutput, ensure_ascii=False)
        fp4jsonoutput.close()

        fp4jsonoutput = open(paramFpathOutIndex2TokenDict,
                             'wt', encoding='utf8')
        json.dump(dictIndex2Vocabulary, fp4jsonoutput, ensure_ascii=False)
        fp4jsonoutput.close()

        # dictVocabulary2Index = None
        dictIndex2Vocabulary = None

        fpointerOutParams = open(
            paramFpathOutTrainParams,
            'wt',
            encoding='utf8')

        str4write = 'TrainingInstances: %d\n' % len(listTrainTxtTokenized)\
            + 'DocumentSeqLen: %d\n' % maxDocumentSize\
            + 'VocabularySize: %d\n' % vocabularySize

        fpointerOutParams.write(str4write)

        fpointerOutParams.close()
        # ----------calculate and save the parameters

        # ----------construct training instances and perform padding
        print('Hello1')

        def __function_tokenlist_to_traininstance(
                tokenlist):
            '''
            from tokenlist to padded instance list
            adding subsampling
            '''
            tokenlist_size = len(tokenlist)
            traininginstance = list()
            for n in range(tokenlist_size):
                # ----------split tokenlist section
                tokenlist_section = None
                if n - HALF_WINDOW_SIZE < 0:
                    if n + HALF_WINDOW_SIZE >= tokenlist_size:
                        tokenlist_section = tokenlist
                    else:
                        tokenlist_section = tokenlist[:n + HALF_WINDOW_SIZE]
                else:
                    if n + HALF_WINDOW_SIZE >= tokenlist_size:
                        tokenlist_section = tokenlist[n - HALF_WINDOW_SIZE:]
                    else:
                        tokenlist_section = tokenlist[n - HALF_WINDOW_SIZE:
                                                      n + HALF_WINDOW_SIZE]
                # ----------calculate tokenlist multiterm
                countlist_vocab = [0 for i in range(vocabularySize)]
                countlist_vocab[dictVocabulary2Index[tokenlist[n]]] += 1
                traininginstance.append(countlist_vocab)
                countlist_vocab = [0 for i in range(vocabularySize)]
                for atoken in tokenlist_section:
                    countlist_vocab[dictVocabulary2Index[atoken]] += 1
                traininginstance.append(countlist_vocab)

            # ----------padding
            for n in range(tokenlist_size, maxDocumentSize):
                fullzero_vocab = [0 for i in range(vocabularySize)]
                traininginstance.append(fullzero_vocab)
                fullzero_vocab = [0 for i in range(vocabularySize)]
                traininginstance.append(fullzero_vocab)

            return traininginstance

        def __function_traininstance_to_string(
                traininstance):
            '''
            from traininstance to a string
            '''
            str_training_instance = ''
            for acountlist_vocab in traininstance:
                acountlist_vocab = list(map(str, acountlist_vocab))
                str_acountlist_vocab = ' '.join(acountlist_vocab)
                str_training_instance += ' ' + str_acountlist_vocab

            str_training_instance += '\n'
            return str_training_instance

        fpointerOutTrainInstance = open(
            paramFpathOutTrainInstance,
            'wt',
            encoding='utf8')
        for aTrainTxtTokenized in listTrainTxtTokenized:
            aTrainInstance = __function_tokenlist_to_traininstance(
                aTrainTxtTokenized)
            aStrTrainInstance = __function_traininstance_to_string(
                aTrainInstance)
            fpointerOutTrainInstance.write(aStrTrainInstance)
        fpointerOutTrainInstance.close()

        return None


if __name__ == '__main__':

    oYelpPreprocessor = YelpPreprocessor()
    # oYelpPreprocessor.json_preprocess_all(
    #     paramFpathInReview='../datasets/review.json',
    #     paramFpathOutReview='../datasets/review.txt')
    # oYelpPreprocessor.yelpTrainAndTestConstructFromWhole(
    #     paramFpathInReview='../datasets/review.txt',
    #     paramFpathOutTrain='../datasets/train.txt',
    #     paramFpathOutTest='../datasets/test.txt',
    #     paramTrainsetPercent=0.001)
    oYelpPreprocessor.yelpInstanceConstructFromTrain(
        paramFpathInTrainTxt='../datasets/train.txt',
        paramFpathOutToken2IndexDict='../datasets/train_voca_to_index.txt',
        paramFpathOutIndex2TokenDict='../datasets/train_index_to_voca.txt',
        paramFpathOutTrainParams='../datasets/train_params.txt',
        paramFpathOutTrainInstance='../datasets/train_instances.csv')
    print('Hello2')
