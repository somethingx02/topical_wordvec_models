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
import file_handling as fh


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

    def json_preprocess_5categories(self,
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

        dictRestaurantBusinessCategories = {
            "Restaurant":
                {"includeif": {"Restaurants"},
                 "excludeif": set()},
        }

        dictShoppingBusinessCategories = {
            "Shopping center":
                {"includeif": {"Shopping"},
                 "excludeif": set()},
        }

        dictBeautyBusinessCategories = {
            "Cosmetic":
                {"includeif": {"Cosmetics & Beauty Supply",
                               "Beauty & Spas",
                               "Plastic Surgeons"},
                 "excludeif": set()},
        }

        dictAutomobileBusinessCategories = {
            "Automobile":
                {"includeif": {"Automotive"},
                 "excludeif": set()},
        }

        set5BussinessIds = set()
        fpointerInBusiness = open(paramFpathInBussiness, 'rt', encoding='utf8')
        for aline in fpointerInBusiness:
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
                        set5BussinessIds.add(bussinessId)
                        break
            # reduce resaurant review size
            float_rand = random.random()
            if float_rand < 0.0049:

                for akey in dictRestaurantBusinessCategories:

                    includedifset = \
                        dictRestaurantBusinessCategories[akey]['includeif']
                    excludedifset = \
                        dictRestaurantBusinessCategories[akey]['excludeif']
                    if len(includedifset.intersection(
                            setBussinessCategory)) != 0:
                        if len(excludedifset.intersection(
                                setBussinessCategory)) != 0:
                            pass
                        else:
                            bussinessId = jo['business_id']
                            set5BussinessIds.add(bussinessId)
                            break
            float_rand = random.random()
            if float_rand < 0.0409:
                for akey in dictShoppingBusinessCategories:
                    includedifset = \
                        dictShoppingBusinessCategories[akey]['includeif']
                    excludedifset = \
                        dictShoppingBusinessCategories[akey]['excludeif']
                    if len(includedifset.intersection(
                            setBussinessCategory)) != 0:
                        if len(excludedifset.intersection(
                                setBussinessCategory)) != 0:
                            pass
                        else:
                            bussinessId = jo['business_id']
                            set5BussinessIds.add(bussinessId)
                            break
            float_rand = random.random()
            if float_rand < 0.0494:
                for akey in dictBeautyBusinessCategories:
                    includedifset = \
                        dictBeautyBusinessCategories[akey]['includeif']
                    excludedifset = \
                        dictBeautyBusinessCategories[akey]['excludeif']
                    if len(includedifset.intersection(
                            setBussinessCategory)) != 0:
                        if len(excludedifset.intersection(
                                setBussinessCategory)) != 0:
                            pass
                        else:
                            bussinessId = jo['business_id']
                            set5BussinessIds.add(bussinessId)
                            break
            float_rand = random.random()
            if float_rand < 0.0824:
                for akey in dictAutomobileBusinessCategories:
                    includedifset = \
                        dictAutomobileBusinessCategories[akey]['includeif']
                    excludedifset = \
                        dictAutomobileBusinessCategories[akey]['excludeif']
                    if len(includedifset.intersection(
                            setBussinessCategory)) != 0:
                        if len(excludedifset.intersection(
                                setBussinessCategory)) != 0:
                            pass
                        else:
                            bussinessId = jo['business_id']
                            set5BussinessIds.add(bussinessId)
                            break

        fpointerInBusiness.close()
        # print( setClinicalBussinessIds)

        fpointerInReview = open(paramFpathInReview, 'rt', encoding='utf8')
        fpointerOutReview = open(paramFpathOutReview, 'wt', encoding='utf8')
        fpointerOutStars = open(paramFpathOutStars, 'wt', encoding='utf8')
        for aline in fpointerInReview:
            aline = aline.strip()
            jo = json.loads(aline)

            bussinessId = jo['business_id']
            if bussinessId in set5BussinessIds:
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

    def json_preprocess_10categories_large(
            self,
            paramFpathInBussiness,
            paramFpathInReview,
            paramFpathOutReview,
            paramFpathOutStars):
        '''
        retrieve clinical documents from business, reviews
        except for clinical documents, other categories are balances
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

        dictRestaurantBusinessCategories = {
            "Restaurant":
                {"includeif": {"Restaurants"},
                 "excludeif": set()},
        }

        dictShoppingBusinessCategories = {
            "Shopping center":
                {"includeif": {"Shopping"},
                 "excludeif": set()},
        }

        dictBeautyBusinessCategories = {
            "Cosmetic":
                {"includeif": {"Cosmetics & Beauty Supply",
                               "Beauty & Spas",
                               "Plastic Surgeons"},
                 "excludeif": set()},
        }

        dictAutomobileBusinessCategories = {
            "Automobile":
                {"includeif": {"Automotive"},
                 "excludeif": set()},
        }

        set5BussinessIds = set()
        fpointerInBusiness = open(paramFpathInBussiness, 'rt', encoding='utf8')
        for aline in fpointerInBusiness:
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
                        set5BussinessIds.add(bussinessId)
                        break
            # reduce resaurant review size
            float_rand = random.random()
            if float_rand < 0.0049:

                for akey in dictRestaurantBusinessCategories:

                    includedifset = \
                        dictRestaurantBusinessCategories[akey]['includeif']
                    excludedifset = \
                        dictRestaurantBusinessCategories[akey]['excludeif']
                    if len(includedifset.intersection(
                            setBussinessCategory)) != 0:
                        if len(excludedifset.intersection(
                                setBussinessCategory)) != 0:
                            pass
                        else:
                            bussinessId = jo['business_id']
                            set5BussinessIds.add(bussinessId)
                            break
            float_rand = random.random()
            if float_rand < 0.0409:
                for akey in dictShoppingBusinessCategories:
                    includedifset = \
                        dictShoppingBusinessCategories[akey]['includeif']
                    excludedifset = \
                        dictShoppingBusinessCategories[akey]['excludeif']
                    if len(includedifset.intersection(
                            setBussinessCategory)) != 0:
                        if len(excludedifset.intersection(
                                setBussinessCategory)) != 0:
                            pass
                        else:
                            bussinessId = jo['business_id']
                            set5BussinessIds.add(bussinessId)
                            break
            float_rand = random.random()
            if float_rand < 0.0494:
                for akey in dictBeautyBusinessCategories:
                    includedifset = \
                        dictBeautyBusinessCategories[akey]['includeif']
                    excludedifset = \
                        dictBeautyBusinessCategories[akey]['excludeif']
                    if len(includedifset.intersection(
                            setBussinessCategory)) != 0:
                        if len(excludedifset.intersection(
                                setBussinessCategory)) != 0:
                            pass
                        else:
                            bussinessId = jo['business_id']
                            set5BussinessIds.add(bussinessId)
                            break
            float_rand = random.random()
            if float_rand < 0.0824:
                for akey in dictAutomobileBusinessCategories:
                    includedifset = \
                        dictAutomobileBusinessCategories[akey]['includeif']
                    excludedifset = \
                        dictAutomobileBusinessCategories[akey]['excludeif']
                    if len(includedifset.intersection(
                            setBussinessCategory)) != 0:
                        if len(excludedifset.intersection(
                                setBussinessCategory)) != 0:
                            pass
                        else:
                            bussinessId = jo['business_id']
                            set5BussinessIds.add(bussinessId)
                            break

        fpointerInBusiness.close()
        # print( setClinicalBussinessIds)

        fpointerInReview = open(paramFpathInReview, 'rt', encoding='utf8')
        fpointerOutReview = open(paramFpathOutReview, 'wt', encoding='utf8')
        fpointerOutStars = open(paramFpathOutStars, 'wt', encoding='utf8')
        for aline in fpointerInReview:
            aline = aline.strip()
            jo = json.loads(aline)

            bussinessId = jo['business_id']
            if bussinessId in set5BussinessIds:
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
        # ----------save the vocabulary table
        # filter stop words
        mallet_stopwords = None
        print("Using MALLET stopwords")
        mallet_stopwords = fh.read_text('mallet_stopwords.txt')
        mallet_stopwords = {s.strip() for s in mallet_stopwords}
        listTrainTxtStopworded = list()
        document_num = 0
        for listTokens in listTrainTxtTokenized:
            document_num += 1
            listTokens = [aToken for aToken in listTokens
                          if aToken not in mallet_stopwords]
            if len(listTokens) == 0:
                print('dstopword document: %d abandoned' % document_num)
            else:
                listTrainTxtStopworded.append(listTokens)

        listTrainTxtTokenized = None

        # filter top 4,000 tokens, following BSG
        dictVocabulary2Freq = dict()
        for listTokens in listTrainTxtStopworded:
            for aToken in listTokens:
                if aToken in dictVocabulary2Freq:
                    dictVocabulary2Freq[aToken] += 1
                else:
                    dictVocabulary2Freq[aToken] = 1

        itemgetter1 = operator.itemgetter(1)
        list_k_v_top_3000 = sorted(dictVocabulary2Freq.items(),
                                   key=itemgetter1,
                                   reverse=True)[0:3000]
        dict_k_v_top_3000 = {k: v for k, v in list_k_v_top_3000}
        dictVocabulary2Freq = None
        list_k_v_top_3000 = None

        listTrainTxtFreqworded = list()
        document_num = 0
        for listTokens in listTrainTxtStopworded:
            document_num += 1
            listTokens = [aToken for aToken in listTokens
                          if aToken in dict_k_v_top_3000]
            if len(listTokens) == 0:
                print('dlowfreq document: %d abandoned' % document_num)
            else:
                listTrainTxtFreqworded.append(listTokens)

        listTrainTxtStopworded = None

        # calculate maxDocumentSize and vocabularySize
        maxDocumentSize = 0
        vocabularySize = 0

        dictVocabulary2Index = dict()
        dictIndex2Vocabulary = dict()
        tokenCurrentIndex = 0
        for listTokens in listTrainTxtFreqworded:
            if maxDocumentSize < len(listTokens):
                maxDocumentSize = len(listTokens)
            for aToken in listTokens:
                # filter rare words, reduce vocabulary size
                if aToken not in dict_k_v_top_3000:
                    continue
                if aToken in dictVocabulary2Index:
                    pass
                else:
                    dictVocabulary2Index[aToken] = tokenCurrentIndex
                    dictIndex2Vocabulary[tokenCurrentIndex] = aToken
                    tokenCurrentIndex += 1
        vocabularySize = tokenCurrentIndex
        assert vocabularySize == len(dictVocabulary2Index)

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

        # ----------construct training instances and perform padding
        print('construct training instances')

        def __function_tokenlist_to_traininstance(
                tokenlist):
            '''
            from tokenlist to a list of window list
            adding subsampling
            ====================
            params:
            ----------
            a tokenlist

            return:
            ----------
            a list of windows/instances
            '''
            tokenlist_size = len(tokenlist)
            traininginstances = list()
            for n in range(tokenlist_size):
                # if word not frequent, then skip
                if tokenlist[n] not in dictVocabulary2Index:
                    print('Error: tokenlist[n] not in dictVocabulary2Index')
                    continue
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
                countlist_pivot = [dictVocabulary2Index[tokenlist[n]]]
                # countlist_pivot = [0 for i in range(vocabularySize)]
                # countlist_pivot[dictVocabulary2Index[tokenlist[n]]] += 1

                countlist_context = [0 for i in range(vocabularySize)]
                for atoken in tokenlist_section:
                    if atoken not in dictVocabulary2Index:
                        continue
                    if atoken == tokenlist[n]:
                        pass
                    else:
                        countlist_context[dictVocabulary2Index[atoken]] += 1
                countlist_pivot.extend(countlist_context)
                traininginstances.append(countlist_pivot)

            return traininginstances

        def __function_traininstance_to_string(
                traininstance):
            '''
            from traininstance to a string
            '''
            traininstance = list(map(str, traininstance))
            str_training_instance = ' '.join(traininstance)
            str_training_instance += '\n'
            return str_training_instance

        fpointerOutTrainInstance = open(
            paramFpathOutTrainInstance,
            'wt',
            encoding='utf8')

        traininstance_count = 0
        # list_traininstances = list()
        for aTrainTxtTokenized in listTrainTxtFreqworded:
            trainInstances = __function_tokenlist_to_traininstance(
                aTrainTxtTokenized)

            traininstance_count += len(trainInstances)
            # list_traininstances.extend(trainInstances)
            for aTrainInstance in trainInstances:
                aStrTrainInstance = __function_traininstance_to_string(
                    aTrainInstance)
                fpointerOutTrainInstance.write(aStrTrainInstance)
        fpointerOutTrainInstance.close()

        # ----------save the parameters
        fpointerOutParams = open(
            paramFpathOutTrainParams,
            'wt',
            encoding='utf8')

        str4write = 'DocumentCount: %d\n' % len(listTrainTxtFreqworded)\
            + 'DocumentSeqLen: %d\n' % maxDocumentSize\
            + 'TrainInstances: %d\n' % traininstance_count\
            + 'VocabularySize: %d\n' % vocabularySize

        fpointerOutParams.write(str4write)

        fpointerOutParams.close()

        return None

    def yelpSparseInstanceConstructFromTrain(
            self,
            paramFpathInTrainTxt,
            paramVocabularySize,
            paramFpathOutToken2IndexDict,
            paramFpathOutIndex2TokenDict,
            paramFpathOutTrainParams,
            paramFpathOutTrainInstance):
        '''
        combine reviews with stars, reshuffle reviews, and split into two sets
        use sparse representation
        ===================================================
        parameters:
        -----------
        paramFpathInTrainTxt: review texted train
        paramVocabularySize: vocabularysize
        paramFpathOutToken2IndexDict: map token to index
        paramFpathOutIndex2TokenDict: map index to token
        paramFpathOutParams: the parameters needed for training
        paramFpathOutTrainInstance: train instances csv

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
        # ----------save the vocabulary table
        # filter stop words
        mallet_stopwords = None
        print("Using MALLET stopwords")
        mallet_stopwords = fh.read_text('mallet_stopwords.txt')
        mallet_stopwords = {s.strip() for s in mallet_stopwords}
        listTrainTxtStopworded = list()
        document_num = 0
        for listTokens in listTrainTxtTokenized:
            document_num += 1
            listTokens = [aToken for aToken in listTokens
                          if aToken not in mallet_stopwords]
            if len(listTokens) == 0:
                print('dstopword document: %d abandoned' % document_num)
            else:
                listTrainTxtStopworded.append(listTokens)

        listTrainTxtTokenized = None

        # filter top 4,000 tokens, following BSG
        dictVocabulary2Freq = dict()
        for listTokens in listTrainTxtStopworded:
            for aToken in listTokens:
                if aToken in dictVocabulary2Freq:
                    dictVocabulary2Freq[aToken] += 1
                else:
                    dictVocabulary2Freq[aToken] = 1

        itemgetter1 = operator.itemgetter(1)
        list_k_v_top_8000 = sorted(dictVocabulary2Freq.items(),
                                   key=itemgetter1,
                                   reverse=True)[0:paramVocabularySize]
        dict_k_v_top_8000 = {k: v for k, v in list_k_v_top_8000}
        dictVocabulary2Freq = None
        list_k_v_top_8000 = None

        listTrainTxtFreqworded = list()
        document_num = 0
        for listTokens in listTrainTxtStopworded:
            document_num += 1
            listTokens = [aToken for aToken in listTokens
                          if aToken in dict_k_v_top_8000]
            if len(listTokens) == 0:
                print('dlowfreq document: %d abandoned' % document_num)
            else:
                listTrainTxtFreqworded.append(listTokens)

        listTrainTxtStopworded = None

        # calculate maxDocumentSize and vocabularySize
        maxDocumentSize = 0
        vocabularySize = 0

        dictVocabulary2Index = dict()
        dictIndex2Vocabulary = dict()
        # tokenCurrentIndex = 0
        tokenCurrentIndex = 1
        for listTokens in listTrainTxtFreqworded:
            if maxDocumentSize < len(listTokens):
                maxDocumentSize = len(listTokens)
            for aToken in listTokens:
                # filter rare words, reduce vocabulary size
                if aToken not in dict_k_v_top_8000:
                    continue
                if aToken in dictVocabulary2Index:
                    pass
                else:
                    dictVocabulary2Index[aToken] = tokenCurrentIndex
                    dictIndex2Vocabulary[tokenCurrentIndex] = aToken
                    tokenCurrentIndex += 1
        dictVocabulary2Index['mynan'] = 0
        dictIndex2Vocabulary[0] = 'mynan'
        vocabularySize = tokenCurrentIndex
        assert vocabularySize == len(dictVocabulary2Index)

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

        # ----------construct training instances and perform padding
        print('construct training instances')

        def __function_tokenlist_to_traininstance(
                tokenlist):
            '''
            from tokenlist to a list of window list
            adding subsampling
            ====================
            params:
            ----------
            a tokenlist

            return:
            ----------
            a list of windows/instances
            '''
            tokenlist_size = len(tokenlist)
            traininginstances = list()
            for n in range(tokenlist_size):
                # if word not frequent, then skip
                if tokenlist[n] not in dictVocabulary2Index:
                    print('Error: tokenlist[n] not in dictVocabulary2Index')
                    continue
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
                countlist_pivot = [dictVocabulary2Index[tokenlist[n]]]
                # countlist_pivot = [0 for i in range(vocabularySize)]
                # countlist_pivot[dictVocabulary2Index[tokenlist[n]]] += 1

                # sparse representation of the window size,
                # each slot represent an index
                countlist_context = [0 for i in range(2 * HALF_WINDOW_SIZE)]
                for c, atoken in enumerate(tokenlist_section):
                    if atoken not in dictVocabulary2Index:
                        print((
                            'Error: tokenlist[n] not '
                            'in dictVocabulary2Index'))
                        continue
                    if atoken == tokenlist[n]:
                        # the pivot word is excluded as the highest freq word
                        # but when computing contextualized word embedding or
                        # doing the word substitution, you should manually add
                        # the pivot word under the BOW rep
                        countlist_context[c] = 0
                    else:
                        countlist_context[c] = dictVocabulary2Index[atoken]
                countlist_pivot.extend(countlist_context)
                traininginstances.append(countlist_pivot)

            return traininginstances

        def __function_traininstance_to_string(
                traininstance):
            '''
            from traininstance to a string
            '''
            traininstance = list(map(str, traininstance))
            str_training_instance = ' '.join(traininstance)
            str_training_instance += '\n'
            return str_training_instance

        fpointerOutTrainInstance = open(
            paramFpathOutTrainInstance,
            'wt',
            encoding='utf8')

        traininstance_count = 0
        # list_traininstances = list()
        for aTrainTxtTokenized in listTrainTxtFreqworded:
            trainInstances = __function_tokenlist_to_traininstance(
                aTrainTxtTokenized)

            traininstance_count += len(trainInstances)
            # list_traininstances.extend(trainInstances)
            for aTrainInstance in trainInstances:
                aStrTrainInstance = __function_traininstance_to_string(
                    aTrainInstance)
                fpointerOutTrainInstance.write(aStrTrainInstance)
        fpointerOutTrainInstance.close()

        # ----------save the parameters
        fpointerOutParams = open(
            paramFpathOutTrainParams,
            'wt',
            encoding='utf8')

        str4write = 'DocumentCount: %d\n' % len(listTrainTxtFreqworded)\
            + 'DocumentSeqLen: %d\n' % maxDocumentSize\
            + 'TrainInstances: %d\n' % traininstance_count\
            + 'VocabularySize: %d\n' % vocabularySize

        fpointerOutParams.write(str4write)

        fpointerOutParams.close()

        return None

    # ==================== intended for after dataset construction

    def yelpDoclist2Parsedlist(
            self,
            paramDocList,
            paramFpathInVocabulary2Index):
        '''
        convert a Doclist to tokenized, vocabularized list
        ====================
        '''

        # ----------tokenization
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

        # ----------Load the vocabulary dict
        listDocTokenized = \
            list(text_processor.pre_process_docs(paramDocList))
        paramDocList = None

        fpointerInVocabulary2Index = open(
            paramFpathInVocabulary2Index,
            'rt',
            encoding='utf8')
        dictVocabulary2Index = \
            json.load(fpointerInVocabulary2Index)
        fpointerInVocabulary2Index.close()

        # ----------filter in vocabulary words
        def __function_vocabularize(
                aTokenizedDoc):
            '''
            filter out unvocabularized words
            '''
            filteredDoc = [aToken for aToken in aTokenizedDoc
                           if aToken in dictVocabulary2Index]
            return filteredDoc

        listDocVocabularized = list(
            map(__function_vocabularize, listDocTokenized))
        # print(listDocVocabularized)

        # ----------construct pivot word list, xn, wc
        def __function_pivot_xn_wc(
                aVocabularizedDoc):
            '''
            transfer a doc into several instances,
            each instance contains a pivot word and xn, wc
            '''
            tokenlist_size = len(aVocabularizedDoc)
            pivotinstances = list()
            vocabularySize = len(dictVocabulary2Index)
            for n in range(tokenlist_size):
                # ----------split aVocabularizedDoc section
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
                countlist_pivot = [dictVocabulary2Index[aVocabularizedDoc[n]]]

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
                    aVocabularizedDoc[n],
                    countlist_pivot,
                    countlist_context))

            return pivotinstances
        listPivotXnWcInstances = list()
        for aDocVocabularized in listDocVocabularized:
            pivotXnWcInstances = __function_pivot_xn_wc(
                aDocVocabularized)
            listPivotXnWcInstances.extend(pivotXnWcInstances)
        return listPivotXnWcInstances

    def yelpTrainTxt2Vocabularized(
            self,
            paramFpathInTrainInstances,
            paramFpathInVocabulary2Index,
            paramFpathOutVocabularized):
        '''
        convert TrainInstances to Vocabularized instances
        ====================
        '''

        # ----------tokenization
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

        fpointerInVocabularized = open(
            paramFpathInTrainInstances,
            'rt', encoding='utf8')
        docList = list(map(
            str.strip,
            fpointerInVocabularized.readlines()))
        fpointerInVocabularized.close()

        # ----------Load the vocabulary dict
        listDocTokenized = \
            list(text_processor.pre_process_docs(docList))
        docList = None

        fpointerInVocabulary2Index = open(
            paramFpathInVocabulary2Index,
            'rt',
            encoding='utf8')
        dictVocabulary2Index = \
            json.load(fpointerInVocabulary2Index)
        fpointerInVocabulary2Index.close()
        print(len(dictVocabulary2Index))

        # ----------filter in vocabulary words
        def __function_vocabularize(
                aTokenizedDoc):
            '''
            filter out unvocabularized words
            '''
            filteredDoc = [aToken for aToken in aTokenizedDoc
                           if aToken in dictVocabulary2Index]
            return filteredDoc

        listDocVocabularized = list(
            map(__function_vocabularize, listDocTokenized))

        fpointerOutVocabularized = open(
            paramFpathOutVocabularized,
            'wt',
            encoding='utf8')
        for aDocVocabularized in listDocVocabularized:
            if len(aDocVocabularized) == 0:
                continue
            strDocVocabularized = ' '.join(aDocVocabularized)
            strDocVocabularized += '\n'
            fpointerOutVocabularized.write(strDocVocabularized)

        fpointerOutVocabularized.close()


if __name__ == '__main__':

    oYelpPreprocessor = YelpPreprocessor()
    # oYelpPreprocessor.json_preprocess_all(
    #     paramFpathInReview='../datasets/review.json',
    #     paramFpathOutReview='../datasets/review.txt')

    # oYelpPreprocessor.json_preprocess_5categories(
    #     paramFpathInBussiness='../datasets/business.json',
    #     paramFpathInReview='../datasets/review.json',
    #     paramFpathOutReview='../datasets/review.txt',
    #     paramFpathOutStars='../datasets/stars.txt')

    # oYelpPreprocessor.yelpTrainAndTestConstructFromWhole(
    #     paramFpathInReview='../datasets/review.txt',
    #     paramFpathOutTrain='../datasets/train.txt',
    #     paramFpathOutTest='../datasets/test.txt',
    #     paramTrainsetPercent=1.0)

    # # ----------usually call this preventing reshuffling the data
    # oYelpPreprocessor.yelpInstanceConstructFromTrain(
    #     paramFpathInTrainTxt='../datasets/train.txt',
    #     paramFpathOutToken2IndexDict='../datasets/train_voca_to_index.txt',
    #     paramFpathOutIndex2TokenDict='../datasets/train_index_to_voca.txt',
    #     paramFpathOutTrainParams='../datasets/train_params.txt',
    #     paramFpathOutTrainInstance='../datasets/train_instances.csv')

    # ----------usually call this preventing reshuffling the data
    oYelpPreprocessor.yelpSparseInstanceConstructFromTrain(
        paramFpathInTrainTxt='../datasets/train.txt',
        paramVocabularySize=8000,
        paramFpathOutToken2IndexDict='../datasets/train_voca_to_index.txt',
        paramFpathOutIndex2TokenDict='../datasets/train_index_to_voca.txt',
        paramFpathOutTrainParams='../datasets/train_params.txt',
        paramFpathOutTrainInstance='../datasets/train_sparseinstances.csv')

    # print('finished')

    # ======================Test Centre
    # test for traininstance vectorize
    # oYelpPreprocessor.yelpTrainTxt2Vocabularized(
    #     paramFpathInTrainInstances='../datasets/train.txt',
    #     paramFpathInVocabulary2Index='../datasets/train_voca_to_index.txt',
    #     paramFpathOutVocabularized=(
    #         '../datasets/train'
    #         '_vocabularized.txt'))

    # test for doclist
    # doc_list = [
    #     'If any one in the UK is thinking of \
    #     cosmetic surgery abroad this doctor and clinic \
    #     I would most certainly recomment.',
    #     'Online reviews of surgeons who perform \
    #     plastic surgery may be unreliable, researchers say.',
    #     'Here are 4 ways to reduce the plastic bags.',
    #     'Our colored paper bags help products get \
    #     noticed and improve presentation.',
    #     'Claiming compensation due to a problem \
    #     with a road or street.',
    #     'Coronary artery disease (CAD) is \
    #     the most common type of heart disease.',
    #     'Clarence street was the artery of \
    #     the central business district.']
    # parsed_list = oYelpPreprocessor.yelpDoclist2Parsedlist(
    #     paramDocList=doc_list,
    #     paramFpathInVocabulary2Index='../datasets/train_voca_to_index.txt')
    # (pivot_list, xn_list, wc_list) = zip(*parsed_list)
    # print(pivot_list)
