# -*- coding: utf8 -*-

import yelpPreprocessor


if __name__ == '__main__':

    oYelpPreprocessor = yelpPreprocessor.YelpPreprocessor()

    ##==========Preprocess data from Yelp Json Whole File. ( separate 
    ##          into train.csv and test.csv )
    # oYelpPreprocessor.jsonPreprocess(
    #     paramFpathInBussiness = 'Datasets/YelpReviewJsonWhole/business.json',
    #     paramFpathInReview = 'Datasets/YelpReviewJsonWhole/review.json',
    #     paramFpathOutReview = 'Outputs/YelpClinicalReview/clinical_reviews_texted.txt',
    #     paramFpathOutStars = 'Outputs/YelpClinicalReview/clinical_reviews_stars.txt')

    # oYelpPreprocessor.yelpTrainAndTestConstructFromWhole(
    #     paramFpathInReview = 'Outputs/YelpClinicalReview/clinical_reviews_texted.txt',
    #     paramFpathInStars ='Outputs/YelpClinicalReview/clinical_reviews_stars.txt',
    #     paramFpathOutTrain ='Outputs/YelpClinicalReview/clinical_reviews_train.csv',
    #     paramFpathOutTest = 'Outputs/YelpClinicalReview/clinical_reviews_test.csv',
    #     paramFpathOutParams = 'Outputs/YelpClinicalReview/clinical_reviews_params.txt',
    #     paramTrainsetSize = 10000)

    ##==========Preprocess data from texted Whole File and label whole file. ( separate 
    ##          into train.csv and test.csv )
    # oYelpPreprocessor.yelpTrainAndTestConstructFromWhole(
    #     paramFpathIn`Review = 'Datasets/YelpReviewTxtWhole/clinical_reviews_texted.txt',
    #     paramFpathInStars ='Datasets/YelpReviewTxtWhole/clinical_reviews_stars.txt',
    #     paramFpathOutTrain ='Outputs/YelpClinicalReview/clinical_reviews_train_2.csv',
    #     paramFpathOutTest = 'Outputs/YelpClinicalReview/clinical_reviews_test_2.csv',
    #     paramFpathOutParams = 'Outputs/YelpClinicalReview/clinical_reviews_params.txt',
    #     paramTrainsetSize = 10000)

    ##==========Preprocess data from separated texted train.txt and test.txt file.
    ##          the first column is the label column
    # oYelpPreprocessor.yelpTrainAndTestConstructFromTrainAndTest(
    #     paramFpathInTrainTxt = 'Datasets/YelpReviewTxtTrainAndTest/clinical_reviews_train.txt', 
    #     paramFpathInTestTxt = 'Datasets/YelpReviewTxtTrainAndTest/clinical_reviews_test.txt', 
    #     paramFpathOutTrain = 'Outputs/YelpClinicalReview/clinical_reviews_train.csv',
    #     paramFpathOutTest = 'Outputs/YelpClinicalReview/clinical_reviews_test.csv',
    #     paramFpathOutParams = 'Outputs/YelpClinicalReview/clinical_reviews_params.txt')

    ##==========Preprocess data from texted file, no label column
    oYelpPreprocessor.yelpTokenizeAndCharEncode(
        paramFpathInTxt = 'Datasets/YelpReviewTxtWhole/clinical_reviews_texted.txt',  
        paramFpathOut = 'Outputs/YelpClinicalReview/clinical_reviews_tokenized_charencoded.txt', 
        paramFpathOutParams = 'Outputs/YelpClinicalReview/clinical_reviews_tokenized_charencoded_params.txt')












