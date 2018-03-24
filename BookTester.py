# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:58:37 2018

@author: MAREES
"""
import os
import pandas as pd
import numpy as np
import sys
import sklearn.metrics as skmets

from keras.models import model_from_json
from keras.optimizers import Adam, SGD



class BookTester():
    
    def __init__(self, model_weights, model_json, test_user_char, test_user_book):
        
        
        # check referenced files are present in current working directory
        if os.path.exists(model_weights) \
            and os.path.exists(model_json) \
            and os.path.exists(test_user_book) \
            and os.path.exists(test_user_char):
                
            print("Input files all present in current working dir")
            
        else:
            # In full programme this would be captured in log.error along with exception instead of print 
            print("Error: files not found. Please ensure csv files are in current directory")
            exit()
            
        # Seeding for reproducability
        np.random.seed(8)   
        
    
    def loadModel(self, model_weights, model_json):
        # load json and create model
        json_file = open(model_json, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_weights)
        print("Loaded model from disk")
        
        return loaded_model
    

    
    
    
    def getTop10(self, model, test_user_books, test_user_chars):
        
        # Read csv and extract required values
        test_user_chars = pd.read_csv(test_user_chars)
        TestX = test_user_chars.iloc[:,1:11].values
        
        # Get user_ids in test dataset
        Test_Users = test_user_chars[test_user_chars.columns[0]]
        # get book_ids
        user_books = pd.read_csv(test_user_books)
        book_headings = user_books.iloc[:,1:1001].columns.values
        
        # Make predictions using reloaded tensorflow model
        preds = loaded_model.predict(TestX)
        print("Model predictions calculated")
    
        # make dataframe with prediciton values
        predictions = pd.DataFrame(preds, columns=book_headings, index=Test_Users)
       
        # Get the column index of the top 10 model predictions and store them in an array
        preds_sorted = np.argsort(predictions,axis=1).iloc[:,-10:].values
        # Use locations in the array to select the book_ids from the header
        books = book_headings[preds_sorted]
        # Define header for new dataframe
        ranks = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
        # make top10 dataframe
        top10 = pd.DataFrame(books, columns=ranks, index=Test_Users)
        # save top10 to csv
        top10.to_csv("BookRecommendationResults.csv")
        print("Top 10 Book recommendations .CSV printed to current working directory")
    
        return preds
    
    
    def performance_check(self, model, model_predictions, actual_predictions):
               
        # Read csv of actual predictions and extract binary values
        user_books = pd.read_csv(actual_predictions)
        act_preds = user_books.iloc[:,1:1001].values
        
        # Round predictions to book Selected/Not-selected indicator
        model_predictions[model_predictions>=0.5] = 1
        model_predictions[model_predictions<0.5] = 0
        
        # Use Hamming_loss to evaluate multilabel classifier
        # hamming loss  is the fraction of labels that are incorrectly predicted 
        hamming_loss = skmets.hamming_loss(act_preds, model_predictions)
        
        print('Hamming Loss: ', hamming_loss)
    
        
        
if __name__ == "__main__":      
    
    # Get command line arguments
    model_weights = sys.argv[1]
    model_details = sys.argv[2]
    prediction_inputs = sys.argv[3] # = test_user_char.csv
    actual_predictions = sys.argv[4] # = test_user_book.csv
    
    # Instantiate BookTester
    BTest = BookTester(model_weights, model_details, prediction_inputs, actual_predictions)
    # Load trained tensorflow model
    loaded_model = BTest.loadModel(model_weights, model_details)
    # Write csv of top 10 book recommendations
    preds = BTest.getTop10(loaded_model, actual_predictions, prediction_inputs)
    # Check Prediction Performance 
    BTest.performance_check(loaded_model, preds, actual_predictions)
    
