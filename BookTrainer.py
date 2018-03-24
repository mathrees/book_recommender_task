# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 16:57:35 2018

@author: MAREES
"""

import pandas as pd
import numpy as np
import os
import sys

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD

 


########################### Runtime Instructions ######################
###### >>> python BookTrainer.py {user_book.csv} {user_char.csv} ######
###### where:                                                    ######
###### user_book.csv = user book selections csv                  ######
###### and user_char.csv = character traits data csv             ######
#######################################################################

class BookTrainer():
    
    
    def __init__(self, user_books, user_char):
        # Initialise with relevent csv files as dataframes; 
        # user_books (string): a csv file containing binary data indicating user book selection
        # user_char (string): csv file containing user characteristics data
        
        
        # check referenced files are present in current working directory
        if os.path.exists(user_books) and os.path.exists(user_char):
            self.user_books = pd.read_csv(user_books)
            self.user_chars = pd.read_csv(user_char)
            print("Input CSVs read Successfully")
        else:
            # In full programme this would be captured in log.error along with exception instead of print 
            print("Error: files not found. Please ensure csv files are in current directory")
            exit()
            
        # Seeding for reproducability
        np.random.seed(8)     
    
                

    def makeDataSets(self, user_books, user_chars):
        # Split user_chars data into Training and Testing Sets
        # user_books: pandas df containing binary data indicating user book selection
        # user_char: pandas df containing user characteristics data
        
        # Make series of All User IDs
        All_Users = user_chars[user_chars.columns[0]]
        # Make series of User IDs which have associated book selection data
        Training_Users = user_books[user_books.columns[0]]
        Test_Users = All_Users[~All_Users.isin(Training_Users)]
        
        
        # Take only User IDs with book selection data from the character trait data
        Training_Data = user_chars[All_Users.isin(Training_Users)]
        # Trim trailing column; trailing comma present in CSV creating column of NAs
        # & Convert to arrays (input format for Keras) with .values
        TrainX = Training_Data.iloc[:,1:11].values

        # Similar trimming for Y training set
        # Training_Y = user_books.iloc[:, :-1]
        TrainY = user_books.iloc[:,1:1001].values
        
        # Store leftover users into Test X-Block
        Test_Data = user_chars[~All_Users.isin(Training_Users)]
        TestX = Test_Data.iloc[:,1:11].values
        print("X and Y Data Formed")
                
        # returns....
        # TrainX = Training X-Block Data
        # TrainY = Trianing Y-Block Data
        # TestX = User Characteristics Data for Users with Unkown book selections
        # Test_Users = User_IDs for Users with Unkown book selections
        return TrainX, TrainY, TestX, Test_Users
    
       
        
        
    def getModel():
        # Method containing neural net construction details
        
        # Initialise sequential neural network architecture for this model
        model = Sequential()
        
        # add dense input layer, with relu activation 
        model.add(Dense(100, activation="relu", input_shape=(10,)))
        # Drop nodes to prevent overfitting
        model.add(Dropout(0.5))
        
        # Repeat similar for another layer
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.5))

        
        # add dense output layer with Sigmoid activation
        # Sigmoid activation to give independent probabilities for each predicted label:
        model.add(Dense(1000, activation="sigmoid"))
        
        # Define optimiser options:
        # adam = Adam(lr=0.001, decay=0.0)
        # SGD generally better for shallow NN's:
        sgd = SGD(lr=0.01, decay=0, momentum=0, nesterov=True)
        
        # compile model with binary_crossentropy loss function
        # binary_crossentropy because each label is being calculated as individual classes
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print("Neural Net Model Compiled")
        
        # returns the tensorflow NN model
        return model
        
    
    
    def saveModel(self, model, save_as):
        # Saves model details to current working directory
        # model = the tensorflow NN model
        # save_as = filename string to save the model as
        
        # serialize model to JSON
        model_json = model.to_json()
        with open("{0}.json".format(save_as), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("{0}.h5".format(save_as))
        print("Saved model to disk")
        
        
                
    
    def getTop10(self, model, user_books, TestX, Test_Users):
        # Get predictions for User_IDs missing from the book selection csv
        # model = the tensorflow NN model
        # user_books: pandas df containing binary data indicating user book selection
        # TestX: User Characteristics Data for Users with Unkown book selections
        # Test_Users = User_IDs for Users with Unkown book selections
        
        
        # Use scavenged user_char data to make book selection predictions
        preds = model.predict(TestX)
        print("Model predictions calculated")
        
        # get book_ids
        book_headings = user_books.iloc[:,1:1001].columns.values
        
        # Rebuild dataframe with model prediction scores
        predictions = pd.DataFrame(preds, columns=book_headings, index=Test_Users)

        # Get the column index of the top 10 model predictions and store them in an array
        preds_sorted = np.argsort(predictions,axis=1).iloc[:,-10:].values
        # Use locations in the array to select the book_ids from the header
        books = book_headings[preds_sorted]
        
        # Define header for new dataframe
        ranks = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
        # make top10 dataframe
        top10 = pd.DataFrame(books, columns=ranks, index=Test_Users)
        #save to csv
        top10.to_csv("BookRecommendationResults.csv")
        
        print("Top 10 Book recommendations .CSV printed to current working directory")
        


if __name__ == "__main__":
    
    # Get command line arguments
    user_book = sys.argv[1]
    user_char = sys.argv[2]
    
    # Instantiate BookTrainer with command line args as input
    BT = BookTrainer(user_book, user_char)
    TrainX, TrainY, TestX, Test_Users = BT.makeDataSets(BT.user_books, BT.user_chars)
    model = BookTrainer.getModel()
    
    # Fit model to training data
    # Selected a minimal validation split because of smaller sample size (90) and
    # because data is replaced by random data, tuning validation accuracy of this 
    # model is not valuable
    model.fit(TrainX, TrainY, epochs=20, verbose=1, validation_split=0.1)
    
    # Save model
    BT.saveModel(model, "model")
    BT.getTop10(model, BT.user_books, TestX, Test_Users)
        
        
        

    
    
        
'''
if __name__ = '__main__'  
BT = BookTrainer("user_book.csv", "user_char.csv")
TrainX, TrainY, TestX, Test_Users = BT.makeDataSets(BT.user_books, BT.user_chars)
model = BookTrainer.getModel()
# Fit model to training data
# Selected a minimal validation split because of smaller sample size (90) and
# because data is replaced by random data, tuning validation accuracy of this 
# model is not valuable
model.fit(TrainX, TrainY, epochs=20, verbose=1, validation_split=0.1)
BT.saveModel(model)
BT.getTop10(model, BT.user_books, TestX, Test_Users)



# score = model.evaluate(TrainX, TrainY, verbose=1)
# print('Train score:', score[0])
# print('Train accuracy:', score[1])

pred_test = model.predict(TestX)
print(pred_test)

'''