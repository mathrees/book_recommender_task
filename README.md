__Quick Start__

in Linux/GitBash:

Move .csv files from repo/Data folder into the current working directory containing the scripts
'''
cd /target/folder/filepath
git clone https://github.com/mathrees/book_recommender_task.git
python BookTrainer.py "user_book.csv" "user_char.csv"
python BookTester.py "model.h5" "model.json" "test_user_char.csv" "test_user_book.csv"
'''

"test_user_char.csv" was created from original user_char.csv
"test_user_book.csv" is dummied to test work


__Version Dependencies__
Python 3.6.3
Pandas 0.20.3
Numpy 1.13.3
Keras 2.1.2
TensorFlow 1.4.0
scikit-learn 0.19.1



__SETTING UP__

Assuming python, python path etc is correctly set up...
As well as Tensorflow, and other python packages...
If not, refer to package documentations for installation instructions

change directory to target directory to clone github project:
  cd /target/folder/filepath

clone github project:
  git clone <project>


__TO TRAIN THE MODEL__

Ensure CSV files containing the training data are present in folder (e.g. user_book.csv & user_char.csv)
These are assumed to be identical in format to those provided in example (i.e. with headers and rows synchronised on user_id)


In linux command line (or GitBash), run the command:
  python BookTrainer.py {USER_BOOKS_CSV} {USER_TRAITS_CSV}
where {} denotes strings of the csv files
E.G:
  python BookTrainer.py "user_book.csv" "user_char.csv"


This trains the model and saves the following to the current working directory:
"model.h5" - Neural Network Model Weights
"model.json" - Neural Network Model details
"BookRecommendationResults.csv" - Top 10 Book Recommendations for users with missing book selection data

Note: The User_IDs must align in the two CSVs


__TO MAKE NEW PREDICTIONS & TEST THE MODEL__

Ensure CSV files containing the test data are present in folder (e.g. test_user_book.csv & test_user_char.csv)
The User_IDs must align in the two CSVs

NOTE!!!! If test_user_char.csv as a subset of user_char.csv doesn't exist...
...you will need to create this and use it as an input alongside test_user_book.csv for it to work

In linux command line (or GitBash), run the command:
  python BookTester.py {MODEL_WEIGHTS} {MODEL_JSON} {USER_TRAITS_CSV} {USER_BOOKS_CSV} 
where {} = strings for various file names in the working directory
E.G.
  python BookTester.py "model.h5" "model.json" "test_user_char.csv" "test_user_book.csv"

This takes data from test_user_char.csv and generates the top 10 recommendations and saves them in: 
"BookRecommendationResults.csv" - Top 10 Book Recommendations for users with missing book selection data

Hamming_Loss is used as the evaluation metric and is printed on the command line
Hamming_Loss is the fraction of labels that are incorrectly predicted

Note: The User_IDs must align in the two CSVs
