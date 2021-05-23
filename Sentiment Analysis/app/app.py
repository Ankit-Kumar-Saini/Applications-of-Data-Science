# import necessary modules
import re
import nltk
import time
import pickle
import sqlite3
import numpy as np
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from flask import Flask, render_template, request

# download stopwords from nltk
nltk.download('stopwords')


## Function to connect to sql database
def sql_init():
    """
    This function creates a connection to the database and 
    then creates a table in the database
    """
    # create connection to the database
    conn = sqlite3.connect('reviews_database.db')
    
    # create cursor
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS REVIEWS;")

    # sql command
    sql_cmd = """
    CREATE TABLE REVIEWS (TimeStamps INTEGER PRIMARY KEY,
                          MovieNames VARCHAR(20),
                          Reviews VARCHAR(50),
                          Predictions VARCHAR(10));"""

    cur.execute(sql_cmd)
    conn.commit()
    # close the connection
    conn.close()


    
## Function to store reviews in sql database
def sql_store(time_stamp, movie_name, review, prediction):
    conn = sqlite3.connect('reviews_database.db')
    cur = conn.cursor()

    cur.execute("INSERT INTO REVIEWS VALUES (?, ?, ?, ?)", (time_stamp, movie_name, review, prediction))
    conn.commit()
    # close the connection
    conn.close()



# instantiate Flask object
app = Flask(__name__, static_folder = '',)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load vectorizer
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# call function to connect to sql_database
sql_init()


# Home page route
@app.route("/home")
@app.route("/")
def home():
    return render_template('home.html')



# predict route
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        movie_name = request.form['movie']
        review = request.form['review']

        time_stamp = int(time.time())

        # call the function to clean the review
        clean_review = clean_reviews(str(review))

        # transform the review using vectorizer object
        transformed_review = vectorizer.transform(np.array([clean_review]))

        prediction = model.predict(transformed_review)

        sentiment = ['Negative', 'Positive'][prediction[0]]
        sql_store(time_stamp, movie_name, review, sentiment)

        return render_template("pred.html", value = sentiment)

    

# Clean raw reviews
def clean_reviews(review):
    """
    Clean and preprocess a review
    1. Remove HTML tags
    2. Use regex to remove all special characters (only keep letters)
    3. Make strings to lower case and tokenize / word split reviews
    4. Remove English stopwords
    5. Rejoin to one string
    
    Args:
        review: raw text review
        
    Returns:
        review: clean text review
    """
    # 1. Remove HTML tags
    review = BeautifulSoup(review, features = "html.parser").text

    # 2. Use regex to find emoticons
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', review)

    # 3. Remove punctuation
    review = re.sub("[^a-zA-Z]", " ", review)

    # 4. Tokenize into words (all lower case)
    review = review.lower().split()

    # 5. Remove stopwords
    eng_stopwords = set(stopwords.words("english"))
    review = [w for w in review if not w in eng_stopwords]

    # 6. Join the review to form one sentence
    review = ' '.join(review + emotions)

    return review