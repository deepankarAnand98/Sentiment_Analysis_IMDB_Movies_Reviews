# Sentiment_Analysis_IMDB_Movies_Reviews
## Table of Contents
* [Demo](#demo)
* [Overview](#overview)
* [Requirements](#requirements)
* [Intallation Steps](#installation-steps)
* [Dataset Description](#dataset-description)
* [Data Preprocessing](#data-preprocessing) 
* [Model Building](#model-preprocessing)
* [Deployment](#deployment)
* [To do](#to-do)
* [License](#license)
* [Credits](#credits)

## Demo
## Overview
<p style='text-align: justify;'>This is a sentiment analysis project. The aim of the project is to classify movie reviews as either positive or negative. I have used [nltk](https://www.nltk.org/) for data preprocessing and scikit-learn implementation of tf-idf algorithm for feature extractiion. The models used for classification are Logistic Regression, XGBoost and Naive Bayes. The reultant model with the least f1_score is used for predictions. GridSearchCV is used for hypertuning each of the three models. The project is deployed using flask on heroku.</p>

## Requirements
**Python version:** 3.6  
**Packages:** numpy pandas scikit-learn nltk matplotlib seaborn flask pickle  
**Softwares:** Anaconda Python, PyCharm, Jupyter  

## Installation Steps
**Anaconda Installation:** https://www.anaconda.com/products/individual  
**Creating Conda Environment:** `conda create -n Sentiment_Analysis_IMDB_Movies_Reviews python=3.6`  
**Activating Conda Environment:** `conda activate Sentiment_Analysis_IMDB_Movies_Reviews`  
**For deployment purpose:** `pip install -r requirements.txt`  

## Dataset Description
Dataset contains 50,000 records for text analytics. The dataset is balanced and consist of 25,000 positive as well as negative reviews each.  
**Dataset link:** [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)  
## Data Preprocessing
<p style="text-align:justify">To perform sentiment analysis first we have to clean the data. For that first I removed all the emojis and symbols from the tweets. To do that I used regular expressions to identify the patterns containing emojis and symbols.</p>  
<p style="text-align:justify">After that I removed stopwords from the tweets because they do not add any meaning to the statement as well as take space. Removing stopwords also reduced size of the training data.For removing stopwords first I downloaded all stopwords using </p>
> `nltk.download('stopwords')`  

and then loaded into the program using,
> `stop_words = nltk.stopwords.words('english')`

Then I applied **Tokenization** and **Stemming** to each tweet.  

### Tokenization 
It is the process of breaking down text into smaller pieces known as tokens.

### Stemming
It is the process of reducing a word to its stem(or root).
> Stem word for runner and running is run.  
## Model Building
<p style="text-align:justify">First, I transformed the categorical variables into numerical variables.I transformed the tweets into features using tf-idf algorithm. The size of resultant matrix is 50000&times;10000.</p>
<p style="text-align:justify"></p>
<p style="text-align:justify">I also split the data into train and tests sets with a test size of 20%. I also split the training set into train and validation sets. The validation set is 25% of the original train set.</p>
<p style="text-align:justify">For modelling I tried three different models</p>

* Logistic Regression
* Naive Bayes
* XGBoost
 
 ### Model Performance
* **Logistic Regression**
* **Naive Bayes**
* **XGBoost**


## Deployment
## To Do
* Add desciprtion to Tokenization and Stemming
* Add EDA ipython notebook
* Add all the plots Plots
## License

## Credits