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
* [Technologies Used](#technologies-used)
* [License](#license)
* [Credits](#credits)

## Demo
## Overview
This is a sentiment analysis project. The aim of the project is to classify movie reviews as either positive or negative. I have used [nltk](https://www.nltk.org/) for data preprocessing and scikit-learn implementation of tf-idf algorithm for feature extractiion. The models used for classification are Logistic Regression, XGBoost and Naive Bayes. The reultant model with the least f1_score is used for predictions. GridSearchCV is used for hypertuning each of the three models. The project is deployed using flask on heroku.

## Requirements
**Python version:** 3.6  
**Packages:** numpy pandas scikit-learn nltk matplotlib seaborn flask pickle  
**Softwares:** Anaconda Python, PyCharm, Jupyter  

## Installation Steps
**Anaconda Installation:** https://www.anaconda.com/products/individual  
**Creating Conda Environment:** `conda create -n <your env name> python=3.6`  
**Activating Conda Environment:** `conda activate <your env name>`  
**For deployment purpose:** `pip install -r requirements.txt`  

## Dataset Description
Dataset contains 50,000 records for text analytics. The dataset is balanced and consist of 25,000 positive as well as negative reviews each.  
**Dataset link:** [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)  
## Data Preprocessing
To perform sentiment analysis first we have to clean the data. For that first I removed all the emojis and symbols from the tweets. To do that I used regular expressions to identify the patterns containing emojis and symbols.  
After that I removed stopwords from the tweets because they do not add any meaning to the statement as well as take space. Removing stopwords also reduced size of the training data.For removing stopwords first I downloaded all stopwords using 
> `nltk.download('stopwords')`  

and then loaded into the program using,
> `stop_words = nltk.stopwords.words('english')`

Then I applied **Tokenization** and **Stemming** to each tweet.  

**Tokenization**  
It is the process of breaking down text into smaller pieces known as tokens.

*Example code*  
`# Tokenize a paragraph  `  
`from nltk.tokenize import sent_tokenize  `  
`para = ['This is a sentence. A sentence consists of a noun object and a verb.]`  
`chunks = sent_tokenize(para)`  
`print(chunks)`  
**Output**  
`['This is a sentence.',`  
`'A sentence consists of a noun object and a verb.']`

`# Tokenize a sentence  `  
`from nltk.tokenize import word_tokenize  `  
`sen = 'This is a sentence. A sentence consists of a noun object and a verb.`  
`chunks = sent_tokenize(sen)`  
`print(chunks)`  
**Output**  
`['This', 'is', 'a', 'sentence', '.', 'A', 'sentence', 'consists', 'of', 'a', 'noun', 'object', 'and', 'a', 'verb', '.']`



 

## Model Building
## Deployment
## To do
## Technologies Used
## License
## Credits