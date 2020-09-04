import re
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# For Windows
path = os.getcwd()
file = '\data\processed_file.csv'

df = pd.read_csv(path+file)

# Replace Categories with numbers 1 and 0
df['sentiment'] = df['sentiment'].replace({'negative':0,'positive':1})

# Feature Selection
# Tf-Idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(df['review'])
print(tfidf.shape)

pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))


# Size of training data
train_size = int(0.80 * tfidf.shape[0])

# Separating data into train and test set
X_TRAIN = tfidf[:train_size,:]
X_TEST = tfidf[train_size:,:]
Y_TRAIN = df['sentiment'][:train_size]
Y_TEST = df['sentiment'][train_size:]

# Splitting training data into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_TRAIN,Y_TRAIN,random_state=10,test_size=0.25)

# Finding the best parameters
# def model_fitting(X_train, y_train, algorithm_name, algorithm, gridsearchparameter, cv, n_jobs=-1):
#     np.random.seed(10)
#
#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
#     grid = GridSearchCV(estimator=algorithm,
#                         param_grid=gridsearchparameter,
#                         cv=cv,
#                         scoring='accuracy',
#                         verbose=True,
#                         n_jobs=-1)
#
#     results = grid.fit(X_train, y_train)
#     best_parameters = results.best_params_
#     return best_parameters

model_score = {}

# # XGBoost Model
# parameters = {
#     "n_estimators":[1000,1250,1500,2000],
#     # "tree_method":["gpu_hist"],
#     "n_jobs":[-1],
#     "max_depth":[4,5,6]
# }
# xgb_best_params = model_fitting(X_train,y_train,"XGBoost",XGBClassifier(),parameters,cv=5)
# print(f"XGB Best Parameters: {xgb_best_params}\n")
"""
On running grid search cv the best parameters are
xgb_best_params = {'max_depth': 6, 'n_estimators': 1250, 'n_jobs': -1}
"""
estimators = 1250 # xgb_best_params['n_estimators']
max_depth = 6     # xgb_best_params['max_depth']
xgb_model = XGBClassifier(n_estimators=estimators, max_depth=max_depth)
xgb_model.fit(X_train,y_train)

# Prediction on Validation Set
xgb_val_predictions = xgb_model.predict(X_val)
xgb_val_f1_score = f1_score(y_val,xgb_val_predictions)
print(f"Validation F1 Score: {xgb_val_f1_score:.3f}")

# Prediction on Test Set
xgb_test_predictions = xgb_model.predict(X_TEST)
xgb_test_f1_score = f1_score(Y_TEST,xgb_test_predictions)
print(f"Validation F1 Score: {xgb_test_f1_score}")

model_score["XGBoost"] = []
model_score["XGBoost"].append(xgb_model)
model_score["XGBoost"].append(xgb_test_f1_score)

# Naive Bayes
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)

# Prediction on Validation Set
nb_val_predictions = naive_bayes_classifier.predict(X_val)
nb_val_f1_score = f1_score(y_val, nb_val_predictions)
print(f"Naive Bayes Validation F1 Score: {nb_val_f1_score:.3f}")

# Prediction on Test Set
nb_test_predictions = naive_bayes_classifier.predict(X_TEST)
nb_test_f1_score = f1_score(Y_TEST, nb_test_predictions)
print(f"Naive Bayes Test F1 Score: {nb_test_f1_score:.3f}")

model_score["Naive Bayes"] = []
model_score["Naive Bayes"].append(naive_bayes_classifier)
model_score["Naive Bayes"].append(nb_test_f1_score)

# Logistic Regression
parameters = {
    "C": np.logspace(-4, 4, 10),
    "penalty":["l1","l2"],
    "solver":["liblinear","saga"],
    "max_iter":[3000, 4000, 5000, 6000,]
}

# lr_best_params = model_fitting(X_train,y_train,"Logistic Regression",LogisticRegression(),parameters,cv=5)
# print(f"Logistic Regression Best Parameters: {lr_best_params}\n")

"""
On running grid search cv the best parameters are
lr_best_params = {'C': 2.782559402207126, 'max_iter': 4000, 'penalty': 'l2', 'solver': 'liblinear'}
"""
C = 2.782 # lr_best_params["C"]
max_iter = 4000 # lr_best_params["max_iter"]
penalty = 'l2' # lr_best_params["penalty"]
solver= 'liblinear' # lr_best_params["solver"]

lr_model = LogisticRegression(C=C,max_iter=max_iter,penalty=penalty,solver=solver)
lr_model.fit(X_train,y_train)

# Prediction on Validation Set
lr_val_predictions = lr_model.predict(X_val)
lr_val_f1_score = f1_score(y_val,lr_val_predictions)
print(f"Logistic Regression Validation F1 Score: {lr_val_f1_score:.3f}")

# Prediction on Test Set
lr_test_predictions = lr_model.predict(X_TEST)
lr_test_f1_score = f1_score(Y_TEST,lr_test_predictions)
print(f"Logistic Regression Test F1 Score: {lr_test_f1_score:.3f}")

model_score["Logistic Regression"] = []
model_score["Logistic Regression"].append(lr_model)
model_score["Logistic Regression"].append(lr_test_f1_score)


print("Model Score : ",model_score)
max_acc = 0
model_name = ""
for key in model_score.keys():
    acc = model_score[key][1]
    print(acc)
    if (acc > max_acc):
        max_acc = acc
        model_name = model_score[key][0]
        fname = key + '.pkl'
# fname = str(model_name) + '.pkl'
pickle.dump(model_name, open(fname,"wb"))


