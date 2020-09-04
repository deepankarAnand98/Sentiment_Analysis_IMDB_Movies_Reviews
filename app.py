from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

classifier = pickle.load(open('Logistic Regression.pkl','rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        data = request.form["message"]
        data = [data]
        X = tfidf_vectorizer.transform(data)
        prediction = classifier.predict(X)

    return render_template('output.html',prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
