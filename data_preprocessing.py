import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df = pd.read_csv('data/IMDB_Dataset.csv')

"""
The tweets contains emojis and words surrounded with angle brackets.

"""
def preprocessing(txt):
    # Removing angle brackets
    txt = re.sub('<[^>]*>','',txt)

    # Removing Emojis
    emojis = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)",txt)
    txt = re.sub('[\W]+', ' ', txt.lower()) +\
          ' '.join(emojis).replace('-','')
    return txt

df['review'] = df['review'].apply(preprocessing)

# Tokenization
df['review'] = df['review'].str.split()

# Stemming
stop_words = stopwords.words('english')
ps = PorterStemmer()

stemmed_reviews = []
for i in range(len(df)):
    txt = df.iloc[i,0]
    txt = [ps.stem(word) for word in txt if word not in stop_words]
    txt = ' '.join(txt)
    stemmed_reviews.append(txt)

df['review'] = stemmed_reviews

df.to_csv('processed_file.csv',index=False)






