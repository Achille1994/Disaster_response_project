import nltk
import re
nltk.download(['punkt', 'wordnet', 'stopwords','omw-1.4'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def tokenize(text):
      
    # Remove stop words
    stop_words = stopwords.words("english")
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # remove punctuation,lemmatize and tokenize
    
    tokens=word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    clean_tokens=[lemmatizer.lemmatize(word)
                  for word in tokens if word not in stop_words]
    
    return clean_tokens