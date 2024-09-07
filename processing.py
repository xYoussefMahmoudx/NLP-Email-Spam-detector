import pandas as pd
import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords

def remove_subject(text):
    return re.sub(r'^Subject:\s*', '', text)
def lower_char(text):
    return text.lower()

def replace_numbers_and_remove_punctuations(text):
   
    text = re.sub(r'\$ ?\d+|\d+ ?\$', ' moneynumber ', text)
   
    text = re.sub(r'\b\d+\b', ' number ', text)
   
    text = re.sub(r'[^\w\s]', '', text)
    return text

def tokenize_email_words(text):
    return nltk.word_tokenize(text)

def stop_words_removal(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]

lemmatizer = nltk.WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun


def lemmatize_text(tokens):
   
    pos_tags = nltk.pos_tag(tokens)
    
    return [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]


def process_Data(text):
    text=remove_subject(text)

    text=lower_char(text)

    text=replace_numbers_and_remove_punctuations(text)
  
    text=tokenize_email_words(text)

    text=lemmatize_text(text)

    return text
    
