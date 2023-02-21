

import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stpwords = stopwords.words("english")

maxtokens=20
maxtokensize=20

def tokenize(row):
    if row  in [None,""]:
        tokens=""
    else:
        tokens=str(row).split(" ")[:maxtokens]
    return tokens

def reg_expressions(row):
    tokens = []
    try:
        for token in row:
            token = token.lower()
            token = re.sub(r'[\W\d]', "", token)
            token = token[:maxtokensize]
            tokens.append(token)
    except:
        token = ""
        tokens.append(token)   
    return tokens

def stop_word_removal(row):
    token = [token for token in row if token not in stpwords]
    token = filter(None, token)
    return token