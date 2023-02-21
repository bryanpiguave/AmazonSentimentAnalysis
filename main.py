import numpy as np
import nltk
import pandas as pd
import re
import time
from sklearn.svm import SVC # Support Vector Classification model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import argparse
import pickle

from utils.preprocessing import tokenize, reg_expressions

maxtokens=30
maxtokensize=20


parser=argparse.ArgumentParser()
parser.add_argument("--model",choices=["SVC","Logistic"],default="SVC")

args=parser.parse_args()



#Building a classifier
def fit(train_x,train_y):
    model = LogisticRegression()
    try:
        model.fit(train_x, train_y)
    except:
        pass
    return model



def unison_shuffle_data(data, header):
    p = np.random.permutation(len(header))
    data = data[p]
    header = np.asarray(header)[p]
    return data, header



nltk.download('stopwords')
from nltk.corpus import stopwords
stpwords = stopwords.words("english")

def stop_word_removal(row):
    token = [token for token in row if token not in stpwords]
    token = filter(None, token)
    return token

def min_word_removal(row):
    token = [token for token in row if len(token) > 3]
    token = filter(None, token)
    return token


def assemble_bag(data):
    used_tokens = []
    all_tokens = []
    for i,item in enumerate(data):
        for token in item:
            if token in all_tokens:
                if token not in used_tokens:
                    used_tokens.append(token)
            else:
                all_tokens.append(token)
    df = pd.DataFrame(0, index = np.arange(len(data)), columns = used_tokens)
    for i, item in enumerate(data):
        for token in item:
            if token in used_tokens:
                df.iloc[i][token] += 1
    return df


def main():
    raw_data =pd.read_csv(filepath_or_buffer='data/amazon.csv')

    #Preprocessing step
    raw_data["tokens"]=raw_data["review_content"].apply(tokenize)
    raw_data["tokens"]=raw_data["tokens"].apply(stop_word_removal)
    raw_data["tokens"]=raw_data["tokens"].apply(lambda x:reg_expressions(x))
    raw_data["tokens"]=raw_data["tokens"].apply(min_word_removal)
    raw_data["tokens"]=raw_data["tokens"].apply(lambda x:reg_expressions(x))

    raw_data["rating"]=pd.to_numeric(raw_data["rating"],errors='coerce')


    raw_data.dropna(subset=["rating"],inplace=True)
    raw_data["rating"]=raw_data["rating"].astype(float)

    raw_data["label"]=1*pd.to_numeric(raw_data["rating"].astype(float)>=4)

    print(raw_data[["tokens","label"]].head())

    df = assemble_bag(raw_data["tokens"].values)
    data, labels = unison_shuffle_data(df.values, raw_data["label"].values)
    print(df)

    #Saving bag of words 
    bag_of_words=list(df.keys())
    pickle.dump(bag_of_words, open("models/current_bag_of_words", 'wb'))

    idx = int(0.7*data.shape[0])
    #Uses 70% of data for training
    train_x = data[:idx]
    train_y = labels[:idx]
    test_x = data[idx:]
    test_y = labels[idx:]


    if args.model == "Logistic":
        model = fit(train_x=train_x,train_y=train_y)
        predicted_labels = model.predict(test_x)
        acc_score = accuracy_score(test_y, predicted_labels)
        print("The logistic regression accuracy score is:")
    else:
        model = SVC(C=1, gamma="auto", kernel='linear',probability=False)
        start_time = time.time()
        model.fit(train_x, train_y)
        end_time = time.time()
        print("Training the SVC Classifier took %2f seconds"%(end_time-start_time))
        predicted_labels = model.predict(test_x)
        acc_score = accuracy_score(test_y, predicted_labels)
        print("The SVC Classifier testing accuracy score is:")
    print(acc_score)

    #Saving model
    print("\n","Saving model")
    pickle.dump(model, open("models/current_model.pkl", 'wb'))


    return 0






if __name__=="__main__":
    main()
