import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
#import tensorflow as tf
import os
import re


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def preprocess(input,type):

    if type=='attribute':
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')
        #w = input.replace('_', ' ')

        tokens = word_tokenize(w)

        camel_tokens=[]

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens=camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter
        # remove punctuation from each word
        # table = str.maketrans('', '', string.punctuation)
        # stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in inter_words2 if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # final_words = []
        # for w in words:
        #     inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words += inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok)]

    elif type=='value':
        #w = input.replace('_', ' ').replace(',', ' ')
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')
        #w=input

        tokens = word_tokenize(w)

        camel_tokens = []

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens = camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter

        # remove punctuation from each word
        # table = str.maketrans('', '', string.punctuation)
        # stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic

        numerical_values=[]
        string_values=[]
        for word in inter_words2:
            try:
                float(word)
                numerical_values.append(word)

            except ValueError:
                string_values.append(word)


        string_values_final=[]
        for w in string_values:
            inter=re.split(r'(\d+)', w)

            for word in inter:
                if len(word)>0:
                    try:
                        float(word)
                        numerical_values.append(word)

                    except ValueError:
                        string_values_final.append(word)

        #keep 0 digits
        #numerical_values = [re.sub('\d', '#', s) for s in numerical_values]

        #keep 1 digit
        numerical_values_inter=[]
        for s in numerical_values:
            if s[0]=='-':
                ss=s[2::]
                ss=re.sub('\d', '#', ss)
                ss=s[0:2]+ss


            else:
                ss = s[1::]
                ss = re.sub('\d', '#', ss)
                ss = s[0] + ss

            numerical_values_inter += [ss]

        #keep 2 digits

        # for s in numerical_values:
        #     ss=s[2::]
        #     ss=re.sub('\d', '#', ss)
        #     ss=s[0:2]+ss
        #     numerical_values_inter+=[ss]

        numerical_values=numerical_values_inter
        inter_words2 = string_values_final

        words = [word for word in inter_words2 if word.isalpha() or word in['$','@','%','£','€','°']]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        stop_words.remove('d')
        stop_words.remove('m')
        stop_words.remove('s')

        words = [w for w in words if not w in stop_words]

        # final_words = []
        # for w in words:
        #     inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words += inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok) or tok in['$','@','%','£','€','°']]

        final_words=final_words+numerical_values



    elif type=='value2':

        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')

        tokens = word_tokenize(w)

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter


        numerical_values=[]
        string_values=[]
        for word in inter_words2:
            try:
                float(word)
                numerical_values.append(word)

            except ValueError:
                string_values.append(word)


        string_values_final=[]
        for w in string_values:
            inter=re.split(r'(\d+)', w)

            for word in inter:
                if len(word)>0:
                    try:
                        float(word)
                        numerical_values.append(word)

                    except ValueError:
                        string_values_final.append(word)



        inter_words2 = string_values_final

        words = [word for word in inter_words2 if word.isalpha() or word in['$','@','%','£','€','°']]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        stop_words.remove('d')
        stop_words.remove('m')
        stop_words.remove('s')

        words = [w for w in words if not w in stop_words]

        final_words = []
        for w in words:
            inter = re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
            final_words += inter

        final_words = [tok for tok in final_words if isEnglish(tok) or tok in['$','@','%','£','€','°']]

        final_words=final_words+numerical_values






    elif type == 'description':
        #w = input.replace('_', ' ').replace(',', ' ').replace('-', " ").replace('.', ' ')
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')


        tokens = word_tokenize(w)

        camel_tokens = []

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens = camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter
        # remove punctuation from each word
        #table = str.maketrans('', '', string.punctuation)
        #stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in inter_words2 if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # final_words=[]
        # for w in words:
        #     inter=re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words+=inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok)]

        not_to_use=['com','u','comma','separated','values','csv','data','dataset','https','api','www','http','non','gov','rows','p','download','downloads','file','files','p']

        final_words=[tok for tok in final_words if tok not in not_to_use]

    elif type == 'query':
        #w = input.replace('_', ' ').replace(',', ' ').replace('-', " ").replace('.', ' ')
        w = input.replace('-', " ").replace('_', ' ').replace('/', ' ').replace(',', ' ').replace('.', ' ').replace('|', ' ').replace(':', ' ')


        tokens = word_tokenize(w)

        camel_tokens = []

        for w in tokens:
            inter = camel_case_split(w)
            camel_tokens += inter

        tokens = camel_tokens

        # convert to lower case
        tokens = [w.lower() for w in tokens]
        inter_words = []
        for w in tokens:
            inter = re.sub(r'\u2013+', ' ', w).split()
            inter_words += inter

        inter_words2 = []
        for w in inter_words:
            inter = re.sub(r'\u2014+', ' ', w).split()
            inter_words2 += inter
        # remove punctuation from each word
        #table = str.maketrans('', '', string.punctuation)
        #stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in inter_words2 if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]

        # final_words=[]
        # for w in words:
        #     inter=re.sub('([a-z])([A-Z])', r'\1 \2', w).split()
        #     final_words+=inter

        final_words=words

        final_words = [tok for tok in final_words if isEnglish(tok)]

        #not_to_use=['com','u','comma','separated','values','csv','data','dataset','https','api','www','http','non','gov','rows','p','download','downloads','file','files','p']

        #final_words=[tok for tok in final_words if tok not in not_to_use]

    return final_words


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best
