import spacy
import nltk
import time
import re

import numpy as np
import pandas as pd

from nltk.corpus import stopwords

def get_post(query): 
    nlp = spacy.load("en_core_web_sm") # python -m spacy download en_core_web_sm
    tokenized = nlp(query)
    return tokenized

def get_input(document): 
    text = ""
    for token in document: 
        if token.is_alpha and not token.is_stop:  
            text += " " + str(token)
    return text.strip().split()

def time_important(document): 
    influence = ['influential', 'important', 'pivotal', 'significant', 'leading', 'noteworthy']
    time = ['latest', 'recent', 'newest', 'current']
    text = ""
    important = False
    for token in document: 
        if token not in influence or time: 
            text += " " + str(token)
        else: 
            important = True
    return text.strip(), important

def get_dates(query):
    dates = []
    for ent in filter(lambda e: e.label_=='DATE',query.ents):
        dates.append(ent.text)
    return dates

def get_authors(query):
    authors = []
    for ent in filter(lambda e: e.label_=='PERSON',query.ents):
        authors.append(ent.text)
    return authors

    