#Data Collections
from bs4 import BeautifulSoup 
import requests  
from time import sleep  
import numpy as np 
from random import randint

#Data Analysis 
import pandas as pd
import numpy as np  
np.random.seed(0) 
import pickle

#Data Visulaization 
import matplotlib.pyplot as plt   
import plotly.express as px 
import plotly.graph_objects as go 
import plotly.figure_factory as ff 
from urllib.request import urlopen
import json  
import seaborn as sns


#Natural Language Processing 
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words("english")
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist 
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.collocations import * 
import string 
import re 
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator 
import gensim 
from gensim.parsing.preprocessing import preprocess_documents 
import py_stringmatching as sm 
from gensim.models import Word2Vec 
from fuzzywuzzy import fuzz 
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

#Modeling 
from sklearn.cluster import MiniBatchKMeans, KMeans 
from sklearn.decomposition import PCA  

from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.metrics import calinski_harabasz_score, confusion_matrix 
from sklearn.metrics import classification_report 

from sklearn.datasets import fetch_20newsgroups 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB  
from scipy.spatial.distance import pdist, squareform 
from sklearn.manifold import TSNE 
from collections import defaultdict 
from sklearn.pipeline import make_pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer  

def graph_high_frequency_words(word_list, index, name):

    x_list = [x[0] for x in word_list[index][1]]
    y_list = [x[1] for x in word_list[index][1]]

    fig = go.Figure(go.Bar(x=x_list, y=y_list, marker_color='rgb(75, 117, 156)')) 

    fig.update_layout(
        title=f'Highest Frequency Words in {name}'.title(),
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Total Number of Uses',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15, # gap between bars of adjacent location coordinates.
        bargroupgap=0.1)

    fig.show() 

#load the expanded ngs standards pdf into a txt file
def pdf_to_text(filepath, filename):  
    """takes in a local PDF file path and saves it as a local txt.file"""
    import pdftotext
    with open(filepath, "rb") as f:
        pdf = pdftotext.PDF(f)
 
    # Save all text to a txt file.
    with open(filename, 'w') as f:
        f.write("\n\n".join(pdf))  

#single document preproccesing
def open_and_flatten(filename): 
    """takes in a local txt file path and returns a flattened file"""
    file = open(filename) 
    yourResult = [line.split(',') for line in file.readlines()] 
    flat_text = [item for sublist in yourResult for item in sublist] 
    seperator = ","
    joined_file = seperator.join(flat_text)  
    #tokenize string
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    joined_file_tokens = nltk.regexp_tokenize(joined_file, pattern)  
    #set all words to lower case
    joined_file_words_lowered = [word.lower() for word in joined_file_tokens] 
    #remove general stopwords 
    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    stopwords_list += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Student','Name','School',
                       'The', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                       'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'st', 'hs', 'ess', 'ps', 'ms', 
                      'ls', 'level', 'students', 'ex', 'wyoming', 'iv', 'ii', 'iii', 'hs', 'ls', 'ms', 'https', 
                       'etc', 'ets',  'ess', 'ps', 'inc','rights', 'alabama', 'alaska', 'arizona', 'colorado', 
                       'flordia', 'georgia', 'idaho', 'louisiana', 'mass', 'minnesota', 'mississippi', 
                       'missouri', 'montana', 'nebraksa', 'northdakota', 'oklahoma', 
                       'dakota', 'tennessee', 'utah', 'westvirginia', 'wisconsin', 'wyoming', 'curriculum', 
                      'grade', 'science', 'su', 'su', 'table', 'contents', 'back', 'texreg', 'january', 
                      'feburary', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 
                      'november', 'december', 'high', 'school', 'maine', 'michigan', 'carolina', 'ohio', 
                       'pennsylvania', 'texas', 'virginia', 'ch', 'appendix', 'north', 'south', 'va', 'pre', 'grades', 
                       'dci', 'sep', 'ccc', 'sci', 'sc', 'standard', 'standards']
    joined_file_words_stopped = [word for word in joined_file_words_lowered if word not in stopwords_list] 
    return joined_file_words_stopped


def write_multiple_pdfs_to_text(path_list, filename):  
    """takes in multiple local PDF file path and saves it as a local txt.file""" 
    import PyPDF2
    for file in path_list: 
        with open(file, mode='rb') as f:
            reader = PyPDF2.PdfFileReader(f) 
            number_of_pages = reader.getNumPages()  
            for page in range(number_of_pages):   
                page = reader.getPage(page) 
                file = open(filename, 'a')
                sys.stdout = file
                print(page.extractText()) 
                file.close()   
                
                
def general_processing(txtfile):  
    """Takes in a txtfile and does general preprocessing to text data 
    flatten/ sperate on comma/ tokenize/ lower words/ remove basic stop words"""
    #open and flatten file
    file_raw = open_and_flatten(txtfile) 
    #seperate file on , -  
    seperator = ","
    joined_file = seperator.join(file_raw)  
    #tokenize string
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)" 
    joined_file_tokens = nltk.regexp_tokenize(joined_file, pattern)  
    #set all words to lower case
    joined_file_words_lowered = [word.lower() for word in joined_file_tokens] 
    #remove general stopwords 
    stopwords_list = stopwords.words('english')
    stopwords_list += list(string.punctuation)
    stopwords_list += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Student','Name','School',
                       'The', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                       'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'questions', 'science', 'st', 
                      'page', 'chapter', 'psc', 'document', 'local', 'regional']
    joined_file_words_stopped = [word for word in joined_file_words_lowered if word not in stopwords_list] 
    return joined_file_words_stopped 

def bigram_generator(word_list, number_of_pairs):  
    """Input" Alist of words and the number of bigram pairs to return
    Output: The number specified of scored bigram pairs 
    *A bigram is just a count*""" 
    bigram_measures = nltk.collocations.BigramAssocMeasures() 
    bigram_finder = BigramCollocationFinder.from_words(word_list) 
    bigram_scored = bigram_finder.score_ngrams(bigram_measures.raw_freq) 
    return bigram_scored[:number_of_pairs]


def pmi_generator(list_of_words, freq_filter):  
    """Input: A list of words and the frequency minumum for those words to have appeared together 
    Output: The paired pmi scored words 
    *PMI is a probability - only one pairing will be very high*""" 
    bigram_measures = nltk.collocations.BigramAssocMeasures() 
    pmi_finder = BigramCollocationFinder.from_words(list_of_words) 
    pmi_finder.apply_freq_filter(freq_filter)  
    pmi_scored = pmi_finder.score_ngrams(bigram_measures.pmi) 
    return pmi_scored 

def word_cloud(word_list): 
    """Input: A list of words 
    Output: a word cloud"""
    unique_string=(" ").join(word_list)
    wordcloud = WordCloud(width = 1000, height = 500, max_font_size=90, max_words=100,
                      background_color="white", colormap="nipy_spectral").generate(unique_string)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("your_file_name"+".png", bbox_inches='tight')
    plt.show()
    plt.close() 
    
def bigram_generator_all(word_list):  
    """Input" Alist of words and the number of bigram pairs to return
    Output: The number specified of scored bigram pairs 
    *A bigram is just a count*""" 
    bigram_measures = nltk.collocations.BigramAssocMeasures() 
    bigram_finder = BigramCollocationFinder.from_words(word_list) 
    bigram_scored = bigram_finder.score_ngrams(bigram_measures.raw_freq) 
    return bigram_score 

def alignment_processing(docname): 
    """Takes in a txt file 
    Returns a bag of words corpus that can but used to measure similarity""" 
    #open the document & append lines to list 
    file_docs = []
    with open (docname) as f:
        tokens = sent_tokenize(f.read())
        for line in tokens:
            file_docs.append(line) 
    #tokenize the document 
    gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs] 
    #create a dictionary of the tokenized document 
    dictionary = gensim.corpora.Dictionary(gen_docs) 
    #create a bag of words 
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs] 
    return corpus 

def compare_docs(textfile):  
  
    current_doc = []
    with open (textfile) as f:
        tokens = sent_tokenize(f.read())
        for line in tokens:
            current_doc.append(line) 
    
    processed_doc = preprocess_documents(current_doc)  
    dictionary_current = gensim.corpora.Dictionary(processed_doc)  
    corpus_current = [dictionary_current.doc2bow(processed_doc) 
                      for processed_doc in processed_doc]  
    tf_idf = gensim.models.TfidfModel(ngss_corpus)
            
    current_doc_tf_idf = tf_idf[corpus_current]
    sum_of_sim =(np.sum(sims[current_doc_tf_idf], dtype=np.float32)) 
    percentage_of_similarity = round(float(sum_of_sim / len(current_doc)))
    name = state.title() 
    print("{} Alignment: %{}".format(name, percentage_of_similarity))