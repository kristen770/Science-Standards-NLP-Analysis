"""This file contains all of the libraries and functions used in the Correlating NGS and State Based Science Standards Project"""

#################################################### Libraries and Packages ######################################################

#Data Collections
from bs4 import BeautifulSoup 
import requests  
from time import sleep  
import numpy as np 
from random import randint 
import pdftotext  
import PyPDF2

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
from nltk.stem.porter import *

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


################################################# Data Gathering #################################################################

def pdf_to_text(filepath, filename):  
   #Function to convert pdf to txt file 
    
    #Args: 
        #filepath: The local file path for pdf file
        #filename: The name txt file will be saved as 
        
    #Returns: 
        #File: A txt file  
        
    #Examples: 
        #pdf_to_text("\Users\Downloads\SomePDF", "examplefile") 
        #returns the contents of the pdf inside txt file with name as specified 
             
    #Required Packages: 
        #pdftotext
   
    
    #Open PDF file 
    with open(filepath, "rb") as f:
        pdf = pdftotext.PDF(f)
 
    # Write complete contents to new TXT file
    with open(filename, 'w') as f:
        f.write("\n\n".join(pdf))    


############################################## Data Cleaning ##################################################################### 

def open_and_flatten(filename): 
    #Takes a txt file and returns a preprocessed file
    
    #Args: 
        #filname: txt file name 
    
    #Returns: 
        #A flattened, joined, lowered, tokenized and cleaned(*project specific cleaning applied*) list of strings 
        
    #Examples:  
        #open_and_flatten('examplefile') **examplefile cointains the following: "This is a text and an example" 
        #returns ["This", "text", "example"] 
    #RequiredPackages: 
        #nltk, nltk stopwords/ nlkt. tokenize: sent_tokenize, word_tokenize, RegexpTokenizer 
    
    #open txt file
    file = open(filename) 
    yourResult = [line.split(',') for line in file.readlines()]  
    
    #flatten txt file
    flat_text = [item for sublist in yourResult for item in sublist]  
    
    #joinfile
    seperator = ","
    joined_file = seperator.join(flat_text)   
    
    #tokenize string
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    joined_file_tokens = nltk.regexp_tokenize(joined_file, pattern)  
    
    #set all words to lower case
    joined_file_words_lowered = [word.lower() for word in joined_file_tokens]  
    
    #remove general & project speficic stopwords 
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
                       'dci', 'sep', 'ccc', 'sci', 'sc', 'standard', 'standards', 'lst', 'questions', 'science', 'st', 
                      'page', 'chapter', 'psc', 'document', 'local', 'regional', 'whst', 'ature', 'th', 'rst', 'ee', 'rp', 
                      'sl', 'md', 'mp', 'nrc', 'nbt', 'rl'] 
    
      #join stopped words
    joined_file_words_stopped = [word for word in joined_file_words_lowered if word not in stopwords_list] 
    return joined_file_words_stopped 

########################################### Natural Language Processing #########################################################


def graph_high_frequency_words(word_list, count, value): #, count, name
    #Graphs a histogram of the highest freqency words in a text list 
    
    #Args: 
    #word_list: A list of strings and their frequency counts 
    
    #Returns: 
        #Plotly bar graph (histogram) of the word and their frequency 
        
    #Required Packages: 
        #Plotly 
    
    #seperate word and their frequency counts
    x_list = [x[0] for x in word_list[count][1]]
    y_list = [x[1] for x in word_list[count][1]]

    #graph x_list & y_list
    fig = go.Figure(go.Bar(x=x_list, y=y_list, text=y_list,
            textposition='auto', marker_color='rgb(75, 117, 156)')) 

    #style graph 
    fig.update_layout(
        title="Highest Frequency Words",
        xaxis_tickfont_size=10,
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
        bargap=0.15, 
        bargroupgap=0.1) 
    
    fig.update_layout(barmode='group', xaxis_tickangle=-45) 

    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    fig.show() 
                

def bigram_generator(word_list, number_of_pairs):  
    #Returns a list of bigram pairs of specificed length 
    
    #Args: 
        #word_list: A list of strings 
        #number_of_pairs: Int 
        
    #Returns: 
        #A specified number of bigram pairs   
        
    #Examples: 
        #bigram_generator(example_list, 10) 
        #returns highest raw freqency bigrams from example list 
    
    #Required Packages: 
        #nltk
    
    #calculate bigram pairs
    bigram_measures = nltk.collocations.BigramAssocMeasures() 
    bigram_finder = BigramCollocationFinder.from_words(word_list)  
    
    #score bigram pairs
    bigram_scored = bigram_finder.score_ngrams(bigram_measures.raw_freq)  
    return bigram_scored[:number_of_pairs]


def pmi_generator(word_list, probability_filter):  
    #Returns a list of pmi pairs of a specified freqency or above 
        
    #Args:  
        #word_list: A list of strings 
        #probability_filter: Int
           
    #Returns: 
        #A list of paired words and the probability of those words appearing together in the text 
        
    #Examples: 
        #pmi_generator(example_list, 50) 
        #returns word pairings with a higher than 50% probability of appearing together 
        
    #Required Packages: 
        #nltk
        
    #calcualte pmi pairs
    bigram_measures = nltk.collocations.BigramAssocMeasures() 
    pmi_finder = BigramCollocationFinder.from_words(word_list)   
    
    #apply probability filter 
    pmi_finder.apply_freq_filter(probability_filter)  
    pmi_scored = pmi_finder.score_ngrams(bigram_measures.pmi)  
    return pmi_scored 

def word_cloud(word_list): 
    #Generates a preformated WordCloud 
    #Args:
        #word_list: A list of strings 
    
    #Returns: 
        #A preformated word cloud 
        
    #Required Packages: 
        #Wordcloud, Wordcloud: ImageColorGenerator
     
    #join text to single string
    unique_string=(" ").join(word_list) 
    
    #make pre formated wordcloud
    wordcloud = WordCloud(width = 1000, height = 500, max_font_size=90, max_words=100,
                      background_color="white", colormap="ocean").generate(unique_string)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig("your_file_name"+".png", bbox_inches='tight')
    plt.show()
    plt.close() 

    
    
##################################################### Modeling ###################################################################

def plot_coefficients(classifier, feature_names, top_features=10): 
    #Plots the most feature importance of specified number positive / negative 
    
    #Args: 
        #classifier: classification model 
        #feature_names: list of names of features  
        #top_features: number of features to graph default =10
        
    #Returns: 
        #A graph of the most postive & negative coefficients with displayed feature name 
     
    #calculate coefficients 
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
   
    #format graph 
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right') 
    plt.title('Feature Imporatance') 
    plt.xlabel('Word')
    plt.ylabel('Importance')
    plt.show() 
    

def tokenize_and_stem(text): 
    #Brandon Rose Function first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token 
    
    #Args: 
        #text: string 
    
    #Returns: string tokenized and stemmed
    
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token) 
     

    stemmer = PorterStemmer()
    
    #stem tokenized words 
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


################################################### String Distances Calculations ################################################
def jaccard_similarity(list1, list2): 
    #Calculates the Jaccard Similarity score of two strings 
    
    #Args: 
        #list1/ list2: list of strings 
        
    #Returns: 
        #A distance measurment between two strings 
        
    #Example: 
        #jaccard_similarity(exampleA, exampleB) ****exampleA= "This is an example" / exampleB= "This is too"
        #returns 7.83 - the lower the score the closer the strings are
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union  


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]
