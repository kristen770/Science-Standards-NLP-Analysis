#load the expanded ngs standards pdf into a txt file
def pdf_to_text(filepath, filename):  
    """takes in a local PDF file path and saves it as a local txt.file"""
    import pdftotext
    with open(filepath, "rb") as f:
        pdf = pdftotext.PDF(f)
 
    # Save all text to a txt file.
    with open(filename, 'w') as f:
        f.write("\n\n".join(pdf))  


#openfile and flatten into single list item
def open_and_flatten(filename): 
    """takes in a local txt file path and returns a flattened file"""
    file = open(filename) 
    yourResult = [line.split(',') for line in file.readlines()] 
    flat_text = [item for sublist in yourResult for item in sublist] 
    return flat_text 


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
                       'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'questions', 'science', 'st']
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
    pmi_finder = BigramCollocationFinder.from_words(list_of_words) 
    pmi_finder.apply_freq_filter(freq_filter)  
    pmi_scored = pmi_finder.score_ngrams(bigram_measures.pmi) 
    return ngss_pmi_scored 

def word_cloud(word_list): 
    """Input: A list of words 
    Output: a word cloud"""
    unique_string=(" ").join(word_list)
    wordcloud = WordCloud(width = 1000, height = 500, max_font_size=50, max_words=100,
                      background_color="white", colormap="gray_r").generate(unique_string)
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