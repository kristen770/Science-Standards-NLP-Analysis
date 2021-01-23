# <p align="center"> Correlating NGS and State Based Science Standards <p align="center">

<p align="center">
  <img width="300" height="150" src="/Images/NGSS.png">
<p align="center">
 
 [](/Images/NGSS.png) 
 
### <p align="center"> Capstone Project - The Flatiron School - By Kristen Davis <p align="center">

#### Summary:  
In April of 2013 a collection of rigorous, and internationally benchmarked standards for K-12 science education standards were released called [Next Generation Science Standards (NGS)](https://www.nextgenscience.org/) . These standards were crafted to prepare students to be better decision makers about scientific and technical issues and to apply science to their daily lives. By blending core science knowledge with scientific practices, students are engaged in a more relevant context that deepens their understanding and helps them build what they need to move forward with their education. However, these standards were of voluntary adoption at the time and many states chose not to change their current (common core) standards.

Currently, 18 states have fully adopted the NGSS for their K -12 science curriculum, 26 are 'aligned' to the NGS standards and eight have independently developed standards. Each of these states deployed teams of industry experts to spend months to achieve this alignment. Much qualitative work is done to measure alignment, yet little quantitative work has been applied to understand alignment. With the surge in tools provided in the Natural Language Processing package, the idea that an organization could quantify its alignment is within reach. By identifying word frequencies and text patterns in the NGSS standards and comparing them to state standards, this project aims to do just that, providing not only insight into the similarities and differences of science education across America, but also develop a tool that could be used more broadly to quantify alignment cross industry. 


### Data: 
31 K-12 science standards documents. The states that have fully adopted the NGSS standards do not have unique documents for comparison and thus are represented in the NGSS standards exploration and examination. 

These documents contain specific langauge about content standards, as well as writing on overaching goals and themes the standards wish to highlight. These documents shed light into not only what content a state's Department of Education deems important but also how it talks (and thus thinks) about itself.  

### Project Progression 

#### 1. NGSS Standard Exploration & Sentiment Analysis 
* A full processing and cleaning of the NGSS corpus using custom built pdf to txt functions
* Word frequency analysis, wordcloud generation, production of bigram and pmi pairs
* Intra text analysis of the top 5 highest frequency words most highly correlated words and visualizations of the text clusters 
* KMeans clustering of intra text vocabulary to identify centriod features 
* Supervised Learning classification of standard preformance statements using cross validation and accuracy to select best fit models 
* Identification & visulaization of feature importance (content words) for content based classification of standards  

#### 2. State Standard Exploration & Sentiment Analysis 
* A full processing and cleaning of each standards corpus using list comprehension and custom built pdf to txt function 
* Word frequency analysis, wordcloud generation, production of bigram and pmi pairs
* Intra text analysis of the top 5 highest frequency words from two example state corpus most highly correlated words and visualizations of the text clusters   

#### 3. Comparing NGSS and State Standards 
* Graphed the age of standards & geographical patterns based on state affilations(adopted/aligned/independent)
* Compared highest frequency words across all corpus, identify a unify theme of 'evidence' regarless of affiliation
* Used KMeans & hierachical clustering to group inter text relationships and association patterns & generate label for future modeling
* Applied the Scattertext library to highlight corpus unique words within each state cluster 

#### 4. Qunatify Alignment Between Standards
* Used Gensim modeling to calculate the cosine distance between each text corpus and the NGSS corpus 
* Applied Word Mover's Distance with Word2Vec to calculate distance between each text corpus and the NGSS corpus  
* Generated levenshtein distances between each text corpus and the NGSS corpus with the Fuzzy Wuzzy libraries' ratio score
* Calculated string matching scores from the py_stringmatching library for the following distance measures: Monge Elkan, Levenshtein, Jaccard, & Bag  
* Used z-score normalization to create a correlation score for each state to the NGSS corpus, thus quantifying alignment

#### Conclusion 
Natural Langauge Processing, Clustering and String Similarity scoring allow for effective quantification of the term 'alignment'. Once a quantification for the level of alignment two documents have is made it becomes easier to make data driven recommendations for actions. For example, a state like Alaska with an overall alignment score of 60% in their own aligned standards to the NGSS standards could choose to simply adopt the standards completely. Thus saving internal expense at updating and revising standards on a cyclical basis and gaining the benefits (both monetary and reputational) of being a fully adopted state. 

Conversely, the NGSS, a highly regarded organization with a vested interest in maintaining that respect could do an audit of those states claiming to be aligned to their values (thus representing the NGSS through their writing) and look at a state like Idaho with a very low alignment score of 9% and re-examine if it is in the best interest of the organization to allow the state to continue to claim to be aligned with them. 

The implications of such a tool stretch far beyond state standards and could be applied to any organization seeking to become 'in line with' another organization or philosophy, and for those organizations to better understand the text associated groups are using in association with their name. 

#### Future Work 
The are many more string distance metrics and NLP text comparison techniques that could be explored and add to an alignment score. Specifcially the py.stringmatching library has a plethera of distance measures that could be computed and added (some would need to be done on cloud processing) and deep nlp with neural netword such as [deep siamese test similarity](https://github.com/dhwajraj/deep-siamese-text-similarity) would add further insight into intra text analysis. Finally, the data used was particularly messy, and while extensive cleaning was done more could be applied. 


 #### Blog Post  
 [Word Cloud Colormap Visualization](https://kristendavis27.medium.com/wordcloud-style-guide-2f348a03a7f8)


#### Sources: 
* [NLP Town](https://github.com/nlptown/nlp-notebooks/blob/master/An%20Introduction%20to%20Word%20Embeddings.ipynb) 
* [Multi Class Text Classification with Scikit Learn](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f) 
* [Text Classification](http://brandonrose.org/clustering) 
* [Fuzzy Wuzzy Matching](https://medium.com/@jmcneilkeller/text-matching-with-fuzzywuzzy-6600eb32c530) 

#### Slide Deck: 
[Non Technical Presentation](https://docs.google.com/presentation/d/1ZEddVTqV8INCNWEz4m1Z8AlpPvOfOuxdUc9-AXd3gZ4/edit#slide=id.gb4150ccf36_0_5131)
