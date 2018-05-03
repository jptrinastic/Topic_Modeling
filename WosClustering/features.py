# Libraries
import sys

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('default')

import re

import nltk
from nltk.stem.snowball import SnowballStemmer as snowstem
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

class Features():
    """
    Initializes with input corpus and provides options for creating features
    to use for clustering.
    Current available features:
        counts (apply_counts)
        tf (apply_tfidf, idf=False)
        tfidf (apply_tfidf, idf=True)
        latent semantic analysis (apply_lsa)
    """
    
    # Default class variables
    _stemmer = snowstem("english") # stemmer object used in private methods for vocab
    
    def __init__(self, dfCorpus, dfTopics, colCorpus='Corpus'):
        """
        Initialize class by loading corpus and vocabulary to use.
        
        Parameters
        ----------
        dfCorpus: pandas dataframe
            dataframe with corpus information including text to use for text analytics.
            Created using Corpus class
        dfTopics: pandas dataframe
            dataframe with final vocabulary to use for text analytics.
            Created using Corpus class, apply_topic_threshold method.
        colCorpus: string
            column name in dfCorpus with text for text analytics
        """
        
        # Initialize variables
        self.dfCorpus = dfCorpus
        self.dfTopics = dfTopics
        self.colCorpus = colCorpus
        self.matrix = None
        
        if colCorpus not in self.dfCorpus.columns:
            sys.exit("Cannot find column to be used for text analytics."\
                      "Try applying create_clustering_text in Corpus class.")         
            
    def apply_count(self, minPhrase=0, maxPhrase=5):
        """
        Apply count vectorizer to corpus using initialized vocabulary.
        
        Parameters
        ----------
        minPhrase: int
            minimum ngram size to include from vocabulary to use for filter
        maxPhrase: int
            maximum ngram size to include from vocabulary to use for filter
        
        Returns
        -------
        updates self.matrix, scipy sparse matrix
        """
        
        # Create tfidf_vectorizer object using feature-reduced vocabulary from above
        try:
            countVect = CountVectorizer(max_df=0.0, min_df=0.0, max_features=None,
                                    analyzer='word', stop_words=None,
                                    encoding='utf-8',
                                    vocabulary = self.dfTopics['Stem'],
                                    tokenizer=self._tokenize_custom_and_stem,
                                    ngram_range=(minPhrase,maxPhrase), binary=False)
        except KeyError:
            sys.exit("Column name for vocabulary cannot be found.")

        # Fit data to vectorizer and output shape
        self.matrix = countVect.fit_transform(self.dfCorpus[self.colCorpus])
        
        return None
            
    def apply_tfidf(self, useIdf=True, inNorm='l2', minPhrase=0, maxPhrase=5):
        """
        Apply tfidf to corpus using initialized vocabulary.
        
        Parameters
        ----------
        useIdf: boolean
            use inverse doc frequency in tfidfvectorizer object
        minPhrase: int
            minimum ngram size to include from vocabulary to use for filter
        maxPhrase: int
            maximum ngram size to include from vocabulary to use for filter
        
        Returns
        -------
        updates self.matrix, scipy sparse matrix
        """
        
        # Create tfidf_vectorizer object using feature-reduced vocabulary from above
        try:
            tfidfVect = TfidfVectorizer(max_df=0.0, min_df=0.0, max_features=None,
                                    analyzer='word', stop_words=None, use_idf=useIdf,
                                    norm=inNorm, encoding='utf-8',
                                    vocabulary = self.dfTopics['Stem'],
                                    tokenizer=self._tokenize_custom_and_stem,
                                    ngram_range=(minPhrase,maxPhrase), binary=False)
        except KeyError:
            sys.exit("Column name for vocabulary cannot be found.")
            
        # Fit data to vectorizer and output shape
        self.matrix = tfidfVect.fit_transform(self.dfCorpus[self.colCorpus])
        
        return None

    def plot_scree(self, nEig):
        """
        Output Scree plot to visualize eigenvalues of scoring matrix (e.g. tf-idf).
        Can be used to identify 'knee' of plot to identify cutoff for number of
        eigenvalues to use from latent semantic analysis.
        
        Parameters
        ----------
        mat: sparse numpy matrix
            Input matrix with scoring values on which to perform LSA.
        nEig: integer
            Top number of eigenvalues to compute using SVD.
        
        Returns
        -------
        None: creates plot using matplotlib
        """

        # Check for NaNs in matrix to be sure tfidf or count calculated first
        if self.matrix.dtype != 'float64':
            sys.exit("Input matrix does not contain correct data type."\
                     "Make sure you have applied tfidf or count first.")
        
        # Calculate U, S, and V matrices by solving eigenvalue equation
        U, S, V = svds(self.matrix, k=nEig) 
        
        # Calculate eigenvalue magnitudes from S matrix
        eigVals = S**2 / np.cumsum(S)[-1]
        
        # Scree plot to identify 'knee' of curve to limit component number
        fig = plt.figure(figsize=(8,5))
        singVals = np.arange(0, len(eigVals), 1)
        plt.plot(singVals, eigVals, 'ro-', linewidth=2)
        plt.title('Scree plot to visualize eigenvector importance')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                         shadow=False,
                         prop=mpl.font_manager.FontProperties(size='large'),
                         markerscale=1.0)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)
        plt.show()
        
        return None
    
    def apply_lsa(self, nComponents):
        """
        Perform latent semantic analysis (LSA) using input matrix (e.g. tf-idf).
        
        Parameters
        ----------
        mat: sparse numpy matrix
            Input matrix with scoring values on which to perform LSA.
            Could be tf, tf-idf, or count values, for exmample.
        nComponents: integer
            Top number of eigenvalues to compute using SVD.
        
        Returns
        -------
        None; updates self.matLSA
        """

        # Check for NaNs in matrix to be sure tfidf or count calculated first
        if self.matrix.dtype != 'float64':
            sys.exit("Input matrix does not contain correct data type."\
                     "Make sure you have applied tfidf or count first.")
        
        self.svd = TruncatedSVD(n_components = nComponents)

        # Fit and transform the input matrix created above
        lsa = self.svd.fit_transform(self.matrix)
        
        # Normalize LSA results
        normalizer = Normalizer(copy=False, norm='l2')
        self.matLSA = normalizer.transform(lsa)
      
        # Return the amount of variance in input data explained by eigenvectors
        print("Variance Explained: {}".format(self.svd.explained_variance_.sum()))
              
        return None
    
    #---Private Methods---#
    
    def _stemmer_custom(self, tokens, stemmer):
        """
        Apply stemming in conjunction with TokenizeCustom below
        
        Parameters
        ----------
        tokens: list
            list of strings tokenized from corpus
        stemmer: stemmer object
        
        Returns
        -------
        stems
        """
        
        stems = [stemmer.stem(item) for item in tokens]
        
        return stems
    
    def _tokenize_custom_and_stem(self, text):
        """
        Tokenize input text after lowering case and replacing hyphens
        
        Parameters
        ----------
        text: string
        
        Returns
        -------
        stems
        """
        
        tokens = nltk.word_tokenize(text)
        filtered = [token.lower().replace('-', ' ') for token in tokens
                    if re.search('[a-zA-Z]', token) is not None]
        stems = self._stemmer_custom(filtered, self._stemmer)
        return stems
        
        
        