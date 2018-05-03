# Libraries
import sys

import pandas as pd
import numpy as np
import re

import nltk
from nltk.stem.snowball import SnowballStemmer as snowstem
from nltk.stem import WordNetLemmatizer

try:
    import RAKE # (python-rake) - Berry & Kogan, Text Mining: Theory and Applications
except ImportError:
    sys.exit("RAKE module required for vocabulary creation: " \
             "https://pypi.python.org/pypi/python-rake")

from sklearn.feature_extraction.text import CountVectorizer

class Corpus():
    """
    Class that
    1) loads corpus and preprocessing parameters,
    2) creates vocabulary, and
    3) preprocesses corpus using various filtering techniques and input parameters.
    """
    
    # Default class variables
    _stemmer = snowstem("english") # stemmer object used in private methods for vocab
    
    def __init__(self, filepath, colIndex=0, colCorpus='Corpus', inSwType='nltk-english',
                 inSwFile=None, inAddons=None):
        """
        Initialize class and load corpus as pandas dataframe.
        Corpus read in with all default read_csv parameters.
        
        Parameters
        ----------
        filepath: string
            path to csv file containing corpus.
        colIndex: int or sequence
            column index to use as index
        colCorpus: string
            name that will be used for column in df representing corpus
        inSwType: string
            type of stopwords list to use (see _set_stop_majors)
        inSwFile: string
            filename if using custom stopwords list
        inAddons: list
            list of additional stopwords to include
        """
        self.filepath = filepath
        self.colIndex = colIndex
        self.colCorpus = colCorpus
        self.colStem = colCorpus + ' Stemmed' # will create stemmed version of corpus
        self.inSwType = inSwType
        self.inSwFile = inSwFile
        self.inAddons = inAddons
        
        # Read corpus from file
        self.dfCorpus = pd.read_csv(self.filepath, index_col = self.colIndex) # corpus df
        
        # Initialize dataframes
        dfVocab = pd.DataFrame() # initialize vocab dataframe
        dfVocabFreqRej = pd.DataFrame() # initalize freq-rejected terms df
        dfVocabCCRej = pd.DataFrame() # initialize CC-ratio-rejected terms df
        dfVocabCorrRej = pd.DataFrame() # initialize Corr-rejected terms df
        dfVocabTopicRej = pd.DataFrame() # initialize Topic-rejected terms df
        dfTopics = pd.DataFrame() # initialize final topic set df

        # Set stopword list as part of object initialization
        # Note: nltk stopwords is default
        self.stopWords = [] # list of stopwords
        self.stopWordsStem = [] # list of stemmed stopwords
        self._set_stop_majors(self.inSwType, self.inSwFile, self.inAddons)
    
    def show_corpus(self):
        """
        Returns the corpus as a pandas dataframe.
        
        Returns
        -------
        self.dfCorpus
        """
        return self.dfCorpus

    def show_vocabulary(self):
        """
        Returns the vocabulary as a pandas dataframe.
        
        Returns
        -------
        self.dfCorpus
        """
        return self.dfVocab
    
    def create_clustering_text(self, colList, listNoSpace=None, listSpace=None):
        """
        Creates new column that is combination of others columns that will be
        used as text for unsupervised clustering.
        Also perform text cleaning by removing or replacing characters.
        
        Parameters
        ----------
        colList: list of strings
            list of columns that will be joined to create new column.
        listNoSpace: list of regex strings
            strings to replace without space in clustering text.
        listSpace: list of regex strings
            strings to replace with a space in clustering text.
        
        Returns
        -------
        None: updates self.dfCorpus
        """
        # Provide default regex operators to replace characters if none given
        # These defaults have been found to give consistency of words across corpus
        if listNoSpace == None:
            listNoSpace = ["\(", "\)", r",(?=[^\d\s])"]
        if listSpace == None:
            listSpace = ["/", ":", "  ", "-", "–", "−"]
        
        # Combine each column, separated by period, using str.cat to avoid NaN cells
        self.dfCorpus[self.colCorpus] = self.dfCorpus[colList].apply(lambda x: x.str.cat(sep='. '),
                                                                     axis=1)
        
        # Remove symbols from self.colCorpus column, and replace with 0 or 1 whitespace
        for i in range(len(listNoSpace)):
            self.dfCorpus[self.colCorpus] = self.dfCorpus[self.colCorpus].replace({listNoSpace[i]: ""},
                                                                                  regex=True)
        for i in range(len(listSpace)):
            self.dfCorpus[self.colCorpus] = self.dfCorpus[self.colCorpus].replace({listSpace[i]: " "},
                                                                                  regex=True)
            
        # Change '+' to 'plus' since '++' can give regex problems
        self.dfCorpus[self.colCorpus] = self.dfCorpus[self.colCorpus].replace({'\+': 'plus'},
                                                                              regex=True)
        # Change '*' to '' since this doesn't provide meaning and gives regex problems
        self.dfCorpus[self.colCorpus] = self.dfCorpus[self.colCorpus].replace({'\*': ''},
                                                                              regex=True)
        
        # Remove all uppercase letters because current vocabulary creation doesn't consider case
        self.dfCorpus[self.colCorpus] = self.dfCorpus[self.colCorpus].apply(lambda x: x.lower())
        
        # Create stemmed version of corpus
        self.dfCorpus[self.colStem] = self.dfCorpus[self.colCorpus].apply(lambda x:
                                                                          self._stemmer_multi_word(x))
        
        return None
    
    def create_vocabulary(self, vocabType='word', scoreLower=-10000, scoreUpper=10000):
        """
        Creates vocabulary from corpus using either single words or words+RAKE for phrases.
        
        Parameters
        ----------
        vocabType: string
            keyword to indicate what type of vocabulary to create.
            Current options: 'word'; 'word_RAKE'
        scoreLower: integer
            Lower bound for RAKE scores
        scoreUpper: integer
            Upper bound for RAKE scores
        """
        
        # Create word-based vocabulary
        vocabWords = self._tokenize_vocab_no_stem()
            
        # Remove duplicates
        vocabWords = vocabWords.drop_duplicates(keep='first')
        vocabWords.reset_index(drop=True, inplace=True)
        
        #Create dataframe that can connect to RAKE dataframe
        dfVocabWords = pd.DataFrame(vocabWords, columns=['Phrase'])
        #Create 'Count' column with all values of 1 (since all single word)
        dfVocabWords['Score'] = 1
        
        print("Completed adding individual words...")
        
        if vocabType == 'word':
            self.dfVocab = dfVocabWords
            
        elif vocabType == 'word_RAKE':
            #Check that scoreLower and scoreUpper are integers
            if (type(scoreLower) or type(scoreUpper)) != int:
                sys.exit("Error: scoreLower and scoreUpper must be provided "\
                         "as integers to use RAKE.")
            
            # Save stopwords to file so RAKE can read it
            outFile = open('RAKE_stopwords.txt', 'w')
            for item in self.stopWords:
                outFile.write("%s\n" % item)
            outFile.close()
            
            # Create RAKE-based vocabulary for ith document in corpus
            dictRAKE = []
            Rake = RAKE.Rake('RAKE_stopwords.txt')
            for i in range(len(self.dfCorpus)):
                docRAKE = Rake.run(self.dfCorpus[self.colCorpus].iloc[i])
                dictRAKE.extend(docRAKE) 
            
            #Put RAKE dictionary into dataframe and remove score outliers
            dfRAKE = pd.DataFrame({'Phrase':[row[0] for row in dictRAKE],
                                   'Score' : [row[1] for row in dictRAKE]})
            dfRAKE = dfRAKE[(dfRAKE['Score'] > scoreLower) &
                            (dfRAKE['Score'] < scoreUpper)]
            
            #Remove duplicates
            dfRAKE = dfRAKE.drop_duplicates(subset='Phrase', keep='first')       
            
            self.dfVocab = dfVocabWords.append(dfRAKE)
            
            print("Completed adding RAKE phrases...")
            
        else:
            sys.exit("Error: incorrect vocabType chosen!")
        
        # Apply stemming to final vocabulary
        self.dfVocab['Stem'] = self.dfVocab['Phrase'].apply(lambda x : self._stemmer_multi_word(x))
        
        #Remove duplicates of final dataframe based on stemmed phrases
        self.dfVocab = self.dfVocab.drop_duplicates(subset='Stem', keep='first')
        print("Completed creating stemmed version of vocabulary...")
        
        #Remove stopmajors
        for i in range(len(self.stopWordsStem)):
            self.dfVocab = self.dfVocab[self.dfVocab['Stem'] != self.stopWordsStem[i]]
        print("Removed all stopwords...")
            
        #Reset indices on vocabulary series
        self.dfVocab.reset_index(drop=True, inplace=True)
        
        return None
    
    def apply_freq_filter(self, colVocab='Stem', minPhrase=1, maxPhrase=5,
                          minFreq=5, maxFreq=0.8):
        """
        Apply minimum and maximum filter frequencies to reduce vocabulary size.
        Note: stopwords not used because already applied in create_vocabulary.
        Note: minFreq only applied using raw number, maxFreq can be either
        integer value of float (percent).
        
        Parameters
        ----------
        colVocab: string
            column name from self.dfVocab to use for count
        minPhrase: int
            minimum ngram size to include from vocabulary to use for filter
        maxPhrase: int
            maximum ngram size to include from vocabulary to use for filter
        minFreq: int
            minimum frequency count for vocabulary term threshold
        maxFreq: int or float
            maximum frequency count (int) or fraction (float) threshold.
            If float, used as percent of docs that include AT LEAST ONE
            mention of the term.
        
        Returns
        -------
        Updates dfVocab
        Updates dfVocabFreqRej
        self.frequencyParam: dictionary of threshold parameters used
        """
        
        # Perform frequency count of vocabulary
        try:
            CountVect = CountVectorizer(stop_words=None, analyzer="word",
                                        ngram_range=(minPhrase, maxPhrase),
                                        vocabulary=self.dfVocab[colVocab],
                                        tokenizer=self._tokenize_custom_and_stem,
                                        encoding='utf-8')
        except KeyError:
            sys.exit("Vocabulary not created yet or column name cannot be found." \
                     "Try running create_vocabulary first!")
            
        matCount = CountVect.fit_transform(self.dfCorpus[self.colCorpus])
                
        # Calculate summed frequency for all vocab elements across documents
        self.dfVocab['Frequency'] = np.ravel(matCount.sum(axis=0))
    
        # Set flag to one if within filter threshold
        # if using integer max frequency, just apply it was upper threshold
        self.dfVocab['Freq Flag'] = 0
        if type(maxFreq)==int:
            self.dfVocab['Freq Flag'] = np.where((self.dfVocab['Frequency'] > minFreq) &\
                                                 (self.dfVocab['Frequency'] < maxFreq), 1, 0)
        elif type(maxFreq)==float: 
            # if using float, get fraction of docs in which term appears
            self.dfVocab['Fraction'] = matCount.getnnz(axis=0) / len(self.dfCorpus)
            # Include only those terms below fraction and above min frequency
            self.dfVocab['Freq Flag'] = np.where((self.dfVocab['Frequency'] > minFreq) &\
                                                 (self.dfVocab['Fraction'] < maxFreq), 1, 0)
        else:
            sys.exit("Incorrect data type for maximum frequency threshold!")
            
        # Only keep those terms that pass threshold
        self.dfVocabFreqRej = self.dfVocab[self.dfVocab['Freq Flag'] == 0]
        self.dfVocab = self.dfVocab[self.dfVocab['Freq Flag'] == 1]

        self.dfVocab.reset_index(drop=True, inplace=True)

        # Create dictionary of frequency parameters used
        self.frequencyParam = {'Min Freq': minFreq, 'Max Freq': maxFreq,
                               'Min Phrase': minPhrase, 'Max Phrase': maxPhrase,
                              }
        
        print("Completed applying frequency filter...")
        
        return None
    
    def apply_cluster_filter(self, colVocab='Stem', minPhrase=1, maxPhrase=5,
                             phasePref=1.0, threshold=0.9):
        '''
        Apply topicality filter by calculating the condensation cluster strength.
        Described in Bookstein et al.
        A low number indicates the term occurs less frequently than expected
        based on a random distribution.  This indicates it is a special topic
        to be used as a topic term.
        Note: Assumes self.dfVocabFreqIn has column names 'Phrase' with each vocab term.
        
        Parameters
        ----------
        colVocab: string
            column name from self.dfVocab to use for count
        minPhrase: int
            minimum ngram size to include from vocabulary to use for filter
        maxPhrase: int
            maximum ngram size to include from vocabulary to use for filter
        phasePref: float
            divide CC ratio by this value to prefer multi-word phrases
        threshold: float
            vocabulary limited to terms below this CC ratio threshold
            
        Returns
        -------
        Updates dfVocab
        Updates dfVocabCCRej
        '''
        
        # Calculate doc-term matrix
        try:
            CountVect = CountVectorizer(stop_words=None, analyzer="word",
                                        ngram_range=(minPhrase, maxPhrase),
                                        vocabulary=self.dfVocab[colVocab],
                                        tokenizer=self._tokenize_custom_and_stem,
                                        encoding='utf-8')
        except KeyError:
            sys.exit("Vocabulary not created yet or column name cannot be found." \
                     "Try running create_vocabulary first!")
        
        matCount = CountVect.fit_transform(self.dfCorpus[self.colCorpus])
        
        # Calculate scaling factors to be used to calculate CC ratio
        corpusLengths = self.dfCorpus[self.colCorpus].str.split().apply(lambda x:len(x))
        meanLength = corpusLengths.mean()
        corpusScaling = corpusLengths / meanLength
        
        # Scale frequencies by normalization value
        countMatScaled = matCount.T.multiply((1/corpusScaling))
        self.dfVocab['Norm Frequency'] = countMatScaled.sum(axis=1)

        # CC Ratio calculation loop
        self.dfVocab['CC Ratio'] = np.nan
        self.dfVocab['CC Ratio'] = self.dfVocab.apply(lambda row: self._cc_calc(row[colVocab],
                                                                                row['Norm Frequency'], 
                                                                                phasePref), axis=1)

        # Update vocab with accepted and rejected vocabulary terms
        self.dfVocabCCRej = self.dfVocab[self.dfVocab['CC Ratio'] >= threshold]
        self.dfVocab = self.dfVocab[self.dfVocab['CC Ratio'] < threshold]        
        self.dfVocab.reset_index(drop=True, inplace=True)

        # Create dictionary of frequency parameters used
        self.CCParam = {'Min Phrase': minPhrase, 'Max Phrase': maxPhrase,
                        'Phase Preference': phasePref, 'Threshold': threshold
                       }
        
        print("Completed applying cluster filter...")
        
        return None

    def apply_correlation_filter(self, colVocab='Stem', minPhrase=1, maxPhrase=5,
                                 corrThreshold=0.7, colCC='CC Ratio'):
        '''
        Calculates correlation between all vocabulary terms and filters out
        those with correlation above threshold  Keeps term with lower
        clustering coefficient (CC).  Each vector being correlated is the
        number of occurrences in each document in corpus.
        
        Parameters
        ----------
        colVocab: string
            column name from self.dfVocab to use for count
        minPhrase: int
            minimum ngram size to include from vocabulary to use for filter
        maxPhrase: int
            maximum ngram size to include from vocabulary to use for filter
        corrThreshold: float
            Threshold above which one of the word pair is removed based on CC ratio
        colCC: string
            column name with CC Ratio information in dfVocab
        
        Returns
        -------
        Updates dfVocab
        Updates dfVocabCorrRej
        '''
        # Check that data in colCC contains floats
        try:
            if self.dfVocab[colCC].dtype == 'O':
                sys.exit("colCC does not appear to contain float or int values!")
        except KeyError:
            sys.exit("colCC cannot be found.  Run apply_cluster_filter first.")
        
        # Initialize outputs
        corrMaxList = [] # list of term indices to remove
        corrMinList = [] # list of term indices to keep
        self.dfVocab['Corr Flag'] = 1 # Initialize: 1 will keep term, 0 will remove

        # Calculate doc-term matrix
        try:
            CountVect = CountVectorizer(stop_words=None, analyzer="word",
                                        ngram_range=(minPhrase, maxPhrase),
                                        vocabulary=self.dfVocab[colVocab],
                                        tokenizer=self._tokenize_custom_and_stem,
                                        encoding='utf-8')
        except KeyError:
            sys.exit("Vocabulary not created yet or column name cannot be found." \
                     "Try running create_vocabulary first!")
        
        matCount = CountVect.fit_transform(self.dfCorpus[self.colCorpus])
        
        # Calculate scaling factors to be used to calculate CC ratio
        corpusLengths = self.dfCorpus[self.colCorpus].str.split().apply(lambda x:len(x))
        meanLength = corpusLengths.mean()
        corpusScaling = corpusLengths / meanLength
        
        # Normalize frequenies to document length (tranpose so rows are terms)
        matCountScaled = matCount.T.multiply((1/corpusScaling))

        # Calculate correlation coefficient matrix for all terms
        matCorr = self._sparse_corr_coef(matCountScaled)

        # Find indices for those phrases with correlation above threshold
        for i in range(len(matCorr)):
            # Find indices for one vocab phrase for all others correlated
            indices = np.where(np.greater(matCorr[i][:], corrThreshold))[1]
            for j in indices:
                # Do not include correlation of phrase with itself
                if j != i:
                    # For each index, append to min or max list based on which has lower CC ratio
                    try:
                        corrMinList.append(np.argmin(self.dfVocab[colCC].iloc[[i,j]]))
                        corrMaxList.append(np.argmax(self.dfVocab[colCC].iloc[[i,j]]))
                    except KeyError:
                        sys.exit("colCC column not created! Use apply_cluster_filter first.")

        corrMaxNoDup = list(set(corrMaxList)) # remove duplicates 
        
        # Set Corr Flag to 0 for relevant indices
        for i in range(len(corrMaxNoDup)): 
            self.dfVocab['Corr Flag'].iloc[corrMaxNoDup[i]] = 0

        # Update vocabulary and separate rejected terms
        self.dfVocabCorrRej = self.dfVocab[self.dfVocab['Corr Flag'] == 0]
        self.dfVocab = self.dfVocab[self.dfVocab['Corr Flag'] == 1]

        self.dfVocab.reset_index(drop=True, inplace=True)
        
        print("Completed applying correlation filter...")

        return None

    def apply_topic_threshold(self, colCC='CC Ratio', colIndex='Stem',
                              threshold=2000):
        '''
        Applies threshold to vocabulary to keep only terms with
        lowest clustering ratio (by default).  Change colCC to apply
        threshold to a different column in dfVocab (e.g., frequency).
        
        Parameters
        ----------
        colCC: string
            column name with CC Ratio information in dfVocab,
            or another column with value to threshold.
        threshold: int
            number of vocabulary terms to keep by applying threshold
        
        Returns
        -------
        Updates dfVocab
        Updates dfVocabTopicRej
        '''
        
        # Check that data in colCC contains floats
        try:
            if self.dfVocab[colCC].dtype == 'O':
                sys.exit("colCC does not appear to contain int or float values!")
        except KeyError:
            sys.exit("colCC cannot be found.  Run apply_cluster_filter first.")
        
        # Sort vocabulary by CC ratio and cut at threshold
        self.dfTopics = self.dfVocab.sort_values(colCC, ascending=True).iloc[0:int(threshold)]
        self.dfTopics.reset_index(drop=True, inplace=True)
        
        # Set column to use as index ('Stem' by default)
        # try:
        #    self.dfTopics.set_index([colIndex], inplace=True)
        #except KeyError:
        #    sys.exit("Cannot find column to use as vocabulary index!")
        
        return None
    
    def column_search_row(self, colName, searchTerm):
        """
        Return rows that contain search string in input column.
        If search term is multiple words, will return documents where both words
            appear but not necessairly in sequential order.
        
        Parameters
        ----------
        colName: string
            Column over which to search for string.
        searchTerm: string
            Terms to search for in column.  Terms separated by space are
            searched separately.
        
        Returns
        -------
        Dataframe with selected rows
        """
    
        #First, create base mask filled with Trues (boolean type so & boolean works properly)
        mask = pd.Series(index=self.dfCorpus[colName].index)
        mask = mask.fillna(True).astype(bool)
    
        #Loop through each word in search term, multiply mask by the tmpMask created for each word
        for i in range(len(searchTerm.split())):
            tmpMask = self.dfCorpus[colName].str.contains(searchTerm.split()[i], case=False)
            mask = mask & tmpMask
        
        return self.dfCorpus.loc[mask]
    
    def column_search_index(self, colName, searchTerm):
        """
        Returns indices of all rows that contain search string.
        If search term is multiple words, will return documents where both words
            appear but not necessarily in sequential order.
            
        Parameters
        ----------
        colName: string
            Column over which to search for string.
        searchTerm: string
            Terms to search for in column.  Terms separated by space are
            searched separately.
        
        Returns
        -------
        Dataframe with selected rows
        """
        
        #First, create base mask filled with Trues
        mask = pd.Series(index=self.dfCorpus[colName].index)
        mask = mask.fillna(True).astype(bool)
    
        #Multiply mask by the tmpMask created for each word
        for i in range(len(searchTerm.split())):
            tmpMask = self.dfCorpus[colName].str.contains(searchTerm.split()[i],
                                                          case=False)
            mask = mask & tmpMask
        
        return self.dfCorpus.loc[mask].index.tolist()
    
    def load_vocabulary_from_file(self, filepath, inIndex=0):
        """
        Loads pre-existing vocabulary from file.
        
        Parameters
        ----------
        filepath: string
            path to vocabulary csv file
            
        Returns
        -------
        None; updates dfVocab
        """
    
        self.dfVocab = pd.read_csv(filepath, index_col = inIndex)
        
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
    
    def _tokenize(self, text):
        """
        Tokenize input text and return list of words without hyphens
        
        Parameters
        ----------
        text: string
            input text to tokenize
            
        Returns
        -------
        filtered
        """
      
        tokens = nltk.word_tokenize(text)
        filtered = [token.lower().replace('-', ' ') for token in tokens
                    if re.search('[a-zA-Z]', token) is not None]
        return filtered
    
    def _tokenize_vocab_no_stem(self):
        """
        Run over corpus and create vocabulary without stemming.
        
        Parameters:
        
        Returns:
        sVocab: pd Series
        """
        
        # Initialize list that will be filled with vocabulary terms
        vocabTokenized = []
    
        # Fill Series with vocabulary words from each input text
        for i in range(len(self.dfCorpus)):
            allWordsTokenized = self._tokenize(self.dfCorpus[self.colCorpus].iloc[i])
            vocabTokenized.extend(allWordsTokenized)
    
        sVocab = pd.Series(vocabTokenized)
    
        #Remove duplicates
        sVocab.drop_duplicates(inplace=True)
    
        #Reset indices for stemmed and tokenized vocab lists
        sVocab.reset_index(drop=True, inplace=True)
  
        return sVocab

    def _stemmer_multi_word(self, inString):
        """
        Applies tokenizing and stemming and then joins tokens together into one string.
        
        Parameters
        ----------
        inString: string
        
        Returns
        -------
        string: joined tokens
        """
        
        stems = self._tokenize_custom_and_stem(inString)
        return ' '.join(stems)
    
    def _set_stop_majors(self, swType, swFile=None, addons=None):
        """
        Set stopword list using NLTK default or custom list.
        Available stopword lists ('type' parameter):
            1) 'nltk-english': standard nltk english word list
            2) 'inspire': standard INSPIRE stopwords from PNNL
            
        Parameters
        ----------
        smType: string
            string name for one of stopwords options above
        swFile: string
            filepath to text file with stopwords separated by '\n'
        addons: list
            list of strings with additional custom stopwords to add

        Returns
        -------
        None: updates self stopword lists
        """
        
        if swType == 'nltk-english':
            try:
                self.stopWords = nltk.corpus.stopwords.words('english')
            except LookupError:
                try:
                    nltk.download('stopwords')
                except:
                    sys.exit("Error: Not able to download stopwords from nltk!")
                self.stopWords = nltk.corpus.stopwords.words('english')
            
        elif swType == 'custom':
            try:
                self.stopWords = open(swFile).read().split('\n')
            except FileNotFoundError:
                sys.exit("Error: File with custom stopwords not found!")
        else:
            sys.exit("Error: Incorrect stopmajors type chosen!")
    
        # Add any additional stopwords to original list
        if addons != None:
            self.stopWords.extend(addons)

        # Create stemmed version of stopmajors list
        self.stopWords = list(set(self.stopWords))
        self.stopWordsStem = list(set([self._stemmer.stem(t) for t in self.stopWords]))
        
        return None
    
    def _cc_calc(self, word, freq, phasePrefer):
        '''
        Calculates cluster condensation ratio (used in apply_cluster_filter)
        
        Parameters
        ----------
        word: string
            vocabulary term for which CC ratio calculated
        freq: float
            normalized frequency value to calculate CC ratio
        columnIn: string
            column name from dfVocab used to calculate CC ratio
        phasePrefer: float
            number to divide CC if multiword phrase
            
        Returns
        -------
        CCratio: float
        '''
        
        # Define length of corpus
        nCorpus = len(self.dfCorpus)

        # Calculate CC ratio
        nOccWord = freq # normalized frequency of occurrence
        try:
            # Calculate number of documents in which the term appears
            nOccDoc = len(self.dfCorpus.loc[self.dfCorpus[self.colStem].str.contains(word, case=False)])
        except KeyError:
            sys.exist("Cannot find stemmed vocabulary column. Run create_clustering_text first!")
        
        # Actual number of textual units containing word i (multiple possible per document)
        CCnum = nOccDoc
        
        # Expected number of textual units containing word i 
        CCden = nCorpus * (1 - ((1 - (1/nCorpus))**nOccWord))
        CCratio = CCnum / CCden
        
        # Use Phase preference do divide cluster ratio by value for multiword vocabulary
        if (len(word.split()) > 1):
            CCratio = CCratio / phasePrefer

        return CCratio
    
    def _sparse_corr_coef(self, A):
        '''
        Caculates correlation matrix for sparse scipiy matrix.
        
        Parameters
        ----------
        A: sparse scipy matrix - if a x b, returns axa matrix
        
        Returns
        -------
        coeffs: numpy matrix
        '''
        A = A.astype(np.float64)
        n = A.shape[1]

        # Compute the covariance matrix
        rowsum = A.sum(1)
        centering = rowsum.dot(rowsum.T.conjugate()) / n
        C = (A.dot(A.T.conjugate()) - centering) / (n - 1)

        # The correlation coefficients are given by
        # C_{i,j} / sqrt(C_{i} * C_{j})
        d = np.diag(C)
        coeffs = C / np.sqrt(np.outer(d, d))

        return coeffs
