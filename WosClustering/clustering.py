"""This module contains the Clustering class that takes an input corpus and vocabulary
and provides methods to perform different clustering and visualization techniques."""

# Libraries
import sys

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('default')

# from sklearn.metrics import silhouette_samples
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

class Clustering():
    """
    Apply various clustering techniques using input feature matrix and corpus.
    Current available features:
        k-means clustering
        gaussian mixture models
    """

    # Default class variables
    # None

    def __init__(self, dfCorpus, dfTopics, featMatrix,
                 svd=None, colCorpus='Corpus'):
        """
        Initialize class by loading feature matrix to use.

        Parameters
        ----------
        dfCorpus: pandas dataframe
            dataframe with corpus information including text to use for text analytics.
            Created using Corpus class
        dfTopics: pandas dataframe
            dataframe with final vocabulary to use for text analytics.
            Created using Corpus class
        featMatrix: scipy sparse matrix
            matrix created by Features class to use for clustering similarity scores
        svd: singular value decomposition object
            created from Features class, used to determine keywords in clustering space
            if LSA used to create featMatrix
        colCorpus: string
            column in dfCorpus that defines text used for text analytics
        """

        # Initialize variables
        self.dfCorpus = dfCorpus
        self.dfTopics = dfTopics
        self.featMatrix = featMatrix
        self.colCorpus = colCorpus
        
        # Initialize variables that are created in methods
        self.KMmodel = None # K-means sklearn object
        self.GMmodel = None # Gaussian Mixture sklearn object
        self.tsne = None # TSNE subsampled object for 2D viz
        self.indices = None # random indices used for TSNE

        # Include svd object from LSA calculation in Features if appropriate
        self.svd = svd
        
    def create_tsne(self, subsmpl, inPerp=75, inExag=12, inRnd=1,
                    inLearn=1000, inIter=4000):
        """
        Perform TSNE transformation on subsample of corpus that can be used
        for 2D visualization of clustering in other methods.
        
        Parameters
        ----------
        subsmpl: int
            number of documents to use in TSNE transformation (randomly sampled)
        inPerp: int
            perplexity value for TSNE
        inExag: int
            exaggeration value for TSNE
        inRnd: int
            random state for TSNE
        inLearn: int
            learning rate for TSNE
        inIter: int
            number of iterations for TSNE
            
        Returns
        -------
        None; updates self.tsne and self.indices
        """
        
        # Randomly sample documents to transform (to reduce computational burden)
        self.indices = np.random.randint(len(self.featMatrix), size=subsmpl)
        matrixSub = self.featMatrix[self.indices,:]
        
        # Create TSNE-reduced corpus for plotting
        self.tsne = TSNE(n_components=2, random_state=inRnd, perplexity=inPerp,
                         early_exaggeration=inExag, learning_rate=inLearn,
                         n_iter=inIter).fit_transform(matrixSub)
        
        return None
        
    def k_means_opt(self, minCluster=5, maxCluster=100, interval=5,
                    silFraction=1.0, testSize=0.25, plot=False):
        """
        Perform k-means clustering using test set and increasing cluster number
        to optimize number of clusters based on silhouette scores.

        Parameters
        ----------
        minCluster: int
            minimum number of clusters to test
        maxCluster: int
            maximum number of clusters to test
        interval: int
            interval used to jump cluster numbers
        silFraction: float
            fraction of total samples to use for calculating silhouette score
        testSize: float
            fraction of samples to use in test set

        Returns
        -------
        listInertia: list
            list of inertia scores by cluster number
        listSilScore: list
            list of silhouette scores by cluster number

        """

        # Initialize lists used to calculate scores per cluster size
        listInertia = list()
        listSilScore = list()

        # Split features into training and test sets
        train, test = train_test_split(self.featMatrix, test_size=testSize)
        # Calculate sample size to use for silhouette scores
        silSamples = int(silFraction * test.shape[0])

        # Perform loop over range of cluster numbers provided
        for i in range(minCluster, (maxCluster+1), interval):
            if i % 20 == 0:
                print("Current cluster number: {}".format(i))
            KMmodel = KMeans(n_clusters=i) # create KMeans object
            KMmodel.fit(train) # train model on training set
            labels = KMmodel.predict(test) # predict labels for test set
            listInertia.append(KMmodel.inertia_) # calculate inertia from training
            silScore = metrics.silhouette_score(test, labels, metric='euclidean',
                                                sample_size=silSamples)
            listSilScore.append(silScore)
        
        # Create range of cluster sizes to find max (returns max)
        x = np.arange(minCluster, (maxCluster+1), interval)
        
        # Plot results
        if plot==True:
            fig = plt.figure(figsize=(14, 4))
            ax1 = fig.add_subplot(121)
            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("Inertia Score")
            ax2 = fig.add_subplot(122)
            ax2.set_xlabel("Number of Clusters")
            ax2.set_ylabel("Silhouette Score")
            ax1.plot(x, listInertia)
            ax2.plot(x, listSilScore)

        return x[np.argmax(listSilScore)], listSilScore

    def k_means_clustering(self, numCluster=None, randomState=None):
        """
        Perform k-means clustering to determine categories for each document
        in corpus using predetermined number of clusters.

        Parameters
        ----------
        numCluster: int
            number of clusters for k-means clustering algorithm
        random_state: int
            random state to be entered to reproduce results

        Returns
        -------
        df: pandas dataframe
        Updates self.dfCorpus with k-mean cluster labels
        """

        # If not numCluster provided, make sqrt of length of corpus
        if numCluster is None:
            numCluster = int(np.sqrt(len(self.dfCorpus)))

        # Create K-means object
        self.KMmodel = KMeans(n_clusters=numCluster, random_state=randomState)

        # Fit using input feature matrix data
        self.KMmodel.fit(self.featMatrix)

        # Add cluster labels as new colum in dataframe
        self.dfCorpus['KM Cluster Label'] = self.KMmodel.labels_

        # Create dataframe with keywords associated with each cluster
        df = pd.DataFrame()
        df = self._output_cluster_themes(self.KMmodel.cluster_centers_,
                                         numCluster, colTopic='Phrase',
                                         numWords=10)

        # Add additional column with document count within each cluster
        df['Document Count'] = self.dfCorpus.groupby('KM Cluster Label').count()[self.colCorpus]

        return df

    def gmm_opt(self, minCluster=5, maxCluster=100, interval=5,
                testSize=0.25, covarType='full', plot=False):
        """
        Perform Gaussian mixture modeling using test set and increasing
        cluster number to optimize number of clusters based on AIC/BIC.

        Parameters
        ----------
        minCluster: int
            minimum number of clusters to test
        maxCluster: int
            maximum number of clusters to test
        interval: int
            interval used to jump cluster numbers
        testSize: float
            fraction of samples to use in test set
        covarType: string
            keyword indicating type of covariance matrix to use

        Returns
        -------
        max of listAIC, max of listBIC, listAIC, listBIC

        """

        # Initialize lists used to calculate scores per cluster size
        listAIC = list()
        listBIC = list()

        # Split features into training and test sets
        train, test = train_test_split(self.featMatrix, test_size=testSize)

        # Perform loop over range of cluster numbers provided
        for i in range(minCluster, (maxCluster+1), interval):
            if i % 20 == 0:
                print("Current cluster number: {}".format(i))
            EMmodel = GaussianMixture(n_components=i, covariance_type=covarType,
                                      init_params='kmeans') # create GMM object
            EMmodel.fit(train) # train model on training set
            
            # Calculate AIC/BIC for test set
            listAIC.append(EMmodel.aic(test))
            listBIC.append(EMmodel.bic(test))
        
        # Create range of cluster sizes to find max (returns max)
        x = np.arange(minCluster, (maxCluster+1), interval)
        
        # Plot results for both AIC and BIC
        if plot==True:
            fig = plt.figure(figsize=(14, 4))
            ax1 = fig.add_subplot(121)
            ax1.set_xlabel("Number of Clusters")
            ax1.set_ylabel("AIC")
            ax2 = fig.add_subplot(122)
            ax2.set_xlabel("Number of Clusters")
            ax2.set_ylabel("BIC")
            ax1.plot(x, listAIC)
            ax2.plot(x, listBIC)

        return x[np.argmin(listAIC)], x[np.argmin(listBIC)], listAIC, listBIC
    
    def gmm_clustering(self, numCluster=None, covarType='full', randomState=None):
        """
        Perform Gaussian Mixture clustering to determine categories for each document
        in corpus using predetermined number of clusters.

        Parameters
        ----------
        numCluster: int
            number of clusters for k-means clustering algorithm
        covarType: string
            keyword indicating type of covariance matrix to use
        random_state: int
            random state to be entered to reproduce results

        Returns
        -------
        df: pandas dataframe
        Updates self.dfCorpus with GMM cluster labels
        """

        # If not numCluster provided, make sqrt of length of corpus
        if numCluster is None:
            numCluster = int(np.sqrt(len(self.dfCorpus)))

        # Create GMM object
        self.GMmodel = GaussianMixture(n_components=numCluster, covariance_type=covarType,
                                       random_state=randomState, init_params='kmeans')

        # Fit using input feature matrix data
        self.GMmodel.fit(self.featMatrix)

        # Add cluster labels as new colum in dataframe
        self.dfCorpus['GM Cluster Label'] = self.GMmodel.predict(self.featMatrix).tolist()

        # Create dataframe with keywords associated with each cluster
        df = pd.DataFrame()
        df = self._output_cluster_themes(self.GMmodel.means_,
                                         numCluster, colTopic='Phrase',
                                         numWords=10)

        # Add additional column with document count within each cluster
        df['Document Count'] = self.dfCorpus.groupby('GM Cluster Label').count()[self.colCorpus]

        return df

    def apply_tsne(self, dfThemes, clustType,
                   mSize=1, figSize=(12,12), inXlim=None, inYlim=None, plot=False):
        """
        Add TSNE locations to clusters and plot 2D visualization of
        TSNE-transformed data.
        
        Parameters
        ----------
        matrix: numpy array
            input matrix (e.g. tf-idf) on which to perform TSNE transform
        dfThemes: pandas dataframe
            df with k-means themes created by k_means_clustering
        clustType: string
            'KM' or 'EM' to indicate which column to use for cluster labels
        mSize: float
            scale size of cluster markers
        figSize: tuple
            figure size for matplotlib plotting
        inXlim: tuple of floats
            (left X limit, right X limit)
        inYlim: tuple of floats
            (bottom Y limit, top Y limit)
        plot: boolean
            indicates whether to plot TSNE map
        
        Returns
        -------
        dfTSNEclusters: pandas dataframe
        """
        
        # Merge TSNE coordinates with corpus df and dfThemes
        dfTSNE = pd.DataFrame(self.tsne, index=self.indices, columns=['TSNE_x', 'TSNE_y'])
        f = {'median', 'count'}
        dfTSNEclusters = self.dfCorpus.merge(dfTSNE, how='left',
                                             left_index=True,
                                             right_index=True).groupby(clustType+' Cluster Label').agg(f)
        dfTSNEclusters = dfTSNEclusters.merge(dfThemes, how='inner',
                                              left_index=True, right_index=True)
        
        # Plot subsample of documents and cluster centers with annotation based on theme keywords
        if plot==True:
            fig = plt.figure(figsize=figSize)

            # Plot document subsample using TSNE coordinates
            plt.scatter(self.tsne[:,0], self.tsne[:,1], marker="o", s=8, color='red', alpha=0.3)

            # Plot cluster centers
            plt.scatter(dfTSNEclusters['TSNE_x', 'median'], dfTSNEclusters['TSNE_y', 'median'],
                            marker='o', s=mSize*(dfTSNEclusters['Document Count']),
                            color='blue', alpha = 0.6)

            # Annotate with cluster theme keywords
            for i in dfTSNEclusters.index:
                if dfTSNEclusters['TSNE_x', 'count'].iloc[i] > 0:
                    plt.annotate(str(dfTSNEclusters['Phrase 0'].iloc[i] + '\n' + dfTSNEclusters['Phrase 1'].iloc[i]),
                                 (dfTSNEclusters['TSNE_x', 'median'].iloc[i], dfTSNEclusters['TSNE_y', 'median'].iloc[i]),
                                 size=8, ha='center', va='center', weight='bold')

            # Remove x and y ticks in plot
            if inXlim!=None:
                plt.xlim(left=inXlim[0], right=inXlim[1])
            if inYlim!=None:
                plt.ylim(bottom=inYlim[0], top=inYlim[1])

        return dfTSNEclusters

    #---Private Methods---#

    def _distance(self, x, y, dim):
        '''
        Calculates distance between two vectors with 'dim' dimensions
        '''
        sumD = 0
        for i in range(0, dim):
            sumD = sumD + (x[i] - y[i])**2
        dist = np.sqrt(sumD)

        return dist

    def _output_cluster_themes(self, clusterCenters, numClusters, colTopic='Phrase',
                               numWords=10):
        '''
        Takes output of clustering technique using LSA and returns the top keyword
        associated with each cluster as well as the number of documents in that cluster.
        Adds column to corpus df with cluster identification.

        Parameters
        ----------
        clusterCenters: list
            list of cluster center locations from clustering object
        numClusters: int
            number of clusters used in clustering algorithm
        colTopic: string
            name of column in self.dfTopics to use for term names
        numWords: int
            number of keywords to output in dataframe corresponding to each cluster
        flagLSA: boolean
            indicate whether LSA used for matrix values.  If so, then apply inverse
            transformation on cluster_centers to return to original term space.

        Returns
        -------
        df: pandas dataframe
        '''

        orderCentroids = None

        # If LSA matrix, perform inverse transformation to get original vocab terms
        if self.svd != None:
            try:
                # Reverse transform the LSA to recover vocabulary associated with each cluster
                origCentroids = self.svd.inverse_transform(clusterCenters)

                # Then, return indices that sort array from most common to least
                orderCentroids = origCentroids.argsort()[:, ::-1]
            except AttributeError:
                sys.exit("self.svd does not appear to be correct object type."\
                         "It should be TruncatedSVD object from Features class.")
        else:
            # Just argsort on cluster centers directly
            orderCentroids = clusterCenters.argsort()[:, ::-1]

        # Initialize dataframe with number of columns based on numWords
        columnNames = list()
        for i in range(numWords):
            columnNames.append("Phrase " + str(i))
        df = pd.DataFrame(columns=columnNames)

        # Initialize list of cluster categories to be used for visualization
        cluster_categories = []

        # Loop through clusters and return top words per cluster based on distance to centroid
        for i in range(numClusters):
            tmpString = list()
            for ind in orderCentroids[i, :numWords]:
                tmpString.append(self.dfTopics[colTopic].loc[ind])
            cluster_categories.append(tmpString)
            df = df.append(pd.Series(tmpString, index=columnNames), ignore_index=True)

        return df
