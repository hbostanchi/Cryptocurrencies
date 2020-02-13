# Cryptocurrencies

## Overview
This week we’ll dive deeper into machine learning using unsupervised algorithms, which help us explore data when we’re not sure what we’re looking for. Now you can analyze data without a clear output in mind.

You’ll work primarily with the K-means algorithm, the main unsupervised algorithm that groups similar data into clusters. We’ll build on this by speeding up the process using principal component analysis (PCA), which employs many different features.

Before starting this module, you should have a strong understanding of training and testing datasets.

## Summery

Describe the differences between supervised and unsupervised learning, including real-world examples of each.
Preprocess data for unsupervised learning.
Cluster data using the K-means algorithm.
Determine the best amount of centroids for K-means using the elbow curve.
Use PCA to limit features and speed up the model.

### Supervised vs. Unsupervised Learning

In unsupervised learning, there are two key differences from the above approach:

+ There are no paired inputs and outcomes.
+The model uses a whole dataset as input.

Unsupervised learning is used in one of the following two ways:

+Transform the data to create an intuitive representation for analysis or to use in another machine learning model; or
+Cluster or determine patterns in a grouping of data, rather than to predict a classification.

### Challenges of Unsupervised Learning

Recall that unsupervised learning does not take in any pairing of input and outcomes from the data—it only looks at the data as a whole. This can cause some challenges when running the algorithm. Since we won’t know the outcome it’s predicting, we might not know that the result is correct.

This can lead to issues where we’re trying to decide if the model has provided any helpful information that we can use to make decisions in the real world. For example, our store owner might run a model that ends up grouping the type of people by how much they’re buying. This could be useful in some contexts—for example, knowing who the top spenders are—but it might not help the store owner better organize the store for maximum purchases per person, or understand the differences in product preferences between top purchasers.

The only way to determine what an unsupervised algorithm did with the data is to go through it manually or create visualizations. Since there will be a manual aspect, unsupervised learning is great for when you want to explore the data. Sometimes you’ll use the information provided to you by the unsupervised algorithm to transition to a more targeted, supervised model.

As with supervised learning, data should be preprocessed into a correct format with only numerical values, null value determination, and so forth. The only difference is unsupervised learning doesn’t have a target variable—it only has input features that will be used to find patterns in the data. It’s important to carefully select features that could help to find those patterns or create groups.

The next section will cover data preprocessing and data munging, and provide a refresher on Pandas and data cleaning. First, you’ll need to install the necessary libraries for practice.

### Data Preprocessing
### Clustering and the K-means Algorithm 
### Using the Elbow Curve to Find Centroids 
### Managing Data Features 
### Hierarchical Clustering 
### Challenge 

## Background
Martha is a senior manager for the Advisory Services Team at Accountability Accounting, one of your most important clients. Accountability Accounting, a prominent investment bank, is interested in offering a new cryptocurrencies investment portfolio for its customers. The company, however, is lost in the immense universe of cryptocurrencies and asks you to present a report of what cryptocurrencies are on the trading market and how cryptocurrencies could be grouped toward creating a classification for developing this new investment product.

The data Martha will be working with is not ideal, so it will be processed to fit the machine learning models. Since there is no known output for what Martha is looking for, she decided to use unsupervised learning. To group the cryptocurrencies, Martha decided on a clustering algorithm to help determine about investing in this product. She’ll use data visualizations to share her findings with the board.

## Objectives
The goals for this challenge are for you to:

Prepare the data for dimensions reduction with PCA and clustering using K-means.
Reduce data dimensions using PCA algorithms from sklearn.
Predict clusters using cryptocurrencies data using the K-means algorithm form sklearn.
Create some plots and data tables to present your results.

## Instructions
Begin by downloading the CSV you need to complete the challenge.

Download cryptocurrency data [crypto_data.csv](https://github.com/hbostanchi/Cryptocurrencies/blob/master/crypto_data.csv)

### Data Preprocessing
In this section, you have to load the information about cryptocurrencies from the provided CSV file and perform some data preprocessing tasks. The data was retrieved from CryptoCompare (Links to an external site.).

Start by loading the data in a Pandas DataFrame named “crypto_df.” Continue with the following data preprocessing tasks:

Remove all cryptocurrencies that aren’t trading.
Remove all cryptocurrencies that don’t have an algorithm defined.
Remove the IsTrading column.
Remove all cryptocurrencies with at least one null value.
Remove all cryptocurrencies without coins mined.
Store the names of all cryptocurrencies on a DataFramed named coins_name, and use the crypto_df.index as the index for this new DataFrame.
Remove the CoinName column.
Create dummies variables for all of the text features, and store the resulting data on a DataFrame named X.
Use the StandardScaler from sklearn (Links to an external site.) to standardize all of the data from the X DataFrame. Remember, this is important prior to using PCA and K-means algorithms.
Reducing Data Dimensions Using PCA
Use the PCA algorithm from sklearn (Links to an external site.) to reduce the dimensions of the X DataFrame down to three principal components.

Once you have reduced the data dimensions, create a DataFrame named “pcs_df” that includes the following columns: PC 1, PC 2, and PC 3. Use the crypto_df.index as the index for this new DataFrame.

You should have a DataFrame like the following:


### Clustering Cryptocurrencies Using K-means
You’ll use the KMeans algorithm from sklearn (Links to an external site.) to cluster the cryptocurrencies using the PCA data.

Complete the following tasks:

Create an elbow curve to find the best value for K, and use the pcs_df DataFrame.
Once you define the best value for K, run the K-means algorithm to predict the K clusters for the cryptocurrencies’ data. Use the pcs_df to run the K-means algorithm.
Create a new DataFrame named “clustered_df,” that includes the following columns: Algorithm, ProofType, TotalCoinsMined, TotalCoinSupply, PC 1, PC 2, PC 3, CoinName, and Class. You should maintain the index of the crypto_df DataFrames as is shown below:
 The DataFrame shows nine columns: Algorithm, ProofType, TotalCoinsMined, TotalCoinSupply, PC1, PC 2, PC 3, CoinName, and Class. It contains ten rows with the following headings: 42, 404, 1337, BTC, ETH, LTC, DASH, XMR, ETC, and ZEC.

### Visualizing Results
You’ll create data visualizations to present the final results.

Complete the following tasks:

Create a 3D scatter plot using Plotly Express to plot the clusters using the clustered_df DataFrame. You should include the following parameters on the plot: hover_name="CoinName" and hover_data=["Algorithm"] to show this additional info on each data point.
Use hvplot.table to create a data table with all the current tradable cryptocurrencies. The table should have the following columns: CoinName, Algorithm, ProofType, TotalCoinSupply, TotalCoinsMined, and Class.
Create a scatter plot using hvplot.scatter to present the clustered data about cryptocurrencies having x="TotalCoinsMined" and y="TotalCoinSupply" to contrast the number of available coins versus the total number of mined coins. Use the hover_cols=["CoinName"] parameter to include the cryptocurrency name on each data point.
## Submission
Make sure your repo is up to date and includes the following:

A README.md file containing a short description of your project.
A Jupyter Notebook with all of your code.
Submit the link to your repository through Canvas.
