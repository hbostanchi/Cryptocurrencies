# Cryptocurrencies

## Challenge Overview
The senior manager for the Advisory Services Team at Accountability Accounting is looking for help with a report of what cryptocurrencies are on the market and how they could be grouped. Because of the data available, it was decided that unsupervised machine learning would be best to create classifications. For the report, the following tasks were completed.

Prepare the data for dimension reduction and clustering.
Reduce data dimensions using PCA from sklearn.
Predict clusters using the K-means algorithm from sklearn.
Create 2D and 3D scatter plots and a data table to present the results.

## Resources
Data source: 
Cryptocurrency data [crypto_data.csv](https://github.com/hbostanchi/Cryptocurrencies/blob/master/challenge/crypto_data.csv)
Software: Python 3.7 using libraries: Pandas, Scikit-learn, hvplot and Plotly Express; Jupyter Notebook


## Challenge Summary

## Objectives
The goals for this challenge are for you to:

- Prepare the data for dimensions reduction with PCA and clustering using K-means.
- Reduce data dimensions using PCA algorithms from sklearn.
= Predict clusters using cryptocurrencies data using the K-means algorithm form sklearn.
- Create some plots and data tables to present your results.
### Data Preprocessing
In this section, we had to load the information about cryptocurrencies from the provided CSV file and perform some data preprocessing tasks. The data was retrieved from CryptoCompare

We started by loading the data in a Pandas DataFrame named [“crypto_df”](https://github.com/hbostanchi/Cryptocurrencies/blob/master/challenge/Crypto_challenge.ipynb) and continued with the following data preprocessing tasks:

- Remove all cryptocurrencies that aren’t trading.
- Remove the IsTrading column.
- Remove all cryptocurrencies with at least one null value.
- Remove all cryptocurrencies without coins mined.
- Store the names of all cryptocurrencies on a DataFramed named coins_name, and use the crypto_df.index as the index for this new DataFrame.
- Remove the CoinName column.
- Create dummies variables for all of the text features, and store the resulting data on a DataFrame named X.
- Use the StandardScaler from sklearn to standardize all of the data from the X DataFrame.

### Reducing Data Dimensions Using PCA
We used the [PCA algorithm from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to reduce the dimensions of the X DataFrame down to three principal components.

Once we had reduced the data dimensions, we created a DataFrame named “pcs_df” that includes the following columns:

- PC 1
- PC 2
- PC 3
We used the crypto_df.index as the index for this new DataFrame.

Clustering Cryptocurrencies Using K-means
We used the [KMeans algorithm from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) to cluster the cryptocurrencies using the PCA data.

Create an elbow curve to find the best value for K, and use the pcs_df DataFrame.
Once you define the best value for K, run the K-means algorithm to predict the K clusters for the cryptocurrencies’ data. Use the pcs_df to run the K-means algorithm.

he data was cleaned, scaled with StandardScaler and reduced to three principal components using PCA.

- The elbow curve showed a prominent bend at k = 4, so the K-means algorithm was run with 4 clusters.

![elbow curve](https://github.com/hbostanchi/Cryptocurrencies/blob/master/challenge/image/Elbow_curve.png)

- Most cryptocurrencies fit into two of the four clusters.
- Bittorrent had such large numbers, that it is in its own cluster.
- The table shows available cryptocurrencies; table is sortable and selectable.

![table](https://github.com/hbostanchi/Cryptocurrencies/blob/master/challenge/image/table.png)


Create a new DataFrame named “clustered_df,” that includes the following columns: Algorithm, ProofType, TotalCoinsMined, TotalCoinSupply, PC 1, PC 2, PC 3, CoinName, and Class.
You should maintain the index of the crypto_df DataFrames as is shown below:
 The DataFrame shows nine columns: Algorithm, ProofType, TotalCoinsMined, TotalCoinSupply, PC1, PC 2, PC 3, CoinName, and Class. It contains ten rows with the following headings: 42, 404, 1337, BTC, ETH, LTC, DASH, XMR, ETC, and ZEC.

### Visualizing Results
You’ll create data visualizations to present the final results.

Complete the following tasks:

Create a 3D scatter plot using Plotly Express to plot the clusters using the clustered_df DataFrame. You should include the following parameters on the plot: hover_name="CoinName" and hover_data=["Algorithm"] to show this additional info on each data point.

![3D scatter plot](https://github.com/hbostanchi/Cryptocurrencies/blob/master/challenge/image/3D%20scatter%20plot.png)

Use hvplot.table to create a data table with all the current tradable cryptocurrencies. The table should have the following columns: CoinName, Algorithm, ProofType, TotalCoinSupply, TotalCoinsMined, and Class.
Create a scatter plot using hvplot.scatter to present the clustered data about cryptocurrencies having x="TotalCoinsMined" and y="TotalCoinSupply" to contrast the number of available coins versus the total number of mined coins. Use the hover_cols=["CoinName"] parameter to include the cryptocurrency name on each data point.

![scatterplot](https://github.com/hbostanchi/Cryptocurrencies/blob/master/challenge/image/scatterplot.png)


