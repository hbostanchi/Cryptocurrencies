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
+ The model uses a whole dataset as input.

### Challenges of Unsupervised Learning

Recall that unsupervised learning does not take in any pairing of input and outcomes from the data—it only looks at the data as a whole. This can cause some challenges when running the algorithm. Since we won’t know the outcome it’s predicting, we might not know that the result is correct.

This can lead to issues where we’re trying to decide if the model has provided any helpful information that we can use to make decisions in the real world. For example, our store owner might run a model that ends up grouping the type of people by how much they’re buying. This could be useful in some contexts—for example, knowing who the top spenders are—but it might not help the store owner better organize the store for maximum purchases per person, or understand the differences in product preferences between top purchasers.

The only way to determine what an unsupervised algorithm did with the data is to go through it manually or create visualizations. Since there will be a manual aspect, unsupervised learning is great for when you want to explore the data. Sometimes you’ll use the information provided to you by the unsupervised algorithm to transition to a more targeted, supervised model.

As with supervised learning, data should be preprocessed into a correct format with only numerical values, null value determination, and so forth. The only difference is unsupervised learning doesn’t have a target variable—it only has input features that will be used to find patterns in the data. It’s important to carefully select features that could help to find those patterns or create groups.

Before moving data to our unsupervised algorithms, complete the following steps for preparing data:


Unsupervised learning is used in one of the following two ways:

## Preprocess data for unsupervised learning.
Before moving data to our unsupervised algorithms, we complete the following steps for preparing data:
•	Data selection
•	Data processing
•	Data transformation
### Data Selection 
entails making good choices about which data will be used. Consider what data is available, what data is missing, and what data can be removed.
### Data processing
 involves organizing the data by formatting, cleaning, and sampling it.
### Data transformation
 entails transforming our data into a simpler format for storage and future use, such as a CSV, spreadsheet, or database file.
Once our data is cleaned and processed, we would export the final version of the data as a CSV file for future analysis.
## Cluster data using the K-means algorithm.
Clustering is a type of unsupervised learning that groups data points together. This group of data points is called a cluster.
### K-means 
is an unsupervised learning algorithm used to identify and solve clustering issues.
+ K represents how many clusters there will be. These clusters are then determined by the means of all the points that will belong to the cluster.
+ The K-means algorithm groups the data into K clusters, where belonging to a cluster is based on some similarity or distance measure to a centroid.
A centroid is a data point that is the arithmetic mean position of all the points on a cluster.
+ The centroid is found by taking the mean of all the x values in a cluster, and the mean of all the y values in a cluster.
## Determine the best amount of centroids for K-means using the elbow curve.
One method for determining the best number for K is the elbow curve. Elbow curves get their names from their shape: they turn on a specific value, which looks like an elbow.
To create an elbow curve:
+ Plot the clusters on the x-axis
+ Plot the values of a selected objective function on the y-axis.
Inertia is one of the most common objective functions to use when creating an elbow curve. The inertia objective function measures the amount of variation in the dataset.
To create an elbow curve:
+ NPlot the number of clusters (also known as the values of K) on the x-axis
+ Plot the inertia values on the y-axis.
## Use PCA to limit features and speed up the model.
### PCA(Principal Component Analysis) is a statistical technique to speed up machine learning algorithms when the number of input features (or dimensions) is too high. PCA reduces the number of dimensions by transforming a large set of variables into a smaller one that contains most of the information in the original large set.
Stats concepts in a feature extraction:
+ Mean is the sum of a group of numbers divided by the total amount of numbers.
+ Variance is the square distance from each point from the center, added together, and divided by the total number of points. The variance measures the spread of a set of numbers.
The center of the line is set to the mean.
Covariance comes into play when the variances are exactly the same; however, it is very obvious that the two graphs are totally different. For example, when each has the same center, with different points on the left and the right, one sloping negatively and the other sloping positively.
+ ### Covariance is a metric that allows us to tell these two different sets of points apart.
Covariance is the sum of the product of coordinates divided by the number of points. The Line can have:
+ a negative covariance
+ a positive covariance
+ covariance of zero.
Eigenvectors and eigenvalues show us the spread of the dataset and by how much.
+ ### eigenvectors are the points that stretch out in our graph in different directions.
+ ### eigenvalue are the magnitude that each of these stretches.
The higher eigenvalue is the axis that carries the most amount of information. The Final data points will give us a table of two columns. Ultimately, just reducing dimensions yet still keeping the values.
