# Clustering.py
Toolset for performing clustering of feeders in a distribution network. Using the FeatureSet class, objects can be created
that contains the set of features about the network that is specified by the user. Four different clustering algorithms
can be performed on FeatureSet objects. This includes the well established K-means++ algorithm as well as Agglomerative
hierarchal clustering, K-medoids++ and Gaussian mixture model.

## GridDataCollection
This scripts imports the network topology from JSON files in a specific format. Such a representation of a distribution network can be obtained using a script
such as createGridDataCollection_dataframes_poly.py. That script in specific does the conversion from excel files as they were recieved by the Spanish DSO. Each feeder
has it's information stored in 4 files:
- Configuration: contains general information and lists the directory's of the other 3 files.
- Devices: contains information about the devices i.e. customers connected. For example the EAN-number and connection capacity. The total yearly consumption (kWh) is included, in createGridDataCollection_dataframes_poly.py this is estimated based of 20 days of smart meter data. The phase connection is as for now, randomly allocated.
- Branches: contains information on cable length, bus connections and type of each branch. The latter is used to determine the impedance of the branch
- Buses: contains info on which bus is the slack bus and voltage limits on each bus. The voltage limits are added to model but are are only used if optimal powerflow calculations are performed.

## FeatureSet class
A Featureset object contains all the data that you want to use to perform the clustering. The path attribute is used to specify the folder which contains the JSON files. The include_empty_feeders is used to specify whether you want to include feeders that doe not have any devices i.e. customers connected to it.
The other attributes make it possible to specify which features to include:
- include_n_customers: Number of devices i.e. customers connected to a feeder
- include_total_length: Total conductor length in the feeder
- include_main_path: Longest path in the network between a device and the head of the feeder
- include_avg_cons: The average active yearly energy consumption of the customers on a feeder
- include_avg_reactive_cons: Idem for reactive energy consumption
- include_n_PV: Number of PV installations on the network
- include_total_impedance: The impedance between a customer and the head of the feeder summed up for all customers
- include_average_length: The average path length between a customer and the head of the feeder
- include_average_impedance: The average impedance between a customer and the head of the feeder
The object will store the features in  a numpy array as well as some metadata such as list of the features used and the ID's of the feeders.
### Limitations
- A FeatureSet can only be made from GridDataCollections, this is a specific format of JSON files. This can be obtained by means of scripts like createDataCollection_dataframes_pola.py. If features from another source needs to be added, FeatureSet.__init__() needs adaption.
- Features that contain NaN, None or other exceptional values will cause problems.
### Methods

## Cluster class
A cluster object contains all info on the result obtained after performing a clustering algorithm. Most notably the
labels to identify which cluster a feeder is allocated to. Besides that, the object contains some metadata about
the number of clusters, which algorithm was used, the score of the result according to the specified criterion.

### Limitations
- Cluster objects can only be created by performing one of the four clustering algorithm methods on a FeatureSet. If another algorithm needs to be performed on the data in a FeatureSet a new method in the FeatureSet class needs to be added or alternatively the data in the FeatureSet object can be accessed using methods like get_features() or get_feature()
### Methods

## Plots
The following functions make it easy to visalize the results of the
#### plot_2D_clusters(FeatureSet,Cluster,x_axis=None,y_axis=None)
Makes a 2D plot of the resulting clusters. You need to specify the FeatureSet object which contains all the used data
as well as the Cluster object which you obtained by performing one on the clustering algorithm methods
on the FeatureSet.
It can be chosen what is plotted on the x and y axis by specifying the name of a feature. This has to be the specific
string corresponding to that feature such as "Yearly consumption per customer (kWh)" (These can be found using
FeatureSet.get_feature_list() )
#### silhouette_analysis(FeatureSet,Cluster)
Makes a silhouette analysis of the resulting clusters (more info: https://en.wikipedia.org/wiki/Silhouette_(clustering) ).
You need to specify the FeatureSet object which contains all the used data as well as the Cluster object
which you obtained by performing one on the clustering algorithm methods on the FeatureSet.
#### compare_algorithms(FeatureSet,criterion,n=1,range)
Makes a graph that compares the 4 algorithms against each other according to their average silhouette coefficient.
A featureset needs to be specified to perform the analysis on.
A range of number of clusters must be specified.
A number of repetitions can be specified, K-means++, K-medoids++ and GMM will then be repeated n times and the best
result is kept. Hierarchal clustering is only performed once because its outcome is not stochastic.
The function returns the best found Cluster objects for each algorithm and cluster size. They can be accessed as follows:
```Python
result, scores = compare_algorithms(f,'avg_silhouette',1000,range(2,25))
plot_2D_clusters(f,results['K-means++'][10],x_axis="Number of customers",y_axis="Main path length (km)")
```



## Representative feeders
The goal of performing clustering on feeders in distribution networks is to exctract feeders that are representative for the entire network the following functions can help extract that information.
#### get_representative_feeders(FeatureSet,Cluster)
Function that returns a pandas dataframe with a summary of the found clusters and the mean and deviation of the
features of the feeders in that cluster.
