import json
import random
import numpy as np
import pandas as pd
import os
import glob
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import seaborn as sns



class FeatureSet:
    """
    A class of ...

    """
    def __init__(self, path='C:/Users/AlexH/OneDrive/Documenten/Julia/Implementation of network + clustering of network feeders/summer job Alexander/POLA',
                 include_n_customer=True, include_total_length=True, include_main_path=False, include_avg_cons=False, \
                 include_avg_reactive_cons=False, include_n_PV=False, include_total_impedance=False, \
                 include_average_length=False, include_average_impedance=False,include_empty_feeders=True):
        """
        Initialize
        """
        features = []
        self._path = path
        list = ["Number of customers","Yearly consumption per customer (kWh)","Yearly reactive consumption per customer (kWh)","Number of PV installations", \
                "Total conductor length (km)","Main path length (km)","Average length to customer (km)", "Total line impedance (Ohm)","Average path impedance (Ohm)"]
        includes = [include_n_customer, include_avg_cons, include_avg_reactive_cons, include_n_PV, \
                    include_total_length, include_main_path, include_average_length, include_total_impedance, include_average_impedance]
        self._feature_list = [list[i] for i in range(len(list)) if includes[i]]
        #cycle through all the json files
        for file in glob.glob(os.path.join(self._path, '*configuration.json')):
            with open(file) as current_file:
                config_data = json.load(current_file)
            row = [config_data['gridConfig']['id']]
            devices_path = config_data['gridConfig']['devices_file']
            branches_path = config_data['gridConfig']['branches_file']
            if include_n_customer == True:
                row += [config_data['gridConfig']['totalNrOfEANS']]

            if include_avg_cons == True or include_n_PV == True or include_avg_reactive_cons == True or \
                    include_average_length == True or include_total_impedance == True or include_average_impedance == True:
                with open(os.path.join(os.path.dirname(self._path), devices_path)) as devices_file:
                    devices_data = json.load(devices_file)
                    if include_avg_cons == True:
                        total_cons = 0
                        for device in devices_data['LVcustomers']:
                            try:
                                cons = device.get('yearlyNetConsumption')/config_data['gridConfig']['totalNrOfEANS']
                            except ZeroDivisionError:
                                cons = 0
                            try:        #Dirty fix because cons contained nan and None
                                total_cons += float(cons)
                            except TypeError:
                                print("yearlyNetConsumption contains NoneType")
                        row += [total_cons]
                    if include_avg_reactive_cons == True:
                        total_reactive_cons = 0
                        for device in devices_data['LVcustomers']:
                            try:
                                reactive_cons = device.get('yearlyNetReactiveConsumption')/config_data['gridConfig']['totalNrOfEANS']
                            except ZeroDivisionError:
                                reactive_cons = 0
                            try:
                                total_reactive_cons += float(reactive_cons)
                            except TypeError:
                                print("yearlyNetReactiveConsumption contains NoneType")
                        row += [total_reactive_cons]
                    if include_n_PV == True:
                        row += [len(devices_data["solarGens"])]

            if include_total_length == True or include_main_path == True or include_average_length == True or \
                    include_total_impedance == True or include_average_impedance == True:
                with open(os.path.join(os.path.dirname(self._path), branches_path)) as branches_file:
                    branches_data = json.load(branches_file)
                    if include_total_length == True:
                        total_length = 0
                        for branch in branches_data:
                            length = branch.get('cableLength')
                            try:        # Dirty fix because cons contained nan and None
                                total_length += float(length)
                            except TypeError:
                                print("cableLength contains NoneType")
                        row += [total_length/1000]
                    if include_main_path == True:
                        row += [longest_path(0,branches_data)/1000]
                    if include_average_length == True:
                        total_length, n_customers = total_path_length(0, branches_data, devices_data)
                        try:
                            row += [total_length/n_customers/1000]
                        except ZeroDivisionError:
                            row += [0]
                    if include_total_impedance == True:
                        row += [total_path_impedance(0,branches_data,devices_data)[0]/1000]
                    if include_average_impedance == True:
                        total_impedance, n_customers = total_path_impedance(0, branches_data, devices_data)
                        try:
                            row += [total_impedance / n_customers/1000]
                        except ZeroDivisionError:
                            row += [0]
            if include_empty_feeders or config_data['gridConfig']['totalNrOfEANS'] > 0:
                features.append(row)
        self._IDs = [i[0] for i in features]
        array = np.array(features)
        self._features = array[:,1:]


        #write the data of interest to the feature-set

    def get_features(self):
        return self._features

    def get_IDs(self):
        return self._IDs

    def get_feature_list(self):
        return self._feature_list

    def get_feature(self,i):
        if isinstance(i,str):
            ind = self.get_feature_list().index(i)
            return self._features[:,ind]
        return self._features[:,i]

    def set_feature(self,i,new_feature):
        self._features[:,i] = new_feature

    def hierarchal_clustering(self,n_clusters=8,normalized=True,criterion='avg_silhouette'):
        if normalized:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.get_features())
        else:
            data = self.get_features()
        labels = AgglomerativeClustering(n_clusters).fit(data).labels_
        if criterion == 'global_silhouette':
            score = global_silhouette_criterion(data, labels)
        if criterion == 'avg_silhouette':
            score = silhouette_score(data, labels)
        return Cluster(labels,'hierarchal clustering',normalized,1,criterion,score)

    def k_means_clustering(self,n_clusters=8,normalized=True,n_repeats=1,criterion='avg_silhouette'):
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.get_features())
        else:
            data = self.get_features()

        if criterion == 'avg_silhouette':
            best_cluster_labels = np.zeros(np.size(data,0))
            score = -1
            for i in range(0,n_repeats):
                i_cluster_labels = KMeans(n_clusters).fit(data).labels_
                i_silhouette_avg = silhouette_score(data, i_cluster_labels)
                if i_silhouette_avg > score:
                    score = i_silhouette_avg
                    best_cluster_labels = i_cluster_labels
        if criterion == 'global_silhouette':
            best_cluster_labels = np.zeros(np.size(data, 0))
            score = -1
            for i in range(0, n_repeats):
                i_cluster_labels = KMeans(n_clusters).fit(data).labels_
                i_silhouette_global = global_silhouette_criterion(data, i_cluster_labels)
                if i_silhouette_global > score:
                    score = i_silhouette_global
                    best_cluster_labels = i_cluster_labels
        return Cluster(best_cluster_labels,'k-means++',normalized,n_repeats,criterion,score)

    def k_medoids_clustering(self,n_clusters=8,normalized=True,n_repeats=1,criterion='global_silhouette'):
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.get_features())
        else:
            data = self.get_features()

        if criterion == 'avg_silhouette':
            best_cluster_labels = np.zeros(np.size(data,0))
            score = -1
            for i in range(0,n_repeats):
                i_cluster_labels = KMedoids(n_clusters,init='k-medoids++').fit(data).labels_
                i_silhouette_avg = silhouette_score(data, i_cluster_labels)
                if i_silhouette_avg > score:
                    score = i_silhouette_avg
                    best_cluster_labels = i_cluster_labels
        if criterion == 'global_silhouette':
            best_cluster_labels = np.zeros(np.size(data, 0))
            score = -1
            for i in range(0, n_repeats):
                i_cluster_labels = KMedoids(n_clusters,init='k-medoids++').fit(data).labels_
                i_silhouette_global = global_silhouette_criterion(data, i_cluster_labels)
                if i_silhouette_global > score:
                    score = i_silhouette_global
                    best_cluster_labels = i_cluster_labels
        return Cluster(best_cluster_labels, 'k-medoids++', normalized, n_repeats, criterion, score)

    def gaussian_mixture_model(self,n_clusters=8,normalized=True,n_repeats=1):
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.get_features())
        else:
            data = self.get_features()

        best_cluster_labels = np.zeros(np.size(data, 0))
        score = -1
        reg_values = np.linspace(0.001,.1, num=n_repeats)
        for i in range(0, n_repeats):
            i_cluster_labels = np.array(GaussianMixture(n_components=n_clusters,reg_covar=reg_values[i]).fit_predict(data))
            i_silhouette_avg = silhouette_score(data, i_cluster_labels)
            if i_silhouette_avg > score:
                score = i_silhouette_avg
                best_cluster_labels = i_cluster_labels
        return Cluster(best_cluster_labels, 'Gaussian mixture model', normalized,n_repeats,'avg_silhouette',score)

def longest_path(busId,branches_data): #Could be faster if you 'pop' the branches such that they are not searched again
    longest_found = 0
    for branch in branches_data:
        if branch.get("upBusId") == busId:
            found = branch.get("cableLength") + longest_path(branch.get("downBusId"),branches_data)
            if found > longest_found:
                longest_found = found
    return longest_found

def total_path_length(busId,branches_data,devices_data): #Could be faster if you 'pop' the branches such that they are not searched again
    path_length = 0
    n_devices = 0
    for branch in branches_data:
        if branch.get("upBusId") == busId:
            subpath_length, n_subpath_devices = total_path_length(branch.get("downBusId"),branches_data,devices_data)
            path_length += subpath_length + branch.get("cableLength")*n_subpath_devices
            n_devices += n_subpath_devices
    n_devices += sum(1 for i in devices_data["LVcustomers"] if i['busId'] == busId)
    return path_length, n_devices

def total_path_impedance(busId,branches_data,devices_data): #Could be faster if you 'pop' the branches such that they are not searched again
    path_impedance = 0
    n_devices = 0
    for branch in branches_data:
        if branch.get("upBusId") == busId:
            impedance = lookup_impedance(branch.get("cableType"))
            subpath_length, n_subpath_devices = total_path_length(branch.get("downBusId"),branches_data,devices_data)
            path_impedance += subpath_length + branch.get("cableLength")*impedance*n_subpath_devices
            n_devices += n_subpath_devices
    n_devices += sum(1 for i in devices_data["LVcustomers"] if i['busId'] == busId)
    return path_impedance, n_devices

def lookup_impedance(cable_type): #Includes DC resistance only, supposes all loads are single phase connected
    lookup_table = {"BT - Desconocido BT": 0.222991031,
                    "BT - MANGUERA" :1.23259888,
                    "BT - RV 0,6/1 KV 2*16 KAL" :2.141891687,
                    "BT - RV 0,6/1 KV 2*25 KAL" :1.343506234,
                    "BT - RV 0,6/1 KV 3(1*150 KAL) + 1*95 KAL" :0.246053082,
                    "BT - RV 0,6/1 KV 3(1*240 KAL) + 1*150 KAL" :0.178676325,
                    "BT - RV 0,6/1 KV 3(1*240 KAL) + 1*95 KAL": 0.178676325,
                    "BT - RV 0,6/1 KV 4*25 KAL" :1.343506234,
                    "BT - RV 0,6/1 KV 4*50 KAL" :0.724490814,
                    "BT - RV 0,6/1 KV 4*95 KAL" :0.369564719,
                    "BT - RX 0,6/1 KV 2*16 Cu" :1.23259888,
                    "BT - RX 0,6/1 KV 2*2 Cu": 9.900284087,
                    "BT - RX 0,6/1 KV 2*4 Cu": 4.950568149,
                    "BT - RX 0,6/1 KV 2*6 Cu": 3.300852163,
                    "BT - RZ 0,6/1 KV 2*16 AL": 2.141891687,
                    "BT - RZ 0,6/1 KV 3*150 AL/80 ALM": 0.246053082,
                    "BT - RZ 0,6/1 KV 3*150 AL/95 ALM" :0.246053082,
                    "BT - RZ 0,6/1 KV 3*25 AL/54,6 ALM" :1.343506234,
                    "BT - RZ 0,6/1 KV 3*35 AL/54,6 ALM" :0.91225999,
                    "BT - RZ 0,6/1 KV 3*50 AL/54,6 ALM" :0.724490814,
                    "BT - RZ 0,6/1 KV 3*70 ALM/54,6 AL" :0.462932187,
                    "BT - RZ 0,6/1 KV 3*95 AL/54,6 ALM" :0.369564719,
                    "BT - RZ 0,6/1 KV 4*16 AL": 2.141891687,
                    "aansluitkabel": 1.15974135}
    if cable_type in lookup_table:
        return lookup_table[cable_type]
    else:
        print(cable_type)
        return 0.124

class Cluster:
    def __init__(self,clusters,algorithm,normalized=False,n_repeats=1,criterion='global_silhouette',score=np.nan):
        self._clusters = clusters
        self._algorithm = algorithm
        self._normalized = normalized
        self._n_clusters = np.max(clusters)+1
        self._n_repeats = n_repeats
        self._criterion = criterion
        self._score = score

    def get_clusters(self):
        return self._clusters

    def get_algorithm(self):
        return self._algorithm

    def get_normalisation(self):
        if self._normalized == True:
            return 'normalized'
        else:
            return 'not normalized'

    def get_n_repeats(self):
        return self._n_repeats

    def get_repeats(self):
        if self.get_n_repeats() == 1:
            return ''
        else:
            return ' repeated %d times' %self.get_n_repeats()

    def get_criterion(self):
        return self._criterion

    def is_normalised(self):
        return self._normalized

    def get_n_clusters(self):
        return self._n_clusters

    def get_score(self):
        return self._score





def plot_2D_clusters(FeatureSet,Cluster,x_axis=None,y_axis=None):
    axis_labels = FeatureSet.get_feature_list()
    cluster_labels = Cluster.get_clusters()
    plt.figure(figsize=(8,6))
    if x_axis == None:
        x = FeatureSet.get_feature(0)
        x_axis = axis_labels[0]
    elif x_axis in axis_labels:
        x = FeatureSet.get_feature(x_axis)
    else:
        raise AttributeError
    if y_axis == None:
        y = FeatureSet.get_feature(1)
        y_axis = axis_labels[1]
    elif y_axis in axis_labels:
        y = FeatureSet.get_feature(y_axis)
    else:
        raise AttributeError
    plt.scatter(x,y, c=cluster_labels,alpha=0.85)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(Cluster.get_algorithm() +" with n_clusters = %d" % Cluster.get_n_clusters() + Cluster.get_repeats())
    plt.show()

def silhouette_analysis(FeatureSet,Cluster):
    features = FeatureSet.get_features()
    cluster_labels = Cluster.get_clusters()
    n_clusters = Cluster.get_n_clusters()
    feature_list = FeatureSet.get_feature_list()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(22, 10))
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(features) + (n_clusters + 1) * 10])

    if Cluster.is_normalised():
        scaler = StandardScaler()
        features_normalised = scaler.fit_transform(features)
        silhouette_avg = silhouette_score(features_normalised, cluster_labels)
        silhouette_global = global_silhouette_criterion(features_normalised, cluster_labels)
        sample_silhouette_values = silhouette_samples(features_normalised, cluster_labels)
    else:
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_global = global_silhouette_criterion(features, cluster_labels)
        sample_silhouette_values = silhouette_samples(features, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.viridis(float(i) / (n_clusters-1))

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="grey", linestyle="--",label='average silhouette coef %f3' % silhouette_avg)
    ax1.axvline(x=silhouette_global, color="grey", linestyle="-.",label='global silhouette coef %f3' % silhouette_global)
    ax1.legend(loc='upper right')

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = plt.cm.viridis(cluster_labels.astype(float) / (n_clusters-1))
    ax2.scatter(features[:, 0], features[:, 1], marker='o', s=30, lw=0, alpha=0.85,
                c=colors, edgecolor='k')

    # Labeling the clusters
    #centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    #ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #            c="white", alpha=1, s=200, edgecolor='k')

    #for i, c in enumerate(centers):
    #    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel(feature_list[0])
    ax2.set_ylabel(feature_list[1])

    plt.suptitle(("Silhouette analysis for "+ Cluster.get_algorithm() +
                  " with n_clusters = %d" % n_clusters + Cluster.get_repeats()),
                 fontsize=14, fontweight='bold')
    plt.show()

def compare_algorithms(FeatureSet,criterion,n,range):
    results = {'Hierarchal':dict(),'K-means++':dict(),'K-medoids++':dict(),'GMM':dict()}
    scores = np.zeros([4,len(range)])
    for i in range:
        results['Hierarchal'][i] = FeatureSet.hierarchal_clustering(n_clusters=i)
        results['K-means++'][i] = FeatureSet.k_means_clustering(n_clusters=i, n_repeats=n)
        results['K-medoids++'][i] = FeatureSet.k_medoids_clustering(n_clusters=i, n_repeats=n)
        results['GMM'][i] = FeatureSet.gaussian_mixture_model(n_clusters=i, n_repeats=n)
        scores[0,i-range[0]] = results['Hierarchal'][i].get_score()
        scores[1,i-range[0]] = results['K-means++'][i].get_score()
        scores[2,i-range[0]] = results['K-medoids++'][i].get_score()
        scores[3,i-range[0]] = results['GMM'][i].get_score()
        print(i," out of ",range[-1]," complete")
    plt.figure(figsize=(12, 10))
    plt.plot(range,scores[0,:],label='Hierarchal')
    plt.plot(range,scores[1,:],label='K-means++')
    plt.plot(range,scores[2,:],label='K-medoids++')
    plt.plot(range,scores[3,:],label='GMM')
    plt.legend()
    plt.xlabel("number of clusters K")
    plt.ylabel("average silhouette coefficient")
    plt.show()
    return results, scores
def variance_ratio_criterion():
    raise NotImplementedError

def global_silhouette_criterion(features,cluster_labels):
    nb_clusters = np.max(cluster_labels)+1
    scores = silhouette_samples(features,cluster_labels)
    score = 0
    for i in range(0,nb_clusters):
        score += scores[cluster_labels==i].mean()
    return score/nb_clusters