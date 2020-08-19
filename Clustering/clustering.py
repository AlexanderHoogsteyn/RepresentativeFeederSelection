import json
import random
import numpy as np
import pandas as pd
import os
import glob
from sklearn.cluster import AgglomerativeClustering



class FeatureSet:
    """
    A class of ...

    """
    def __init__(self,path='C:/Users/AlexH/OneDrive/Documenten/Julia/Implementation of network + clustering of network feeders/summer job Alexander/POLA/',
                 include_n_customer=True,include_total_length=True):
        """
        Initialize
        """
        self.features = []
        self.path = path
        #cycle through all the json files
        for file in glob.glob(os.path.join(path, '*configuration.json')):
            with open(file) as current_file:
                data = json.load(current_file)
            id = [data['gridConfig']['id']]
            #devices_file = data['gridConfig']['devices_file']
            #branches_file = data['gridConfig']['branches_file']
            #bus_file = data['gridConfig']['bus_file']
            if include_n_customer == True:
                id += [data['gridConfig']['totalNrOfEANS']]
            if include_total_length == True:
                devices_path = '*'+str(id[0])+'_devices.json'
                with open(glob.glob(os.path.join(path, devices_path))) as devices_file:
                    data = json.load(devices_file)
                    id += []
            self.features.append(id)

        #write the data of interest to the feature-set

    def get_features(self):
        return self.features

    def add_customer_number(self):
        """
        Reads the total amount of customers from json files
        """
        raise NotImplementedError

    def add_path_distance(self):
        raise NotImplementedError

    def hierarchal_clustering(self):
        return AgglomerativeClustering().fit(self.features).labels_

