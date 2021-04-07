from clustering import *

f = FeatureSet(include_total_length=True,include_avg_cons=True,include_main_path=True,include_average_impedance=True)

cluster = f.k_means_clustering(n_clusters=6,n_repeats=100)

table = get_representative_feeders(f,cluster)
IDs = np.array(f.get_IDs())
cust = f.get_feature(0)
length = f.get_feature(1)

plot_2D_clusters(f,cluster,"Number of customers","Main path length (km)")
plot_2D_clusters(f,cluster,"Average path impedance (Ohm)","Yearly consumption per customer (kWh)")

print(table.head(10))

#print(IDs[cust==125])
#print(IDs[length<0.5])

#print(length[cust==125])
