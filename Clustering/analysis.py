from clustering import *
import pickle

f = FeatureSet(include_total_length=False, include_n_customer=True,include_avg_cons=True,include_avg_reactive_cons=True,include_main_path=True, include_average_impedance=True,include_empty_feeders=False)

results = pickle.load(open("save.p", "rb"))
#results_ens = pickle.load(open("save_ensemble.p","rb"))

#plot_2D_clusters(f,results['K-means++'][4],x_axis="Number of customers",y_axis="Main path length (km)")
i =7

plot_2D_clusters(f,results['K-means++'][i],x_axis="Number of customers",y_axis="Main path length (km)")
plot_2D_clusters(f,results['GMM'][i],x_axis="Number of customers",y_axis="Main path length (km)")
plot_2D_clusters(f,results['K-means++'][i],x_axis="Number of customers",y_axis="Yearly consumption per customer (kWh)")
plot_2D_clusters(f,results['GMM'][i],x_axis="Number of customers",y_axis="Yearly consumption per customer (kWh)")
plot_2D_clusters(f,results['K-means++'][i],x_axis="Number of customers",y_axis="Average path impedance (Ohm)")
plot_2D_clusters(f,results['GMM'][i],x_axis="Number of customers",y_axis="Average path impedance (Ohm)")
plot_2D_clusters(f,results['K-means++'][i],x_axis="Yearly consumption per customer (kWh)",y_axis="Yearly reactive consumption per customer (kWh)")
plot_2D_clusters(f,results['GMM'][i],x_axis="Yearly consumption per customer (kWh)",y_axis="Yearly reactive consumption per customer (kWh)")
silhouette_analysis(f,results['K-means++'][i])
silhouette_analysis(f,results['GMM'][i])

dataframe = get_representative_feeders(f,results['K-means++'][5])
print(dataframe)

plot_2D_clusters(f,results_ens["Average varying"][i],x_axis="Number of customers",y_axis="Main path length (km)")
plot_2D_clusters(f, results_ens["Average fixed"][i], x_axis="Number of customers", y_axis="Main path length (km)")
plot_2D_clusters(f, results_ens["Average varying"][i], x_axis="Number of customers", y_axis="Yearly consumption per customer (kWh)")
plot_2D_clusters(f, results_ens["Average fixed"][i], x_axis="Number of customers", y_axis="Yearly consumption per customer (kWh)")
plot_2D_clusters(f, results_ens["Average varying"][i], x_axis="Number of customers", y_axis="Average path impedance (Ohm)")
plot_2D_clusters(f, results_ens["Average fixed"][i], x_axis="Number of customers", y_axis="Average path impedance (Ohm)")
plot_2D_clusters(f, results_ens["Average varying"][i], x_axis="Yearly consumption per customer (kWh)", y_axis="Yearly reactive consumption per customer (kWh)")
plot_2D_clusters(f, results_ens["Average fixed"][i], x_axis="Yearly consumption per customer (kWh)", y_axis="Yearly reactive consumption per customer (kWh)")
silhouette_analysis(f, results_ens["Average varying"][i])
silhouette_analysis(f, results_ens["Average fixed"][i])

