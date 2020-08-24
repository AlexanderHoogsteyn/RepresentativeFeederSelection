from clustering import *

#Number of customers & Total coductor length
f = FeatureSet()
assert np.size(f.get_features(),1) == 2
assert np.size(f.get_features(),0) == 160
assert not np.isnan(f.get_features()).any()
assert f.get_feature("Number of customers").all() == f.get_feature(0).all()
#plot_2D_clusters(f,f.hierarchal_clustering())
silhouette_analysis(f,f.hierarchal_clustering(normalized=True))

#Number of customers & Total yearly consumption
h = FeatureSet(include_total_cons=True, include_total_length=False)
assert np.size(h.get_features(),1) == 2
assert np.size(h.get_features(),0) == 160
assert not np.isnan(h.get_features()).any()
#plot_2D_clusters(h,h.hierarchal_clustering())


#Total yearly consumption & Total coductor length
i = FeatureSet(include_total_cons=True, include_total_length=True, include_n_customer=False)
assert np.size(i.get_features(),1) == 2
assert np.size(i.get_features(),0) == 160
assert not np.isnan(i.get_features()).any()
'''
plot_2D_clusters(i,i.hierarchal_clustering())
plot_2D_clusters(i,i.hierarchal_clustering(normalized=True))
plot_2D_clusters(i,i.k_means_clustering(normalized=True,n_repeats=10))
plot_2D_clusters(i,i.k_medoids_clustering(normalized=True,n_repeats=10))
plot_2D_clusters(i,i.gaussian_mixture_model(normalized=True,n_repeats=10))
silhouette_analysis(i,i.k_means_clustering(normalized=True,n_repeats=10))
silhouette_analysis(i,i.k_means_clustering(normalized=True,n_repeats=10,criterion='avg_silhouette'))
'''

#Number of customers & Total coductor length & Total yearly consumption
g= FeatureSet(include_total_cons=True)
assert g.get_feature_list() == ["Number of customers","Total yearly consumption (kWh)","Total conductor length (km)"]
assert np.size(g.get_features(),1) == 3
assert np.size(g.get_features(),0) == 160
assert not np.isnan(g.get_features()).any()
assert g.get_feature("Total conductor length (km)").all() == g.get_feature(2).all()
'''
plot_2D_clusters(g,g.hierarchal_clustering(),x_axis="Number of customers",y_axis="Total conductor length (km)")
plot_2D_clusters(g,g.hierarchal_clustering(normalized=True))
plot_2D_clusters(g,g.gaussian_mixture_model(normalized=True,n_repeats=10))
silhouette_analysis(g,g.k_medoids_clustering(normalized=True))
'''

#Number of customers & Total coductor length & Total yearly consumption & Total reactive power
j= FeatureSet(include_total_length=True,include_total_cons=True,include_total_reactive_cons=True)
assert j.get_feature_list() == ["Number of customers","Total yearly consumption (kWh)","Total yearly reactive consumption (kWh)","Total conductor length (km)"]
assert np.size(j.get_features(),1) == 4
assert np.size(j.get_features(),0) == 160
assert not np.isnan(j.get_features()).any()
'''
plot_2D_clusters(j,j.hierarchal_clustering(normalized=True))
plot_2D_clusters(j,j.gaussian_mixture_model(normalized=True,n_repeats=10),x_axis="Total yearly consumption (kWh)",y_axis="Total conductor length (km)")
silhouette_analysis(j,j.k_medoids_clustering(normalized=True))
'''

#Main path
k= FeatureSet(include_main_path=True)
assert k.get_feature_list() == ["Number of customers","Total conductor length (km)","Main path length (km)",]
assert np.size(k.get_features(),1) == 3
assert np.size(k.get_features(),0) == 160
assert not np.isnan(k.get_features()).any()
assert k.get_feature("Main path length (km)").all() <= k.get_feature(1).all()
assert k.get_feature("Main path length (km)").all() == k.get_feature(2).all()