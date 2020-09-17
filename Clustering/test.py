from clustering import *

#Number of customers & Total coductor length
f = FeatureSet()
assert np.size(f.get_features(),1) == 2
assert np.size(f.get_features(),0) == 160
assert not np.isnan(f.get_features()).any()
assert f.get_feature("Number of customers").all() == f.get_feature(0).all()
plot_2D_clusters(f,f.hierarchal_clustering())
silhouette_analysis(f,f.hierarchal_clustering(normalized=True))

#Number of customers & Total yearly consumption
h = FeatureSet(include_avg_cons=True, include_total_length=False)
assert np.size(h.get_features(),1) == 2
assert np.size(h.get_features(),0) == 160
assert not np.isnan(h.get_features()).any()
plot_2D_clusters(h,h.hierarchal_clustering())


#Total yearly consumption & Total coductor length
i = FeatureSet(include_avg_cons=True, include_total_length=True, include_n_customer=False)
assert np.size(i.get_features(),1) == 2
assert np.size(i.get_features(),0) == 160
assert not np.isnan(i.get_features()).any()

#Number of customers & Total coductor length & Total yearly consumption
g= FeatureSet(include_avg_cons=True)
assert g.get_feature_list() == ["Number of customers","Yearly consumption per customer (kWh)","Total conductor length (km)"]
assert np.size(g.get_features(),1) == 3
assert np.size(g.get_features(),0) == 160
assert not np.isnan(g.get_features()).any()
assert g.get_feature("Total conductor length (km)").all() == g.get_feature(2).all()
plot_2D_clusters(g,g.hierarchal_clustering(),x_axis="Number of customers",y_axis="Total conductor length (km)")


#Number of customers & Total coductor length & Total yearly consumption & Total reactive power
j= FeatureSet(include_total_length=True,include_avg_cons=True,include_avg_reactive_cons=True)
assert j.get_feature_list() == ["Number of customers","Yearly consumption per customer (kWh)","Yearly reactive consumption per customer (kWh)","Total conductor length (km)"]
assert np.size(j.get_features(),1) == 4
assert np.size(j.get_features(),0) == 160
assert not np.isnan(j.get_features()).any()
plot_2D_clusters(j,j.gaussian_mixture_model(normalized=True,n_repeats=10),x_axis="Yearly consumption per customer (kWh)",y_axis="Total conductor length (km)")


#Main path
k= FeatureSet(include_main_path=True)
assert k.get_feature_list() == ["Number of customers","Total conductor length (km)","Main path length (km)",]
assert np.size(k.get_features(),1) == 3
assert np.size(k.get_features(),0) == 160
assert not np.isnan(k.get_features()).any()
assert k.get_feature("Main path length (km)").all() <= k.get_feature(1).all()
assert k.get_feature("Main path length (km)").all() == k.get_feature(2).all()

#Impedance
l = FeatureSet(include_main_path=True, include_total_impedance=True)
assert l.get_feature_list() == ["Number of customers","Total conductor length (km)","Main path length (km)","Total line impedance (Ohm)"]
assert np.size(l.get_features(),1) == 4
assert np.size(l.get_features(),0) == 160
assert not np.isnan(l.get_features()).any()
assert np.array(l.get_IDs())[l.get_feature(2)==0].all() == np.array([1246503, 1246507, 1464991, 1464997, 1465008, 1440552, 1450257,
       1931561, 1866127,   77873, 2366496, 2366498, 1405071]).all() #Cases with empty LVcustomer
plot_2D_clusters(l, l.gaussian_mixture_model(n_repeats=50),x_axis="Main path length (km)",y_axis="Total line impedance (Ohm)")

