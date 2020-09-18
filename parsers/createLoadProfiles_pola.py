import json
import random
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

griddata_dir = "C:/Users/AlexH/OneDrive/Documenten/Julia/Implementation of network + clustering of network feeders/summer job Alexander/"


def read_smartmeter_data():
    """
    Reads the csv files with 20 day load profiles from smart meter data
    Parses all information to a DataFrame (including ID, reactive power (kWh),...)
    """
    smartmeter_data = pd.DataFrame()
    for m in range(1, 8):
        csv = griddata_dir + "test_network/Sim_files_190128_OK_V0/GIS_data/file" + str(m) + ".csv"
        if not os.path.exists(csv):  # Check whether path exist
            raise NameError("Path doesn't exist")
        else:
            smartmeter_data = smartmeter_data.append(
                pd.read_csv(csv, sep=";", skipinitialspace=True, decimal=',', thousands=' '))

            #smartmeter_data["Activa E"].mask(smartmeter_data["Activa E"] > 10, np.nan, inplace=True)  # remove Feeder transformer measurements
            #smartmeter_data["Activa S"].mask(smartmeter_data["Activa S"] > 10, np.nan, inplace=True)
            #smartmeter_data["Reactiva1"].mask(smartmeter_data["Reactiva1"] > 10, np.nan, inplace=True)
            #smartmeter_data["Reactiva4"].mask(smartmeter_data["Reactiva4"] > 10, np.nan, inplace=True)

    # active_cons_dict = loadProfile.groupby("Referencia")[["Activa E", "Activa S", "Reactiva1"]].mean()
    # active_cons_dict += loadProfile.groupby("Referencia")["Activa S"].mean()    #include night tariff in yearly consumption

    # active_cons_dict = active_cons_dict[~np.isnan(active_cons_dict)]
    # active_cons_dict = active_cons_dict * 24 * 20 * 15  # 24 samples per day, 20 days of data available, *15 to estimate yearly (300 days) consumption
    return smartmeter_data


def read_times_demand_data(filename="Hourly_profile.xlsx",scenario="2030"):
    """
    Reads the TIMES demand sheet and parses it to a DataFrame
    """
    path = griddata_dir + "parsers/" + filename

    if not os.path.exists(path):  # Check whether path exist
        raise NameError("Path doesn't exist")
    else:
        if scenario == "2030":
            times_data = pd.read_excel(path, sheet_name="Demand", header=[3],usecols="A,C:L",  index_col=0)
        elif scenario == "2050":
            times_data = pd.read_excel(path, sheet_name="Demand", header=[3],usecols="A,N:W", index_col=0)
        else:
            raise AttributeError("scenario does not exist")
    times_data = times_data.dropna(axis='index', how='all')
    times_data = times_data.dropna(axis='columns', how='all')

    return times_data


def read_times_generation_data(filename="Hourly_profile.xlsx"):
    """
    Reads the TIMES generation sheet and parses it to a DataFrame
    """
    path = griddata_dir + "parsers/" + filename
    if not os.path.exists(path):  # Check whether path exist
        raise NameError("Path doesn't exist")
    else:
        times_data = pd.read_excel(path, sheet_name="Generation", header=[0, 1, 2], index_col=0)
    times_data = times_data.dropna(axis='index', how='all')
    times_data = times_data.dropna(axis='columns', how='all')
    return times_data


def estimate_daily_load_profiles(smartmeter_data):
    """
    Build a Dataframe with EAN numbers as column names and 24 samples each containing 1h samples of the active power
    Day and night tariff measurements are added
    the smartdata needs to be sorted because its samples are not in chronological order as they were recieved by the DSO
    TO DO: The smartdata needs to be sorted first. The implementation I had was way too slow, so I commented
    it out for now
    """
    load_profiles = pd.DataFrame()
    grouped = smartmeter_data.groupby("Referencia")
    for id, group in grouped:
        #group["Fecha"] = pd.to_datetime(group["Fecha"])
        #group.sort_values("Fecha",inplace=true)
        long_load_profile = group["Activa E"] + group["Activa S"]
        if long_load_profile.median() < 5: #Elliminate feeder transformer measurements
            load_profile = []
            last_value = 0
            for i in range(0,24):
                value = np.median(long_load_profile[i:20*24:24])
                if value > 10:      # Remove outliers by replacing by the last valid sample
                    value = last_value
                else:
                    last_value = value
                load_profile.append(value)
            load_profiles[id] = load_profile
    return load_profiles

def estimate_scale_factors(times_demand_data_2030,times_demand_data_2050,load_profiles,time_dependent=False):
    """
    Constructs a dictonary that contains scale factors for the different clusters for the different years
    The factor represents the magnitude of a load profile in a certain cluster relative to the magnitude currently
    by putting time_dependent to true or false you can choose to just use 1 average for each cluster
    or 12 2h samples for each cluster.
    TO DO: With the data I have I cannot determine how to weigh the clusters in the current scenario
    TO DO: Implementation that scales every time step instead of according to the average increase
    """
    scale_factors = dict()

    total_pola_consumption = load_profiles.apply(np.nanmean) #(kW)
    total_belgium_consumption = total_pola_consumption.sum()*100 #(kW)
    total_belgium_consumption = total_belgium_consumption*(1e-6) #(GW)
#    for column ,content in times_demand_data.items():
    #TO DO: set scale_factors["2030"] and scale_factors["2050"] appropriate
    cluster_weigths = np.zeros(10)
    for day in range(0,365):
        cluster_weigths[get_cluster(day)] += 1
    cluster_weigths = cluster_weigths/365
    factors_2030 = times_demand_data_2030.apply(sum,axis=1)
    factors_2050 = times_demand_data_2050.apply(sum,axis=1)
    factors_2030_avg = np.array([sum(factors_2030[i * 12:i * 12 + 12]) / 12 for i in range(0, 10)])
    factors_2050_avg = np.array([sum(factors_2050[i * 12:i * 12 + 12]) / 12 for i in range(0, 10)])

    weighted_total_2030 = sum(np.array([sum(factors_2030[i*12:i*12+12])/12 for i in range(0,10)])*cluster_weigths)
    weighted_total_2050 = sum(np.array([sum(factors_2050[i*12:i*12+12])/12 for i in range(0,10)])*cluster_weigths)
    if time_dependent:
        scale_factors["2020"] = np.ones(120)
        scale_factors["2030"] = np.array(factors_2030)
        scale_factors["2050"] = np.array(factors_2050)
    else:
        scale_factors["2020"] = np.ones(10)
        scale_factors["2030"] = factors_2030_avg/total_belgium_consumption
        scale_factors["2050"] = factors_2050_avg/total_belgium_consumption
    return scale_factors

def determine_shape(id, load_profiles, day_of_year=1):
    """
    Function that determines the shape of the load profile based on customer id and the day of the year.
    For now the shape does not vary throughout the year
    """
    return load_profiles[id]

def determine_scale(id, scale_factors, day_of_year=1,scenario="2020"):
    """
    Function that determines a scale factor based on the day of the year
    """
    cluster = get_cluster(day_of_year)
    scale = scale_factors[scenario][cluster]
    return scale

def get_cluster(day_of_year):
    """
    A more thoughtfull implementation needs to be done to map the day of the year to which of the 10 clusters a day belongs
    """
    cluster = round(day_of_year/40)
    return cluster

def concatenate_daily_profiles(times_demand_data,load_profiles,scale_factors):
    """
    concatenates the daily load profiles togheter for an entire year. and writes to csv file
    TO DO: Although that time varying scale factors are already calculated they are not used yet
    to build "profile_day"
    TO DO: creating thousands of csv files takes a lot of time, are there functions that are faster
    then savetxt from Numpy
    """
    for id, content in load_profiles.items():
        profile_year = np.array([])
        for day in range(0,366):
            profile_day = determine_scale(id, day_of_year=day) * determine_shape(id, load_profiles, day_of_year=day)
            profile_year = np.append(profile_year,profile_day)
        filename = griddata_dir + "Load profiles POLA/2020/profile_" + id + "_2020.csv"
        np.savetxt(filename,profile_year,delimiter=",")

    # Repeat for 2030
    for id, content in load_profiles.items():
        profile_year = np.array([])
        for day in range(0,365):
            profile_day = determine_scale(id, scale_factors, day_of_year=day,scenario="2030") * determine_shape(id, load_profiles, day_of_year=day)
            profile_year = np.append(profile_year,profile_day)
        filename = griddata_dir + "Load profiles POLA/2030/profile_" + id + "_2030.csv"
        np.savetxt(filename,profile_year,delimiter=",")

    #repeat for 2050
    for id, content in load_profiles.items():
        profile_year = np.array([])
        for day in range(0,366):
            profile_day = determine_scale(id, day_of_year=day,scenario="2050") * determine_shape(id, load_profiles, day_of_year=day)
            profile_year = np.append(profile_year,profile_day)
        filename = griddata_dir + "Load profiles POLA/2050/profile_" + id + "_2050.csv"
        np.savetxt(filename,profile_year,delimiter=",")
    return None

#smartmeter_data = read_smartmeter_data()
#times_demand_data_2030 = read_times_demand_data()
#times_demand_data_2050 = read_times_demand_data(scenario=2050)
#times_generation_data = read_times_generation_data()
#load_profiles = estimate_daily_load_profiles(smartmeter_data)
#scale_factors = estimate_scale_factors(times_demand_data_2030,times_demand_data_2050,load_profiles)
#concatenate_daily_profiles(times_demand_data,load_profiles)