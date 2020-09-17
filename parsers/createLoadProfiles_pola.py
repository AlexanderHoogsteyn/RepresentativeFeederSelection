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
    Removes outliers in "Activa E" and "S", "Reactiva 1" and "4"
    """
    smartmeter_data = pd.DataFrame()
    for m in range(1, 8):
        csv = griddata_dir + "test_network/Sim_files_190128_OK_V0/GIS_data/file" + str(m) + ".csv"
        if not os.path.exists(csv):  # Check whether path exist
            raise NameError("Path doesn't exist")
        else:
            smartmeter_data = smartmeter_data.append(
                pd.read_csv(csv, sep=";", skipinitialspace=True, decimal=',', thousands=' '))
            smartmeter_data["Activa E"].mask(smartmeter_data["Activa E"] > 200, np.nan, inplace=True)  # remove outliers
            smartmeter_data["Activa S"].mask(smartmeter_data["Activa S"] > 200, np.nan, inplace=True)
            smartmeter_data["Reactiva1"].mask(smartmeter_data["Reactiva1"] > 200, np.nan, inplace=True)  # remove outliers
            smartmeter_data["Reactiva4"].mask(smartmeter_data["Reactiva4"] > 200, np.nan, inplace=True)

    # active_cons_dict = loadProfile.groupby("Referencia")[["Activa E", "Activa S", "Reactiva1"]].mean()
    # active_cons_dict += loadProfile.groupby("Referencia")["Activa S"].mean()    #include night tariff in yearly consumption

    # active_cons_dict = active_cons_dict[~np.isnan(active_cons_dict)]
    # active_cons_dict = active_cons_dict * 24 * 20 * 15  # 24 samples per day, 20 days of data available, *15 to estimate yearly (300 days) consumption
    return smartmeter_data


def read_times_demand_data(filename="Hourly_profile.xlsx"):
    """
    Reads the TIMES demand sheet and parses it to a DataFrame
    """
    path = griddata_dir + "parsers/" + filename
    if not os.path.exists(path):  # Check whether path exist
        raise NameError("Path doesn't exist")
    else:
        times_data = pd.read_excel(path, sheet_name="Demand", header=[1, 3], index_col=0)
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
    Only day tariff is taken into account for now
    the smartdata is sorted because its samples are not in chronological order as they were recieved by the DSO
    TO DO: The load profiles don't look correct, my guess is that the sorting is not done correctly (perhaps the dates
    are sorted alphabetically instead of chronologically)
    """
    load_profiles = pd.DataFrame()
    grouped = smartmeter_data.groupby("Referencia")
    for id, group in grouped:
        long_load_profile = group.sort_values("Dia")["Activa E"]
        load_profile = [np.nanmean(long_load_profile[i:20*24:24]) for i in range(0, 24)]
        load_profiles[id] = load_profile
    return load_profiles

def estimate_scale_factors(times_demand_data):
    """
    Constructs a dictonary that contains scale factors for the different clusters for the different years
    TO DO: Find a baseline for the 10 clusters, otherwise the other years cannot be scaled proportionally
    """
    scale_factors = {"2020" : np.ones(10)}
    #TO DO: set scale_factors["2030"] and scale_factors["2050"] appropriate
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
    cluster = round(day_of_year/35)
    return cluster

def concatenate_daily_profiles(times_demand_data,load_profiles):
    """
    concatenates the daily load profiles togheter for an entire year. and writes to csv file
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
        for day in range(0,366):
            profile_day = determine_scale(id, day_of_year=day) * determine_shape(id, load_profiles, day_of_year=day)
            profile_year = np.append(profile_year,profile_day)
        filename = griddata_dir + "Load profiles POLA/2030/profile_" + id + "_2030.csv"
        np.savetxt(filename,profile_year,delimiter=",")

    #repeat for 2050
    for id, content in load_profiles.items():
        profile_year = np.array([])
        for day in range(0,366):
            profile_day = determine_scale(id, day_of_year=day) * determine_shape(id, load_profiles, day_of_year=day)
            profile_year = np.append(profile_year,profile_day)
        filename = griddata_dir + "Load profiles POLA/2050/profile_" + id + "_2050.csv"
        np.savetxt(filename,profile_year,delimiter=",")
    return None

smartmeter_data = read_smartmeter_data()
times_demand_data = read_times_demand_data()
times_generation_data = read_times_generation_data()
load_profiles = estimate_daily_load_profiles(smartmeter_data)
concatenate_daily_profiles(times_demand_data,load_profiles)