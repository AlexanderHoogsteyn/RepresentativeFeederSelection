# createPowerModelDistributionFeeder.jl
Script that contains functionalities to use GridDataCollections in PowerModelsDistribution. This makes it possible to do powerflow calculations on feeders in distribution networks. To do calculations in PowerModelsDistribution the network needs to be in a Julia dictonary in specific format (more info on this format: https://lanl-ansi.github.io/PowerModelsDistribution.jl/latest/math-model/). This script parses the GridDataCollections to this specific Julia dictonary. Using this script, time series analysis can be performed (which is currently not natively supported in PowerDistributionModels, as of sept 2020).

## Obtaining GridDataCollections
This scripts imports the network topology from JSON files in a specific format. Such a representation of a distribution network can be obtained using a script
such as createGridDataCollection_dataframes_poly.py. That script in specific does the conversion from excel files as they were recieved by the Spanish DSO. Each feeder
has it's information stored in 4 files:
- Configuration: contains general information and lists the directory's of the other 3 files.
- Devices: contains information about the devices i.e. customers connected. For example the EAN-number and connection capacity. The total yearly consumption (kWh) is included, in createGridDataCollection_dataframes_poly.py this is estimated based of 20 days of smart meter data. The phase connection is as for now, randomly allocated.
- Branches: contains information on cable length, bus connections and type of each branch. The latter is used to determine the impedance of the branch
- Buses: contains info on which bus is the slack bus and voltage limits on each bus. The voltage limits are added to model but are are only used if optimal powerflow calculations are performed.
This script is however not supposed to be used for optimal powerflow calculations. If this is of interest, additional fields will have to be added to the data model.
The .JSON files should be in a folder, the directory of the folder should be specified on top of the script. As well as the configuration file of the feeder to be analyzed.
```Julia
dir = "C:Home/User/folder/"
config_file_name = "CITY/feeder-name_configuration.json"
```

## Building PowerModelsDistribution MATHEMATICAL model
If a powerflow calculation is required for a single instance in time the function
build_mathematical_model() can be used. Three phase active & reactive power
for all devices in the network can be set using pd & qd respectively (in pu)
```Julia
network = build_mathematical_model(dir, config_file_name)
solver = with_optimizer(Ipopt.Optimizer, print_level = 0, tol=1e-9)
result = run_mc_pf(network,,ACPPowerModel,solver)
```
### Limitations
- The impedance matrix is diagonal for each branch since of each cable type only the R/km and X/km were known. Mutual Impedance is not accounted for.
- R/km and X/km are hard coded in the script for the specific cable types in the Spanish data set. If other cable types are used this needs adaption.
- The impedance that the transformer at the head of the feeder adds, is not accounted for.
- Ground admittance is not accounted for

## Time series analysis
To perform a time series analysis, load profiles should be estimated and assigned to each device i.e. customer. Load profiles can be imported from csv files using read_reference_profile_from_csv() or using read_dublin_data(). The resulting data format is an array of samples of used energy (kWh). Or in case of read_dublin_data() an array of array's with samples of used energy. This needs to be passed on as argument to build_mn_mathematical_model(). The time interval of the samples (expressed in hours) should also be passed, this is needed to convert energy to power. If a load profile with instantaneous power samples is used, set the time interval to 1. A scale factor can also be applied to the load profile.
```Julia
reference_profiles = read_dublin_data()
time_steps = 48 #one day
multinetwork = build_mn_mathematical_model(dir, config_file_name, reference_profiles, time_steps, scale_factor=1.0, time_unit=0.5)
result = run_mn_mc_pf(multinetwork,ACPPowerModel,solver)
```
In PowerModels, the package where PowerModelsDistribution is based on, there is support for multinetworks which are networks that change for example their load over time i.e. it is possible to do time series analyses.
run_mn_mc_pf() will perform a multinetwork variant of the multiconductor powerflow problem (Thus taking into account phase inbalances). This is currently not natively supported in PowerModelsDistribution but this script includes the functions needed to build a multinetwork formulation of the multiconductor powerflow problem

### Limitaions
- Devices that have a three phase connection have their power equally divided over the three phases
- The network topology is fixed during the time series analysis. Influence of tap changes or switches in the topology are not possible in its current form.
- Writing back the solution takes a lot of time, this mitigates the advantage of using a multinetwork variant. On some machines it can be faster to use naive_time_series_analysis() instead. This implementation just performs a power flow calculation for each time step.   
