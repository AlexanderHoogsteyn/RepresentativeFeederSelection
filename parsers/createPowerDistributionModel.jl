using JSON
using DataFrames
using CSV
using PowerModelsDistribution
using PowerModels
using Ipopt
using Statistics
using Plots
"""
Parser for distribution network model in .JSON format to
PowerModelsDistribution MATHEMATICAL formultation (more info on this format:
https://lanl-ansi.github.io/PowerModelsDistribution.jl/latest/math-model/).
Time series analysis can be performed (which is currently not natively supported
in PowerDistributionModels, as of sept 2020)
.JSON files should be in a folder, the directory of the folder should be specified below
the configuration file of the feeder to be analyzed as well
Date: 16 sept 2020
Author: Alexander Hoogsteyn
"""
dir = "C:/Users/AlexH/OneDrive/Documenten/Julia/Implementation of network + clustering of network feeders/summer job Alexander/"
feeder = "POLA/65019_74469_configuration.json"

"""
Specify voltage and power base, voltage base should be the phase-to-ground voltage
of the feeder studied (kV), the power base can be arbitrairly chosen (MW)
"""
const voltage_base = 0.230  # (kV)
const power_base = 0.5  # (MW)
const Z_base = voltage_base^2/power_base # (Ohm)

"""
Function that builds a network model in the PowerModelsDistribution format
(mathematical) from a JSON file name & location. Three phase active & reactive power
for all devices in the network can be set using pd & qd respectively (in pu).
"""
function build_mathematical_model(dir, config_file_name,pd=[0.0, 0.0, 0.0],qd=[0.0, 0.0, 0.0],scale_factor=1.0)
    configuration = "star"
    network_model = Dict{String,Any}()
    configuration_json_dict = Dict{Any,Any}()

    network_model["is_kron_reduced"] = true
    network_model["dcline"] = Dict{String,Any}()
    network_model["switch"] = Dict{String,Any}()
    network_model["is_projected"] = true
    network_model["per_unit"] = true
    network_model["data_model"] = MATHEMATICAL
    network_model["shunt"] = Dict{String,Any}()
    network_model["transformer"] = Dict{String,Any}()
    network_model["bus"] = Dict{String,Any}()
    network_model["map"] = Dict{String,Any}()
    network_model["conductors"] = 3
    network_model["baseMVA"] =  power_base
    network_model["basekv"] =  voltage_base
    network_model["bus_lookup"] = Dict{Any,Int64}()
    network_model["load"] = Dict{String,Any}()
    network_model["gen"] = Dict{String,Any}("1" => Dict{String,Any}(
    "pg"            => [1.0, 1.0, 1.0],
    "model"         => 2,
    "connections"   => [1, 2, 3],
    "shutdown"      => 0.0,
    "startup"       => 0.0,
    "configuration" => WYE,
    "name"          => "virtual_generator",
    "qg"            => [1.0, 1.0, 1.0],
    "gen_bus"       => 1,
    "vbase"         =>  voltage_base,
    "source_id"     => "virtual_generator",
    "index"         => 1,
    "cost"          => [0.0, 0.0],
    "gen_status"    => 1,
    "qmax"          => [1.0, 1.0, 1.0],
    "qmin"          => [-1.0, -1.0, -1.0],
    "pmax"          => [1.0, 1.0, 1.0],
    "pmin"          => [-1.0, -1.0, -1.0],
    "ncost"         => 2
    ))
    network_model["settings"] = Dict{String,Any}(
    "sbase_default"        => power_base,
    "vbases_default"       => Dict{String,Any}(), #No default is specified for now, since default is never used
    "voltage_scale_factor" => 1E3, #Voltages are thus expressed in kV
    "sbase"                => power_base,
    "power_scale_factor"   => 1E6, #Power is expressed in MW
    "base_frequency"       => 50.0 #Hertz
    )
    network_model["branch"] = Dict{String,Any}()
    network_model["storage"] = Dict{String,Any}()
    open(dir * config_file_name,"r") do io
    configuration_json_dict = JSON.parse(io)
    end;
    #voltage_base = configuration_json_dict["gridConfig"]["basekV"]
    #power_base = configuration_json_dict["gridConfig"]["baseMVA"]
    configuration = configuration_json_dict["gridConfig"]["connection_configuration"]
    branches_file_name = configuration_json_dict["gridConfig"]["branches_file"]
    buses_file_name = configuration_json_dict["gridConfig"]["buses_file"]
    devices_file_name = configuration_json_dict["gridConfig"]["devices_file"]


    open(dir * buses_file_name,"r") do io
    buses_json_dict = JSON.parse(io)
        for bus in buses_json_dict
            id = bus["busId"] + 1 #Indexing starts at one in Julia
            id_s = string(id)
            network_model["bus_lookup"][id_s] = id
            network_model["settings"]["vbases_default"][id_s] =  voltage_base

            if id == 1 #Settings for slack bus
                network_model["bus"][id_s] = Dict{String,Any}(
                    "name"      => "slack",
                    "bus_type"  => 3,
                    "grounded"  => Bool[0, 0, 0],
                    "terminals" => [1, 2, 3],
                    "vbase"     =>  voltage_base,
                    "index"     => id,
                    "bus_i"     => id,
                    "vmin"      => [0.0, 0.0, 0.0],
                    "vmax"      => [1.5, 1.5, 1.5],
                    "va"        => [0.0, 2.0944, -2.0944],
                    "vm"        => [1.0, 1.0, 1.0])
            else
                network_model["bus"][id_s] = Dict{String,Any}(
                    "name"      => id_s,
                    "bus_type"  => 1,
                    "grounded"  => Bool[0, 0, 0],
                    "terminals" => [1, 2, 3],
                    "vbase"     =>  voltage_base,
                    "index"     => id,
                    "bus_i"     => id,
                    "vmin"      => [0.0, 0.0, 0.0],
                    "vmax"      => [1.5, 1.5, 1.5])
            end;
        end;
    end;

    open(dir * devices_file_name,"r") do io
    devices_json_dict = JSON.parse(io)
      for device in devices_json_dict["LVcustomers"]
        id = device["deviceId"] + 1 #Indexing starts at one in Julia
        id_s = string(id)
        cons = convert(Float64,device["yearlyNetConsumption"])
        network_model["load"][id_s] = Dict{String,Any}(
            "model"         => POWER,
            "connections"   => [2, 1, 3],
            "configuration" => configuration=="star" ? WYE : DELTA,
            "name"          => id_s*"-"*device["coded_ean"],
            "status"        => 1,
            "vbase"         =>  voltage_base,
            "vnom_kv"       => 1.0,
            "source_id"     => device["coded_ean"],
            "load_bus"      => device["busId"] + 1,
            "dispatchable"  => 0,
            "index"         => id,
            "yearlyNetConsumption" => cons,
            "phases"        => device["phases"],
            "pd"            => pd,
            "qd"            => qd
        )
        end;
    end;

    open(dir * branches_file_name,"r") do io
    branches_json_dict = JSON.parse(io)
    impedance_dict = Dict{String,Any}(
    "BT - Desconocido BT" => [0.21, 0.075],
    "BT - MANGUERA" =>[1.23, 0.08],
    "BT - RV 0,6/1 KV 2*16 KAL" => [2.14, 0.09],
    "BT - RV 0,6/1 KV 2*25 KAL" => [1.34, 0.097],
    "BT - RV 0,6/1 KV 3(1*150 KAL) + 1*95 KAL" => [0.2309, 0.085],
    "BT - RV 0,6/1 KV 3(1*240 KAL) + 1*150 KAL" => [0.1602, 0.079],
    "BT - RV 0,6/1 KV 3(1*240 KAL) + 1*95 KAL" => [0.1602, 0.079],
    "BT - RV 0,6/1 KV 4*25 KAL" => [1.34, 0.097],
    "BT - RV 0,6/1 KV 4*50 KAL" => [0.71849, 0.093],
    "BT - RV 0,6/1 KV 4*95 KAL" => [0.3586, 0.089],
    "BT - RX 0,6/1 KV 2*16 Cu" => [1.23, 0.08],
    "BT - RX 0,6/1 KV 2*2 Cu" => [9.9, 0.075],
    "BT - RX 0,6/1 KV 2*4 Cu" => [4.95, 0.075],
    "BT - RX 0,6/1 KV 2*6 Cu" => [3.3, 0.075],
    "BT - RZ 0,6/1 KV 2*16 AL" => [2.14, 0.09],
    "BT - RZ 0,6/1 KV 3*150 AL/80 ALM" => [0.2309, 0.85],
    "BT - RZ 0,6/1 KV 3*150 AL/95 ALM" => [0.2309, 0.85],
    "BT - RZ 0,6/1 KV 3*25 AL/54,6 ALM" => [1.34, 0.097],
    "BT - RZ 0,6/1 KV 3*35 AL/54,6 ALM" => [0.9073, 0.095],
    "BT - RZ 0,6/1 KV 3*50 AL/54,6 ALM" => [0.718497, 0.093],
    "BT - RZ 0,6/1 KV 3*70 ALM/54,6 AL" => [0.4539, 0.091],
    "BT - RZ 0,6/1 KV 3*95 AL/54,6 ALM" => [0.3586, 0.089],
    "BT - RZ 0,6/1 KV 4*16 AL" => [2.14, 0.09],
    "aansluitkabel" => [1.15, 0.150]
    )
        for branch in branches_json_dict
            id = branch["branchId"] +1
            id_s = string(id)
            network_model["branch"][id_s] = Dict{String,Any}(
                "shift"         => [0.0, 0.0, 0.0],
                "f_connections" => [1, 2, 3],
                "name"          => id_s,
                "switch"        => false,
                "g_to"          => [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                "c_rating_a"    => [0.8, 0.8, 0.8],
                "vbase"         =>  voltage_base,
                "g_fr"          => [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                "t_connections" => [1, 2, 3],
                "f_bus"         => branch["upBusId"]+1,
                "b_fr"          => [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                "c_rating_b"    => [0.8, 0.8, 0.8],
                "br_status"     => 1,
                "t_bus"         => branch["downBusId"]+1,
                "b_to"          => [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                "index"         => id,
                "angmin"        => [-1.0472, -1.0472, -1.0472],
                "angmax"        => [1.0472, 1.0472, 1.0472],
                "transformer"   => false,
                "tap"           => [1.0, 1.0, 1.0],
                "c_rating_c"    => [0.8, 0.8, 0.8]
            )
            if haskey(impedance_dict,branch["cableType"])
                network_model["branch"][id_s]["br_r"] = impedance_dict[branch["cableType"]][1] .* (branch["cableLength"]/1000+1E-6) ./  Z_base .* [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
                network_model["branch"][id_s]["br_x"] = impedance_dict[branch["cableType"]][2] .* (branch["cableLength"]/1000+1E-6) ./  Z_base .* [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
            end;
        end;
    end;
    return network_model
end;

"""
Returns an array containing an estimated load_profile based on the households
mean power consumption. The array whithin reference_profiles with the nearest
mean to target_mean is chosen. Units of target_mean and samples within
reference_profiles should be the same (ussualy kWh).
"""
function pick_load_profile(target_mean,reference_profiles)
    load_profile = zeros(Float64,length(reference_profiles[1]))
    smallest_error = Inf
    for i in reference_profiles
        error = abs(target_mean-mean(i))
        if error < smallest_error
            smallest_error = error
            load_profile  = i
        end;
    end;
    return load_profile
end;

function read_reference_profile_from_csv()
end;

"""
Reads out CSV files from dublin data set and returns it as reference profiles
that can be used to form a multinetwork model
"""
function read_dublin_data()
    reference_profiles = []
    for file in 2:2
        csv_name = dir*"DUBLIN/38_CER Electricity_Gas/File"*string(file)*".txt"
        df = DataFrame(CSV.File(csv_name,delim=" ",header=["id","time","power"]))
        data = unstack(df,:time,:id,:power)
        for i in propertynames(data)
            if findfirst(ismissing,data[1:17520,i]) == nothing
                push!(reference_profiles, data[1:17520,i])
            end;
        end;
    end;
    if length(reference_profiles) == 0
        print("NO REFERENCE PROFILES FOUND")
    end;
    return reference_profiles
end;

"""
Builds a multinetwork model in a format similar to the one used in PowerModels
but extended to be able to account for phase inbalance such as in PowerModelsDistribution
Additionaly, it needs reference load profiles to attach time series data to the customers
reference_profiles should by a Array that contains kWh use for each
half hour for an entire year. Optionally, a scaling factor for the profile can be given.
Optionally, the reference_profiles can be a mulidimentional array which contains a
series of load profile Arrays, then a profile that is closest to the customer
is selected (based on the total consumption)
"""
function build_mn_mathematical_model(dir,feeder,reference_profiles,time_steps,scale_factor=1.0,time_unit=0.5)
    if eltype(reference_profiles) == Float64    #If no multidimensional array is given
        reference_profiles = [reference_profiles]   #make it into one
    end;

    network_model = build_mathematical_model(dir,feeder)
    mn_model = replicate(network_model,time_steps,global_keys=Set{String}())

    for (id_s,device) in network_model["load"]
        mean_power = device["yearlyNetConsumption"]*time_unit/365/24 #Assuming yearlyNetConsumption contains total consumption of 365 days in kWh
        load_profile = pick_load_profile(mean_power,reference_profiles)   #Pick the best fit load profile
        load_profile = scale_factor*load_profile/1000/time_unit #scale from kWh to MW
        load_profile = load_profile/power_base #convert to per-uits
        for step in 1:time_steps
            pd = load_profile[step]
            qd = pd/20
            if length(device["phases"]) == 3   #Three phase connection
                mn_model["nw"]["$(step)"]["load"][id_s]["pd"] = pd .* [0.33, 0.33, 0.33]
                mn_model["nw"]["$(step)"]["load"][id_s]["qd"] = qd .* [0.33, 0.33, 0.33]
            elseif device["phases"][1] == 1   #Connected to phase 1
                mn_model["nw"]["$(step)"]["load"][id_s]["pd"] = pd .* [0.0, 1.0, 0.0]
                mn_model["nw"]["$(step)"]["load"][id_s]["qd"] = qd .* [0.0, 1.0, 0.0]
            elseif device["phases"][1] == 2   #Connected to phase 2
                mn_model["nw"]["$(step)"]["load"][id_s]["pd"] = pd .* [1.0, 0.0, 0.0]
                mn_model["nw"]["$(step)"]["load"][id_s]["qd"] = qd .* [1.0, 0.0, 0.0]
            elseif device["phases"][1] == 3   #Connected to phase 3
                mn_model["nw"]["$(step)"]["load"][id_s]["pd"] = pd .* [0.0, 0.0, 1.0]
                mn_model["nw"]["$(step)"]["load"][id_s]["qd"] = qd .* [0.0, 0.0, 1.0]
            else
                print(device["phases"] * " is an unknown phase connection")
            end;
        end;
    end;
    return mn_model
end;

"""
Alias for when a single array is passed instead of a multidimentional array
"""
function build_mn_mathematical_model(dir,feeder,reference_profiles::Array{Float64,1},kwargs...)
    return build_mn_mathematical_model(dir,feeder,reference_profiles=[reference_profiles],kwargs...)
end;

"""
Variant on build_mc_pf() from PowerModelsDistribution that allows you to build
multinetworks, similar to how it is implemented in build_mn_pf() from PowerModels
"""
function build_mn_mc_pf(pm::AbstractPowerModel)
    for (n, network) in nws(pm)
        variable_mc_bus_voltage(pm, nw=n, bounded=false)
        variable_mc_branch_power(pm,nw=n,bounded=false)
        variable_mc_transformer_power(pm, nw=n, bounded=false)
        variable_mc_gen_power_setpoint(pm, nw=n, bounded=false)
        variable_mc_load_setpoint(pm, nw=n, bounded=false)
        variable_mc_storage_power(pm, nw=n, bounded=false)
        constraint_mc_model_voltage(pm,nw=n)
        for (i,bus) in ref(pm, :ref_buses, nw=n)
            @assert bus["bus_type"] == 3
            constraint_mc_theta_ref(pm, i, nw=n)
            constraint_mc_voltage_magnitude_only(pm, i, nw=n)
        end;
        # gens should be constrained before KCL, or Pd/Qd undefined
        for id in ids(pm, :gen, nw=n)
        constraint_mc_gen_setpoint(pm, id, nw=n)
        end;
        # loads should be constrained before KCL, or Pd/Qd undefined
        for id in ids(pm, :load, nw=n)
            constraint_mc_load_setpoint(pm, id, nw=n)
        end;
        for (i,bus) in ref(pm, :bus, nw=n)
            constraint_mc_load_power_balance(pm, i, nw=n)
            # PV Bus Constraints
            if length(ref(pm, :bus_gens, i, nw=n)) > 0 && !(i in ids(pm,:ref_buses, nw=n))
                # this assumes inactive generators are filtered out of bus_gens
                @assert bus["bus_type"] == 2

                constraint_mc_voltage_magnitude_only(pm, i, nw=n)
                for j in ref(pm, :bus_gens, i, nw=n)
                    constraint_mc_gen_power_setpoint_real(pm, j, nw=n)
                end;
            end;
        end;
        for i in ids(pm, :storage, nw=n)
            constraint_storage_state(pm, i, nw=n)
            constraint_storage_complementarity_nl(pm, i, nw=n)
            constraint_mc_storage_losses(pm, i, nw=n)
            constraint_mc_storage_thermal_limit(pm, i, nw=n)
        end;
        for i in ids(pm, :branch, nw=n)
            constraint_mc_ohms_yt_from(pm, i, nw=n)
            constraint_mc_ohms_yt_to(pm, i, nw=n)
        end;
        for i in ids(pm, :transformer, nw=n)
            constraint_mc_transformer_power(pm, i, nw=n)
        end;
    end;
end;
"Alias of build_mn_mc_pf() for when IVR formultation of pf is used"
function build_mn_mc_pf(pm::AbstractIVRModel)
    for (n, network) in nws(pm)
        # Variables
        variable_mc_bus_voltage(pm, nw=n, bounded = false)
        variable_mc_branch_current(pm, nw=n, bounded = false)
        variable_mc_transformer_current(pm, nw=n, bounded = false)
        variable_mc_gen_power_setpoint(pm, nw=n, bounded = false)
        variable_mc_load_setpoint(pm, nw=n, bounded = false)
        # Constraints
        for (i,bus) in ref(pm, :ref_buses, nw=n)
            @assert bus["bus_type"] == 3
            constraint_mc_theta_ref(pm, i, nw=n)
            constraint_mc_voltage_magnitude_only(pm, i, nw=n)
        end;
        # gens should be constrained before KCL, or Pd/Qd undefined
        for id in ids(pm, :gen, nw=n)
            constraint_mc_gen_setpoint(pm, id, nw=n)
        end;
        # loads should be constrained before KCL, or Pd/Qd undefined
        for id in ids(pm, :load, nw=n)
            constraint_mc_load_setpoint(pm, id, nw=n)
        end;
        for (i,bus) in ref(pm, :bus, nw=n)
            constraint_mc_load_current_balance(pm, i, nw=n)
            # PV Bus Constraints
            if length(ref(pm, :bus_gens, i, nw=n)) > 0 && !(i in ids(pm,:ref_buses, nw=n))
                # this assumes inactive generators are filtered out of bus_gens
                @assert bus["bus_type"] == 2
                constraint_mc_voltage_magnitude_only(pm, i, nw=n)
                for j in ref(pm, :bus_gens, i, nw=n)
                    constraint_mc_gen_power_setpoint_real(pm, j, nw=n)
                end;
            end;
        end;
        for i in ids(pm, :branch, nw=n)
            constraint_mc_current_from(pm, i, nw=n)
            constraint_mc_current_to(pm, i, nw=n)

            constraint_mc_bus_voltage_drop(pm, i, nw=n)
        end;

        for i in ids(pm, :transformer)
            constraint_mc_transformer_power(pm, i, nw=n)
        end;
    end;
end;

"Alias to run multinetwork powerflow using run_mc_model() from PowerModelsDistribution"
function run_mn_mc_pf(data::Union{Dict{String,<:Any},String}, model_type::Type, solver; kwargs...)
    return run_mc_model(data, model_type, solver, build_mn_mc_pf, multiconductor=true, multinetwork=true, kwargs...)
end;

"Run time series analysis of the specified feeder in POLA using Dublin data load profiles as reference"
function time_series_analysis()
    reference_profiles = read_dublin_data()
    multinetwork_model = build_mn_mathematical_model(dir,feeder,reference_profiles,48)
    solver = with_optimizer(Ipopt.Optimizer, print_level = 3, tol=1e-9)
    return run_mn_mc_pf(multinetwork_model,ACPPowerModel,solver)
end

"Run Time series analysis by simply rerunning PowerModelsDistribution for each time step "
function naive_time_series_analysis()
    reference_profiles = read_dublin_data()
    multinetwork_model = build_mn_mathematical_model(dir,feeder,reference_profiles,48)
    solver = with_optimizer(Ipopt.Optimizer, print_level = 1, tol=1e-9)
    result = Dict{String,Any}()
    for (n,network) in multinetwork_model["nw"]
        network["per_unit"] = true
        result[n] = run_mc_pf(network,ACPPowerModel, solver)
    end;
    return result
end;
