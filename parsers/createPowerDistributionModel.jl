using JSON
using DataFrames
using CSV
using PowerModelsDistribution
using Ipopt
using Query
using Statistics

dir = "C:/Users/AlexH/OneDrive/Documenten/Julia/Implementation of network + clustering of network feeders/summer job Alexander/"
feeder = "POLA/65028_84569_configuration.json"

voltage_base = 0.230  # (kV)
power_base = 0.5  # (MW)
Z_base = voltage_base^2/power_base # (Ohm)

function build_mathematical_network_model(dir, config_file_name, time_step,reference_profiles,scale_factor=1.0)
    #
    #   Function that builds a network model in the PowerModelsDistribution format
    #   (mathematical) from a JSON file name & location
    #   Additionaly, it needs reference load profiles to attach time series data to the customers
    #   reference_profiles should by a Array that contains kWh use for each
    #   half hour for an entire year. Optionally, a scaling factor for the profile can be given.
    #   Optionally, the reference_profiles can be a mulidimentional array which contains a
    #   series of load profile Arrays, then a profile that is closest to the customer
    #   is selected (based on the total consumption)
    #
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
    network_model["baseMVA"] = power_base
    network_model["basekv"] = voltage_base
    network_model["bus_lookup"] = Dict{Any,Int64}()
    network_model["load"] = Dict{String,Any}()
    network_model["gen"] = Dict{String,Any}() # For now not implemented since there is no PV in the system
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
    voltage_base = configuration_json_dict["gridConfig"]["basekV"]
    power_base = configuration_json_dict["gridConfig"]["baseMVA"]
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
            network_model["settings"]["vbases_default"][id_s] = voltage_base

            if id == 1 #Settings for slack bus
                network_model["bus"][id_s] = Dict{String,Any}(
                    "name"      => "slack",
                    "bus_type"  => 3,
                    "grounded"  => Bool[0, 0, 0],
                    "terminals" => [1, 2, 3],
                    "vbase"     => voltage_base,
                    "index"     => id,
                    "bus_i"     => id,
                    "vmin"      => [0.0, 0.0, 0.0],
                    "vmax"      => [1.5, 1.5, 1.5],
                    "va"        => [0.0, -2.0944, -2.0944],
                    "vm"        => [1.0, 1.0, 1.0])
            else
                network_model["bus"][id_s] = Dict{String,Any}(
                    "name"      => id_s,
                    "bus_type"  => 1,
                    "grounded"  => Bool[0, 0, 0],
                    "terminals" => [1, 2, 3],
                    "vbase"     => voltage_base,
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
            "vbase"         => voltage_base,
            "vnom_kv"       => 1.0,
            "source_id"     => device["coded_ean"],
            "load_bus"      => device["busId"] + 1,
            "dispatchable"  => 0,
            "index"         => id,
            "yearlyNetConsumption" => cons,
            "phases"        => device["phases"]
        )
        if eltype(reference_profiles) == Float64    #If no multidimensional array is given
            reference_profiles = [reference_profiles]   #make it into one
        end;
        load_profile = pick_load_profile(cons,reference_profiles)   #Pick the best fit load profile
        load_profile = scale_factor*load_profile/500 #scale from kWh to MW
        pd = load_profile[time_step]/power_base #convert to per-uits
        qd = pd/20 #Estimation for reactive power

        if length(device["phases"]) == 3   #Three phase connection
            network_model["load"][id_s]["pd"] = pd .* [0.33, 0.33, 0.33]
            network_model["load"][id_s]["qd"] = qd .* [0.33, 0.33, 0.33]
        elseif device["phases"][1] == 1   #Connected to phase 1
            network_model["load"][id_s]["pd"] = pd .* [0.0, 1.0, 0.0]
            network_model["load"][id_s]["qd"] = qd .* [0.0, 1.0, 0.0]
        elseif device["phases"][1] == 2   #Connected to phase 2
            network_model["load"][id_s]["pd"] = pd .* [1.0, 0.0, 0.0]
            network_model["load"][id_s]["qd"] = qd .* [1.0, 0.0, 0.0]
        elseif device["phases"][1] == 3   #Connected to phase 3
            network_model["load"][id_s]["pd"] = pd .* [1.0, 0.0, 0.0]
            network_model["load"][id_s]["qd"] = qd .* [1.0, 0.0, 0.0]
        else
            print(device["phases"] * " is an unknown phase connection")
        end;
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
            id = branch["branchId"]
            id_s = string(id)
            network_model["branch"][id_s] = Dict{String,Any}(
                "shift"         => [0.0, 0.0, 0.0],
                "f_connections" => [1, 2, 3],
                "name"          => id_s,
                "switch"        => false,
                "g_to"          => [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                "c_rating_a"    => [0.8, 0.8, 0.8],
                "vbase"         => voltage_base,
                "g_fr"          => [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                "t_connections" => [1, 2, 3],
                "f_bus"         => branch["upBusId"]+1,
                "b_fr"          => [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                "c_rating_b"    => [1.2, 1.2, 1.2],
                "br_status"     => 1,
                "t_bus"         => branch["downBusId"]+1,
                "b_to"          => [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0],
                "index"         => 1,
                "angmin"        => [-1.0472, -1.0472, -1.0472],
                "angmax"        => [1.0472, 1.0472, 1.0472],
                "transformer"   => false,
                "tap"           => [1.0, 1.0, 1.0],
                "c_rating_c"    => [1.2, 1.2, 1.2]
            )
            if haskey(impedance_dict,branch["cableType"])
                network_model["branch"][id_s]["br_r"] = impedance_dict[branch["cableType"]][1] .* branch["cableLength"] ./ Z_base .* [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
                network_model["branch"][id_s]["br_x"] = impedance_dict[branch["cableType"]][2] .* branch["cableLength"] ./ Z_base .* [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
            end;
        end;
    end;
    return network_model
end;

function update_mathematical_network_model(network_model,time_step,reference_profiles,scale_factor)
    updated_network_model = network_model
    if eltype(reference_profiles) == Float64    #If no multidimensional array is given
        reference_profiles = [reference_profiles]
    end;   #make it into one
        for (id_s,device) in network_model["load"]
            load_profile = pick_load_profile(device["yearlyNetConsumption"],reference_profiles)   #Pick the best fit load profile
            load_profile = scale_factor*load_profile/500 #scale from kWh to MW
            pd = load_profile[time_step]/power_base #convert to per-uits
            qd = pd/20 #Estimation for reactive power

            if length(device["phases"]) == 3   #Three phase connection
                updated_network_model["load"][id_s]["pd"] = pd .* [0.33, 0.33, 0.33]
                updated_network_model["load"][id_s]["qd"] = qd .* [0.33, 0.33, 0.33]
            elseif device["phases"][1] == 1   #Connected to phase 1
                updated_network_model["load"][id_s]["pd"] = pd .* [0.0, 1.0, 0.0]
                updated_network_model["load"][id_s]["qd"] = qd .* [0.0, 1.0, 0.0]
            elseif device["phases"][1] == 2   #Connected to phase 2
                updated_network_model["load"][id_s]["pd"] = pd .* [1.0, 0.0, 0.0]
                updated_network_model["load"][id_s]["qd"] = qd .* [1.0, 0.0, 0.0]
            elseif device["phases"][1] == 3   #Connected to phase 3
                updated_network_model["load"][id_s]["pd"] = pd .* [1.0, 0.0, 0.0]
                updated_network_model["load"][id_s]["qd"] = qd .* [1.0, 0.0, 0.0]
            else
                print(device["phases"] * " is an unknown phase connection")
            end;
        end;
    return updated_network_model
end;

function pick_load_profile(yearly_consumption,reference_profiles)
    #
    # Returns an array containing an estimated load_profile based on the households
    # yearly consumption. The array consists of 30min samples of the used Energy (kWh)
    # for exactly one year (thus 17520 samples)
    #
    load_profile = zeros(Float64,48*365)
    smallest_error = Inf
    for i in reference_profiles
        mean_consumption = convert(Float64, mean(i)*48*365)
        error = abs(yearly_consumption-mean_consumption)
        if error < smallest_error
            smallest_error = error
            load_profile  = i
        end;
    end;
    return load_profile
end;

function read_dublin_data()
    reference_profiles = []
    for file in 1:6
        csv_name = dir*"DUBLIN/38_CER Electricity_Gas/File"*string(file)*".txt"
        df = DataFrame(CSV.File(csv_name,delim=" ",header=["id","time","power"]))
        data = unstack(df,:time,:id,:power)
        for i in propertynames(data)
            if findfirst(ismissing,data[1:17520,i]) == nothing
                #mean_power = convert(Float64, mean(data[1:17520,i])*48*365)
                push!(reference_profiles, data[1:17520,i])
            end;
        end;
    end;
    if length(reference_profiles) == 0
        print("NO REFERENCE PROFILES FOUND")
    end;
    return reference_profiles
end;

function time_code(decoded_time)
    half_hours = decoded_time % 48
    days = decoded_time ÷ 48
    return 100*days + half_hours
end;

function decode_time_code(time_code )
    time = (time_code % 100)
    day = (time_code ÷ 100) % 365

    return day*48+time
end;



#reference_profiles = read_dublin_data()
time_series_solution = Dict{Int32,Any}()
for time_step in 1:1
    network_model = build_mathematical_network_model(dir,feeder,time_step,reference_profiles,1E6)
    solver = with_optimizer(Ipopt.Optimizer, print_level = 0, tol=1e-6)
    result = run_mc_pf(network_model, ACPPowerModel, solver)
    time_series_solution[time_step] = result
end;
