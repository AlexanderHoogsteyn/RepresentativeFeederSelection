'''
Created on 5 mrt. 2018

@author: dhulstr
'''


import json
import random
import numpy as np
import sys
import os, os.path
import errno
import pandas as pd
import time
from itertools import islice

#from dataprocessing.data_constants import ADRIAN_DATA_DIR as data_dir
#from dataprocessing.data_constants import ADRIAN_SOURCE_GRID_DATA_DIR as source_griddata_dir

#griddata_dir =  os.path.join(data_dir,
#                                    'grid data',
#                                    'gridDataFiles',
 #                                   'Vlaanderen_statSector') #Z:\\grid data\\gridDataFiles\\temp_gridData_Files'
#griddata_dir += os.sep

#griddata_dir= 'C:/Users/karpan/Documents/grid/'
griddata_dir= "C:/Users/AlexH/OneDrive/Documenten/Julia/Implementation of network + clustering of network feeders/summer job Alexander/"
#%%
def loadGridFile(jsonFileNameList):

    gridFormatDict = dict()
    for jsonFileName in  jsonFileNameList:
        with open(jsonFileName) as grid_file:
            gridFormatDict_fromfile = json.load(grid_file, strict=False)

        gridFormatDict = {**gridFormatDict, **gridFormatDict_fromfile}

    return gridFormatDict

#%%
def parseGridData(gridFormatDict):
    '''
    if 'Cabine' in gridFormatDict.keys():
        cabines = gridFormatDict["Cabine"]
    elif 'MS Cabines' in gridFormatDict.keys():
        cabines = gridFormatDict["MS Cabines"]
    cabines_df = pd.DataFrame(cabines)

    if 'Transformer' in gridFormatDict.keys():
        transformers = gridFormatDict["Transformer"]
    elif 'Transformers' in gridFormatDict.keys():
        transformers = gridFormatDict["Transformers"]
    transformers_df = pd.DataFrame(transformers)

    if 'Circuit' in gridFormatDict.keys():
        circuits = gridFormatDict["Circuit"]
    elif 'Circuits' in gridFormatDict.keys():
        circuits = gridFormatDict["Circuits"]
    circuits_df = pd.DataFrame(circuits)



    if 'Cable' in gridFormatDict.keys():
        cables = gridFormatDict["Cable"]
    elif 'LS Cables' in gridFormatDict.keys():
        cables = gridFormatDict['LS Cables']
    cables_df = pd.DataFrame(cables)

    if 'Connection' in gridFormatDict.keys():
        connections = gridFormatDict["Connection"]
    elif 'LS Connections' in gridFormatDict.keys():
        connections = gridFormatDict['LS Connections']
    connections_df = pd.DataFrame(connections)

    if 'Customer' in gridFormatDict.keys():
        customers = gridFormatDict["Customer"]
    elif 'Customers' in gridFormatDict.keys():
        customers = gridFormatDict['Customers']
    customers_df = pd.DataFrame(customers)
    '''
    c_nr = 1
    circuit_nr = 1
    for cab_index, cab_row in islice(cabines_df.iterrows(), 0, None): #for c in cabines[1027:]:#[0:10]:
        #print('###### cabine nr: ',c_nr)
        c_id = cab_row["id"]#c["id"]
        #cabine_city = c["city"]
        #cab_trafos = [trafo for trafo in transformers if trafo['cabineId'] == c_id]
        cab_trafos = transformers_df[transformers_df["cabineId"]==c_id]
        for trafo_index, trafo_row in cab_trafos.iterrows(): #for trafo in cab_trafos:
            tr_id = trafo_row["id"]#trafo['id']
            #trafo_circuits = [circ for circ in circuits if circ['transfoId'] == tr_id]
            trafo_circuits = circuits_df[circuits_df["transfoId"]==tr_id]
            for circ_index, circ_row in trafo_circuits.iterrows():#for circuit in trafo_circuits:
                circuit_name = circ_row["name"]#circuit['name']
                #print('circuit name: ' + circuit_name)
                #### lists where grid information should be parsed to
                bus_list = []
                branch_list = []
                devices_dict = dict()
                devices_dict['LVcustomers'] = []
                devices_dict['solarGens'] = []


                ## add slack bus to bus list
                bus_dict = dict()
                bus_dict['busId'] = 0
                bus_dict['slackBus'] = True
                bus_list.append(bus_dict)

                ### find cables in circuit (in correct order)
                circuit_id = circ_row['id']#circuit['id']
                if 'circuitVoltage' in trafo_circuits.columns: #circuit.keys():
                    circuit_voltage = circ_row['circuitVoltage']#circuit['circuitVoltage']
                else:
                    circuit_voltage = 400
                    print('unknown circuit voltage, value not given, 400V assumed')

                #circuit_cables = [cable for cable in cables if cable['circuitId'] == circuit_id]
                circuit_cables = cables_df[cables_df['circuitId']==circuit_id]#[cable for cable in cables if cable['circuitId'] == circuit_id]
                #first cable has empty 'prev' list
                #first_cable = next((cable for cable in circuit_cables if cable["prev"] == []), False)
                first_cable = circuit_cables[circuit_cables["prev"].apply(np.size)==0]
#                 ### to debug eandis case ###
#                 if not first_cable:
#                     #first_cable = next((cable for cable in circuit_cables if cable["prev"] == cable['id']), False)
#                     first_cable = circuit_cables[circuit_cables["prev"]==circuit_cables['id']] ##don't know if this works

                if len(first_cable)>0:
                    bus_list, branch_list, devices_dict = addNextCable(first_cable['id'].values[0], circuit_cables, connections_df, customers_df, bus_list, branch_list, devices_dict,active_cons_dict)
                else:
                    print('no cable present with prev == [] for circuitId: ', circuit_id)
                #print(bus_list)
                #print(branch_list)
                #print(devices_dict)
                #print('circuit done')
                print('circuit nr:', circuit_nr)
                circuit_nr = circuit_nr + 1

                printGridDataToFiles(bus_list, branch_list, devices_dict, circ_row, trafo_row, cab_row, type='feeder_level')
        c_nr = c_nr + 1



def addNextCable(cable_id, circuit_cables, connections_df, customers_df, bus_list, branch_list, devices_dict,active_cons_dict, last_cablebus_id_prev_cable=0, length_prev_cable=0):

    #circuit_cable = next((cable for cable in circuit_cables if cable["id"] == cable_id), False)
    circuit_cable = circuit_cables[circuit_cables['id']==cable_id]

    if len(circuit_cable)>0:

        bus_list, branch_list, devices_dict, last_cablebus_id, length_prev_cable = addConnections(circuit_cable, connections_df, customers_df, bus_list, branch_list, devices_dict, active_cons_dict,  last_cablebus_id_prev_cable, length_prev_cable)
        print(cable_id)
        next_cables = circuit_cable['next'].values[0]
        ### fix for eandis
        if isinstance(next_cables, int):
            if circuit_cable['id'] == next_cables:
                next_cables = []
        #and now find the following cables in the right order
        if isinstance(next_cables, list):
            if len(next_cables) > 0:
                for next_cable_id in next_cables:
                    bus_list, branch_list, devices_dict = addNextCable(next_cable_id, circuit_cables, connections_df, customers_df, bus_list, branch_list, devices_dict, active_cons_dict, last_cablebus_id, length_prev_cable)

            else:
                return bus_list, branch_list, devices_dict
        if isinstance(next_cables, int):
            next_cable_id = next_cables
            bus_list, branch_list, devices_dict = addNextCable(next_cable_id, circuit_cables, connections_df, customers_df, bus_list, branch_list, devices_dict,active_cons_dict, last_cablebus_id,length_prev_cable)

#                             circuit_cable = next((cable for cable in circuit_cables if cable["id"] == next_cable_id), False)
#                             if circuit_cable:
#                                 circuit_cable_id = circuit_cable['id']
#                                 bus_list, branch_list, devices_dict, last_cablebus_id = addConnections(circuit_cable, connections, customers, bus_list, branch_list, devices_dict, last_cablebus_id)
#                                 print(circuit_cable_id)
#                                 next_cables = circuit_cable['next']
#                             else:
#                                 print('something wrong with cable data: cable with right cable_id missing')
#                                 #print(next_cables)
    else:
        print('error in cable data: no cable with requested id: ', cable_id)

    return bus_list, branch_list, devices_dict


def addConnections(cable_dict, connections_df, customers_df, bus_list, branch_list, devices_dict, active_cons_dict, last_cablebus_id_prev_cable, length_prev_cable):
    cable_id = cable_dict['id'].values[0]
    cable_type = cable_dict['cableType'].values[0]
    cable_length = cable_dict['length'].values[0]
    ### dirty fix for wrong key names
    if 'cableId' in connections_df.columns:
        cable_connections = connections_df[connections_df['cableId']==cable_id]#[conn for conn in connections if conn['cableId'] == cable_id]
    elif 'cablelink' in connections_df.columns:
        cable_connections = connections_df[connections_df['cablelink']==cable_id]# [conn for conn in connections if conn['cablelink'] == cable_id]
    elif 'cableLink' in connections_df.columns:
        cable_connections = connections_df[connections_df['cableLink']==cable_id]# [conn for conn in connections if conn['cableLink'] == cable_id]

    #print('nrOfConnections:', len(cable_connections))

    if len(cable_connections) > 0:
#        cable_distances = [5] #Dirty fix as connection cable length is not given
        cable_distances = 20
        no_connections=len(cable_connections)#sorted([conn['cableDistance'] for conn in cable_connections]) #sort the distances
    else: cable_distances = []

    last_cablebus_id = last_cablebus_id_prev_cable  ## the up-bus id of the next cable-part
    #print('cable distances : ', sorted(cable_distances))
#    #print('cable length :', cable_length)
#    if (len(cable_distances) ==0) or (min(cable_distances) >= length_prev_cable):
#        previous_dist = length_prev_cable
#    else:
#        previous_dist = 0

        ### bus + branch part of cable
    last_bus_id = bus_list[-1]['busId']
    conn_bus_id = last_bus_id + 1  #the bus id of the node at the 'circuit-cable'

    bus_dict = dict()
    bus_dict['busId'] = conn_bus_id
    bus_dict['slackBus'] = False
    bus_dict['upperVoltageLimit'] = 1.1
    bus_dict['lowerVoltageLimit'] = 0.9
    bus_list.append(bus_dict)

    if len(branch_list)>0:
        circuitCable_branch_id = branch_list[-1]['branchId'] + 1
    else:
        circuitCable_branch_id = 0


    branch_dict = dict()
    branch_dict['branchId'] = circuitCable_branch_id
    branch_dict['upBusId'] = last_cablebus_id
    branch_dict['downBusId'] = conn_bus_id
    branch_dict['type'] = 'Cable'
    branch_dict['cableType'] = cable_type
    branch_dict['cableLength'] = cable_length
    if branch_dict['cableLength'] == 0:
        print('0 length')
        #print(branch_dict['cableLength'])

    branch_list.append(branch_dict)

#        previous_dist = dist

    last_cablebus_id = conn_bus_id

        ### bus + branch part of connectioncable (one per connection)

        #connections = [conn for conn in cable_connections if conn["cableDistance"] == dist] #there can be more than one connecton with the same distance
    connections = cable_connections #[conn for conn in cable_connections if conn["cableDistance"] == dist]

    for conn_index, conn_row in connections.iterrows():  #for connection in connections:
        cable_offset = 20

        customer_conn_bus_id = bus_list[-1]['busId'] + 1 #the bus id of the node after the 'connectioncable' (i.e. where the devices/customers are connected to)
        bus_dict = dict()
        bus_dict['busId'] = customer_conn_bus_id
        bus_dict['slackBus'] = False
        bus_dict['upperVoltageLimit'] = 1.1
        bus_dict['lowerVoltageLimit'] = 0.9
        bus_list.append(bus_dict)

        connectionCable_branch_id = branch_list[-1]['branchId'] + 1
        branch_dict = dict()
        branch_dict['branchId'] = connectionCable_branch_id
        branch_dict['upBusId'] = conn_bus_id
        branch_dict['downBusId'] = customer_conn_bus_id
        branch_dict['type'] = 'Cable'
        branch_dict['cableType'] = 'aansluitkabel'
        branch_dict['cableLength'] = np.random.choice(20)
        branch_list.append(branch_dict)

        devices_dict = addCustomers(conn_row, customer_conn_bus_id, customers_df, devices_dict,active_cons_dict)

    ## the last part of the cable (after the last connection)
    last_bus_id = bus_list[-1]['busId']
    conn_bus_id = last_bus_id + 1  #the bus id of the node at the 'circuit-cable'
#    if len(cable_distances) < 1:
#        branch_length = cable_length
#    else:
#        branch_length = max([0,cable_length - max(cable_distances)])

#    if cable_length > 0: # if length == 0: do not add a branch
#
#        bus_dict = dict()
#        bus_dict['busId'] = conn_bus_id
#        bus_dict['slackBus'] = False
#        bus_dict['upperVoltageLimit'] = 1.1
#        bus_dict['lowerVoltageLimit'] = 0.9
#        bus_list.append(bus_dict)
#
#        if len(branch_list)>0:
#            circuitCable_branch_id = branch_list[-1]['branchId'] + 1
#        else:
#            circuitCable_branch_id = 0
#        branch_dict = dict()
#        branch_dict['branchId'] = circuitCable_branch_id
#        branch_dict['upBusId'] = last_cablebus_id
#        branch_dict['downBusId'] = conn_bus_id
#        branch_dict['type'] = 'Cable'
#        branch_dict['cableType'] = cable_type
#        branch_dict['cableLength'] = cable_length
#        branch_list.append(branch_dict)
#
#        last_cablebus_id = conn_bus_id

    return bus_list, branch_list, devices_dict, last_cablebus_id, cable_length

def addCustomers(connection, conn_bus_id, customers_df, devices_dict,active_cons_dict):
    connection_id = connection['id']
    if 'connectionId' in customers_df.columns:
        conn_customers = customers_df[customers_df['connectionId']==connection_id] # [cust for cust in customers if cust['connectionId'] == connection_id]
    elif 'connection_id' in customers_df.columns:
        conn_customers = customers_df[customers_df['connection_id']==connection_id] #[cust for cust in customers if cust['connection_id'] == connection_id]

    if 'numberOfEANs' in customers_df.columns: #if connection['numberOfEANs']:
        if connection['numberOfEANs'] > len(conn_customers):
            print('customers not found with connection id: ', connection_id)

    LVCustomers_list = devices_dict['LVcustomers']
    solarGens_list = devices_dict['solarGens']

    for conn_index, conn_row in conn_customers.iterrows(): #for conn in conn_customers:

        if (len(LVCustomers_list)>0) & (len(solarGens_list)>0):
            device_id = max([LVCustomers_list[-1]['deviceId'], solarGens_list[-1]['deviceId']])  + 1
        elif len(LVCustomers_list)>0:
            device_id = LVCustomers_list[-1]['deviceId'] + 1
        elif len(solarGens_list)>0:
            device_id = solarGens_list[-1]['deviceId'] + 1
        else:
            device_id = 0

        device_dict = dict()
        device_dict['deviceId'] = device_id
        device_dict['busId'] = conn_bus_id
#        device_dict['statSector'] = connection['statSector']

        if conn_row['id'] in active_cons_dict:
            device_dict['yearlyNetConsumption'] = active_cons_dict[conn_row['id']] #added
            print("yearlyNet Consumption for " + conn_row['id'] + " is " + str(device_dict['yearlyNetConsumption']))
        else:
            print("No yearlyNet Consumption found for " + conn_row['id'])



        if 'periodP1' in conn_customers.columns:
            device_dict['connectionCapacity'] = conn_row['periodP1']
            if not np.isnan(conn_row['periodP2']):
                device_dict['connectionCapacity'] =max(conn_row['periodP1'],conn_row['periodP2'],conn_row['periodP3'])
#                else:
#                    device_dict['connectionCapacity'] = 9.2
#            elif conn_row['connectionCapacity'] == 0 or not conn_row['connectionCapacity']:
#                #print('no connectionCapacity property in json file for connection id: ' + str(conn['connectionId']) + ', default 9.2/22.2 kVA assumed')
#                device_dict['connectionCapacity'] = 0

        else:
            print('no connectionCapacity property in json file for connection id: ' + str(conn_row['connectionId']) + ', default 9.2/22.2 kVA assumed')
            if device_dict['yearlyNetConsumption'] > 10000:
                device_dict['connectionCapacity'] = 22.2
            else:
                device_dict['connectionCapacity'] = 9.2


        if 'connectionPhase' in conn_customers.columns:
            if conn_row['connectionPhase'] == 'U':
                device_dict['phases'] = [1,2,3]
            elif conn_row['connectionPhase'] == 'M':
                device_dict['phases'] = [np.random.choice(3)+1]
            else:
                if  device_dict['connectionCapacity'] in [11.1, 13.9, 17.3, 22.2, 27.7, 31.2, 34.6, 43.6, 55.4, 6.4, 8, 10.0, 12.7, 15.9, 19.9, 25.1, 31.9]: ### if phase connection not given base nr of phases on connectioncapacity
                    device_dict['phases'] = [1,2,3]
                elif  device_dict['connectionCapacity'] > 15:
                    device_dict['phases'] = [1,2,3]
                else:
                    #check phase: try to connect 1,2,3,1,2,3,1,2,3
                    phase = 0
                    device_id_prev = device_id - 1
                    while phase < 1:
                        if device_id_prev < 0:
                            phase = 1
                        else:
                            prev_dev = next((dev for dev in LVCustomers_list if dev["deviceId"] == device_id_prev), False)
                            if prev_dev:
                                prev_phase = prev_dev['phases']
                                if len(prev_phase) > 1:
                                    device_id_prev = device_id_prev-1
                                else:
                                    phase = prev_phase[0]%3 + 1  # % is matlab mod function
                            else:
                                device_id_prev = device_id_prev-1

                    device_dict['phases'] = [phase]
        else:
            print('no connectionPhase property in json file for connection id: ' + str(conn_row['connectionId']))
            if  device_dict['connectionCapacity'] in [11.1, 13.9, 17.3, 22.2, 27.7, 31.2, 34.6, 43.6, 55.4, 6.4, 8, 10.0, 12.7, 15.9, 19.9, 25.1, 31.9]: ### if phase connection not given base nr of phases on connectioncapacity
                device_dict['phases'] = [1,2,3]
            elif  device_dict['connectionCapacity'] > 15:
                device_dict['phases'] = [1,2,3]
            else:
                phase = 0
                device_id_prev = device_id - 1
                while phase < 1:
                    if device_id_prev < 0:
                        phase = 1
                    else:
                        prev_dev = next((dev for dev in LVCustomers_list if dev["deviceId"] == device_id_prev), False)
                        if prev_dev:
                            prev_phase = prev_dev['phases']
                            if len(prev_phase) > 1:
                                device_id_prev = device_id_prev-1
                            else:
                                phase = prev_phase[0]%3 + 1  # % is matlab mod function
                        else:
                            device_id_prev = device_id_prev-1

                device_dict['phases'] = [phase]


        device_dict['exclusiveNight'] = False
        if 'exNight' in conn_customers.columns:
            device_dict['exclusiveNight'] = bool(conn_row['exNight'])
        elif 'ExNight' in conn_customers.columns:
            device_dict['exclusiveNight'] = bool(conn_row['ExNight'])

        if 'ean' in conn_customers.columns:
            device_dict['ean'] = conn_row['ean']
        #add coded ean to device_dict anyway
        device_dict['coded_ean'] = conn_row['id']
        #LVCustomers_list.append(device_dict)

#        if conn_row['PV'] == 1:
##             device_id = device_id + 1
##             device_dict = dict()
##             device_dict['deviceId'] = device_id
##             device_dict['busId'] = conn_bus_id
#            if device_dict['phases'] != [1,2,3]: #single phase devices: PV is not modeled as separate device
#                device_dict['PV'] = True
#                device_dict['PVCapacity'] = conn_row['PVCapacity']
#                device_dict['PVInverterPower'] = conn_row['PVInverterPower']
#            elif conn_row['PVInverterPower'] > 10: #three phase device with PV installation > 10kW, modeled as single device
#                device_dict['PV'] = True
#                device_dict['PVCapacity'] = conn_row['PVCapacity']
#                device_dict['PVInverterPower'] = conn_row['PVInverterPower']
#            elif conn_row['PVInverterPower'] > 5: #PV installation > 5 kW, < 10 kW: modeled as 2 PV installation (divided over 2 different phases)
#                device_dict['PV'] = False
#                device_dict['PVCapacity'] = 0
#                device_dict['PVInverterPower'] = 0
#                PVgen1_dict = dict()
#                PVgen2_dict = dict()
#                PVgen1_dict['deviceId'] = device_id + 1
#                PVgen1_dict['busId'] = conn_bus_id
#                all_phases = [1,2,3]
#                phase_gen1 = random.choice(all_phases)
#                PVgen1_dict['phases'] = [phase_gen1]
#                PVgen1_dict['PVCapacity'] = conn_row['PVCapacity']/2
#                PVgen1_dict['PVInverterPower'] = conn_row['PVInverterPower']/2
#                PVgen2_dict['deviceId'] = device_id + 2
#                PVgen2_dict['busId'] = conn_bus_id
#                all_phases.remove(phase_gen1)
#                PVgen2_dict['phases'] = [random.choice(all_phases)]
#                PVgen2_dict['PVCapacity'] = conn_row['PVCapacity']/2
#                PVgen2_dict['PVInverterPower'] = conn_row['PVInverterPower']/2
#                solarGens_list.append(PVgen1_dict)
#                solarGens_list.append(PVgen2_dict)
#            else: #PV installation <= 5 kW :allocated to 1 phase
#                device_dict['PV'] = False
#                device_dict['PVCapacity'] = 0
#                device_dict['PVInverterPower'] = 0
#                PVgen_dict = dict()
#                PVgen_dict['deviceId'] = device_id + 1
#                PVgen_dict['busId'] = conn_bus_id
#                PVgen_dict['phases'] = [random.choice([1,2,3])]
#                PVgen_dict['PVCapacity'] = conn_row['PVCapacity']
#                PVgen_dict['PVInverterPower'] = conn_row['PVInverterPower']
#                solarGens_list.append(PVgen_dict)
#




#             if 'connectionPhase' in conn.keys():
#                 if conn['connectionPhase'] == '4dr + 400/230V' or conn['connectionPhase'] == 'RSTN':
#                     if conn['PVInverterPower'] > 5:
#                         device_dict['phases'] = [1,2,3]
#                     else:
#                         device_dict['phases'] = [random.choice([1,2,3])]
#
#                 else:
#                     if  conn['connectionCapacity'] in [11.1, 13.9, 17.3, 22.2, 27.7, 31.2, 34.6, 43.6, 55.4, 6.4, 8, 10.0, 12.7, 15.9, 19.9, 25.1, 31.9]: ### if phase connection not given base nr of phases on connectioncapacity
#                         if conn['PVInverterPower'] > 5:
#                             device_dict['phases'] = [1,2,3]
#                         else:
#                             device_dict['phases'] = [random.choice([1,2,3])]
#
#                     elif  conn['connectionCapacity'] > 15:
#                         if conn['PVInverterPower'] > 5:
#                             device_dict['phases'] = [1,2,3]
#                         else:
#                             device_dict['phases'] = [random.choice([1,2,3])]
#                     else:
#                         device_dict['phases'] = [phase]
#             else:
#                 #see above: no connection phase info available, single phase assumed
#
#                 device_dict['phases'] = LVCustomers_list[-1]['phases']


            #

        LVCustomers_list.append(device_dict)


    return devices_dict


def printGridDataToFiles(bus_list, branch_list, devices_dict, circuit, trafo, cabine, type='feeder_level'):
    if type == 'feeder_level':
        #circuit_name = str(cabine['id']) + '_' + str(circuit['id'])#str(cabine['id']) + '_' + str(trafo['id'])#str(cabine['id']) + '_' + str(circuit['id'])#circuit['name']

        ## Fix: use Trafo Id in filename  ==> makes it possible to query for trafo's in filenames to do trafo current calculations
        circuit_name = str(trafo['id']) + '_' + str(circuit['id'])
    elif type == 'trafo_level':
        circuit_name = str(cabine['id']) + '_' + str(trafo['id'])
    circuit_name = circuit_name.replace('/','_')
    circuit_name = circuit_name.replace('>','to')
    circuit_name = circuit_name.replace('?',' ')
    print('filename = '+ circuit_name)

    cabine_city = str(cabine["city"])

    #make buses, branches, devices files
    bus_filename = griddata_dir + cabine_city + '/' + circuit_name + '_buses.json'
    branches_filename = griddata_dir + cabine_city + '/' + circuit_name + '_branches.json'
    devices_filename = griddata_dir + cabine_city + '/' + circuit_name + '_devices.json'
    bus_filename_in_config = cabine_city + '/' + circuit_name + '_buses.json'
    branches_filename_in_config = cabine_city + '/' + circuit_name + '_branches.json'
    devices_filename_in_config = cabine_city + '/' + circuit_name + '_devices.json'


    create_json_file(bus_list, bus_filename)
    create_json_file(branch_list, branches_filename)
    create_json_file(devices_dict, devices_filename)

    #make config file
    config_filename = griddata_dir + cabine_city + '/' + circuit_name + '_configuration.json'
    config_dict = dict()
    config_dict['name'] = circuit['name']
    config_dict['cabineName'] = cabine['name']
    config_dict['trafoId'] = int(trafo['id'])
    config_dict['city'] = cabine['city']
    config_dict['totalNrOfEANS'] = len(devices_dict['LVcustomers'])
    config_dict['id'] = int(circuit['id'])

    config_dict['baseMVA'] = trafo['nominalCapacity']/1000
    config_dict['basekV'] = 400/1000

#    if circuit['gridType'] == '230/400 V':
#        config_dict['connection_configuration'] = 'star'
#    elif circuit['gridType'] == '3,230':
#        config_dict['connection_configuration'] = 'delta'
#    elif circuit['gridType'] == '3N400':
#        config_dict['connection_configuration'] = 'star'
#    else:
#        config_dict['connection_configuration'] = 'star'
    config_dict['connection_configuration'] = 'star'
    config_dict['slack_voltage'] = 1
    config_dict['branches_file'] = branches_filename_in_config
    config_dict['buses_file'] = bus_filename_in_config
    config_dict['devices_file'] = devices_filename_in_config

    config_config_dict = {'gridConfig' : config_dict}
    create_json_file(config_config_dict, config_filename)



def create_json_file(some_list_or_dict, file_name):
    json_string = json.dumps(some_list_or_dict, sort_keys=True, indent=4, separators=(',',':'))

    with safe_open_w(file_name) as jsonFile:
        jsonFile.write("%s" % json_string)
    #print('json file created :' +  file_name)
    return json_string


## see https://stackoverflow.com/questions/23793987/python-write-file-to-directory-doesnt-exist
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')



def getAllCableTypes(gridFormatDict, filename):
    if 'Cable' in gridFormatDict.keys():
        cables = gridFormatDict["Cable"]
    elif 'LS Cables' in gridFormatDict.keys():
        cables = gridFormatDict['LS Cables']

    cableTypes = []
    for ind,cable in islice(cables_df.iterrows(), 0, None):
        cable_type = cable['cableType']
        if cable_type not in cableTypes:
            cableTypes.append(cable_type)
    cableTypes.sort()
    create_json_file(cableTypes, cable.json)

    return cableTypes

def makeFeederIdNameList(gridFormatDict):

    if 'Circuit' in gridFormatDict.keys():
        circuits = gridFormatDict["Circuit"]
    elif 'Circuits' in gridFormatDict.keys():
        circuits = gridFormatDict["Circuits"]

    name_versus_id_dataframe = pd.DataFrame(columns=['feeder_name', 'feeder_id', 'numberOfEANS'])
    for circ in circuits:
        new_row = {'feeder_name' : circ['name'], 'feeder_id' : circ['id'], 'numberOfEANS': circ['numberOfEANs']}
        name_versus_id_dataframe = name_versus_id_dataframe.append(new_row,  ignore_index=True)

    name_versus_id_dataframe.to_csv('feeder_name_vs_id.csv')


def makeTransformerFiles(gridFormatDict):
    if 'Cabine' in gridFormatDict.keys():
        cabines = gridFormatDict["Cabine"]
    elif 'MS Cabines' in gridFormatDict.keys():
        cabines = gridFormatDict["MS Cabines"]

    if 'Transformer' in gridFormatDict.keys():
        transformers = gridFormatDict["Transformer"]
    elif 'Transformers' in gridFormatDict.keys():
        transformers = gridFormatDict["Transformers"]

    if 'Circuit' in gridFormatDict.keys():
        circuits = gridFormatDict["Circuit"]
    elif 'Circuits' in gridFormatDict.keys():
        circuits = gridFormatDict["Circuits"]

    c_nr = 1

    trafo_dict_list = [] #list per city
    city_list = []
    for c in cabines:
        print('cabine nr: ',c_nr)
        c_nr = c_nr+1
        c_id = c["id"]
        cabine_city = str(c["city"])
        if not cabine_city in city_list:
            city_list.append(cabine_city)
        cabine_name = str(c["name"])
        cab_trafos = [trafo for trafo in transformers if trafo['cabineId'] == c_id]
        for trafo in cab_trafos:
            tr_id = trafo["id"]
            trafo_circuits = [circ for circ in circuits if circ['transfoId'] == tr_id]
            trafo_dict = {}
            trafo_dict['city'] = cabine_city
            trafo_dict['id'] = tr_id
            trafo_dict['cabineName'] = cabine_name
            circuit_list = []
            circuit_id_list = []
            for circuit in trafo_circuits:
                circuit_name = str(c['id']) + '_' + str(circuit['id'])
                circuit_config_file = cabine_city + '/' + circuit_name + '_configuration.json'
                circuit_list.append(circuit_config_file)
                circuit_id_list.append(circuit['id'])
            trafo_dict['feeder_config_files'] = circuit_list
            trafo_dict['feeder_ids'] = circuit_id_list
            trafo_dict['nominalCapacity'] = trafo["nominalCapacity"]
            trafo_dict['constructionYear'] = trafo["constructionYear"]
            trafo_dict_list.append(trafo_dict)

    trafo_per_city_list  = [[] for _ in range(len(city_list))]
    for t_dict in trafo_dict_list:
        t_city = t_dict["city"]
        city_index = city_list.index(t_city)
        trafo_per_city_list[city_index].append(t_dict)

    for i, city_trafos in enumerate(trafo_per_city_list):
        create_json_file(city_trafos, griddata_dir +  city_list[i] + '/' + 'trafo_list.json')



def orig_dest(row):
    if row['result'] == 1:
#        print(cables_df.id[(cables_df.Origin == row['Dest'])])
        return []

def end_dest(row):
    if row['next'] == 0:
        return []
    else:
        return row['next']

#%%
if __name__ == '__main__':

#%%

    #dir_n="U:/PhD/probablistic_pf/test_network/Sim_files_190128_OK_V0/GIS_data/master.xlsx"
    dir_n= griddata_dir + "test_network/Sim_files_190128_OK_V0/GIS_data/master.xlsx"
    trafo=pd.read_excel(dir_n, sheet_name='CT - TRAFO')
    linea=pd.read_excel(dir_n, sheet_name='Linea BT')
    segment=pd.read_excel(dir_n, sheet_name='Segmento BT')
    load_point=pd.read_excel(dir_n, sheet_name='Acometidas')
    fuse=pd.read_excel(dir_n, sheet_name='Fusible')

    #load_n="U:/PhD/probablistic_pf/test_network/Sim_files_190128_OK_V0/GIS_data/load_det.xlsx"
    load_n= griddata_dir + "test_network/Sim_files_190128_OK_V0/GIS_data/load_det.xlsx"
    load_details=pd.read_excel(load_n, sheet_name='Load')
#%% load profile and phase
#TO Do for Alexander commented for now as the code below is quite time consuming and not complete
    loadProfile=pd.DataFrame()
    #for m in range(1,8):
    #    #load_file = "U:/PhD/probablistic_pf/test_network/Sim_files_190128_OK_V0/GIS_data/file"+ str(m)+".xlsx"
    #    load_file = griddata_dir + "/Sim_files_190128_OK_V0/GIS_data/file"+ str(m)+".xlsx"
    #    loadProfile=loadProfile.append(pd.read_excel(load_file))
    for m in range(1,8):
        csv = griddata_dir + "test_network/Sim_files_190128_OK_V0/GIS_data/file"+ str(m)+".csv"
        print(csv)
        #For automatically converting excell to csv, for now I convert them manually and load the csv in a dataframe below
        #excel_dir = griddata_dir + "test_network/Sim_files_190128_OK_V0/GIS_data/"
        #excel_name = "file" + str(m)+".xlsx"
        #subprocess.call(['cscript.exe', griddata_dir +"parsers/ExcelToCsv.vbs", excel_dir + excel_name , csv, m])
        loadProfile= loadProfile.append(pd.read_csv(csv,sep=";",skipinitialspace=True, decimal=',', thousands=' '))
        #loadProfile["Activa E"].apply(locale.atof)
        #loadProfile = loadProfile.replace(',','.')
        #loadProfile["Activa E"] = loadProfile["Activa E"].astype(float)
        loadProfile["Activa E"].mask(loadProfile["Activa E"]>1000,np.nan,inplace=True)  #remove outliers
        loadProfile["Activa S"].mask(loadProfile["Activa S"] > 1000, np.nan, inplace=True)

    active_cons_dict = loadProfile.groupby("Referencia")["Activa E"].mean()
    active_cons_dict += loadProfile.groupby("Referencia")["Activa S"].mean()    #include night tariff in yearly consumption
    active_cons_dict = active_cons_dict* 24*20*15 # 24 samples per day, 20 days of data available, *15 to estimate yearly (300 days) consumption

#%%
    cabines_df=trafo[['CLAVE_BDI','DES']]
    cabines_df=cabines_df.rename(columns={'CLAVE_BDI':'id','DES':'name'})
    cabines_df['city']='POLA'
#%%
    df_count=load_details['Codigo Ct'].value_counts()
    df_count=df_count.to_frame()
    df_count=df_count.reset_index()
    df_count=df_count.rename(columns={"index":"CLAVE_BDI","Codigo Ct":"numberOfEANs"})
    trafo=pd.merge(trafo, df_count, on=['CLAVE_BDI'])
    df=trafo.drop(columns={'MSLINK CELDA','X','Y','CLAVE CELDA','DES','I','CLAVE TRAFO','NUDO CELDA TRAFO'})
    df=df.rename(columns={"MSLINK":"id","CLAVE_BDI":"cabineId",'POTENCIA TRAFO':'nominalCapacity'})
    transformers_df=pd.DataFrame(columns=['id', 'cabineId', 'constructionYear', 'numberOfEANs', 'nominalCapacity',
         'ratedSecondaryVoltage1', 'ratedSecondaryVoltage2'])
    transformers_df=transformers_df.append(df)

#%%
    circuits_df=linea.drop(columns={'X','Y'})
    numberOfEANs=load_details.groupby(['Codigo Ct','Linea']).size()
    numberOfEANs=numberOfEANs.to_frame()
    numberOfEANs=numberOfEANs.reset_index()
    circuits_df=circuits_df.rename(columns={'Mslink':'id','Clave Línea':'name', 'Mslink CT':'transfoId', 'NO':'originBus', 'NE':'destionBus'})
    circuits_df=circuits_df[circuits_df.EST_OPERAC=='C']
    circuits_df=pd.merge(circuits_df,numberOfEANs,right_on=['Codigo Ct', 'Linea'], left_on=['Clave CT','name'], how='left').fillna(0)
    circuits_df=circuits_df.drop(columns={'Linea','Codigo Ct','Clave CT'})
    circuits_df=circuits_df.rename(columns={0:'numberOfEANs'})
#%%
    fusible=fuse[fuse['Est. Operac']=='C']
    fusible=fusible.rename(columns={"Nudo Origen":"Origin","Nudo Extremo":"Dest"})
    fusible=fusible[['Origin','Dest']]
    fus_list=fusible.values.tolist()


#%%

    cables_df=segment.rename(columns={'Mslink':'id','Mslink Línea':'circuitId','Longitud':'length','Nudo Origen':'Origin','Nudo Destino':'Dest','Tipo Cable':'cableType'})
    cables_df = cables_df.assign(result=cables_df['Dest'].
                                isin(circuits_df['destionBus']))
    cond = cables_df.result == True
#df.loc[cond, ['a', 'b']] = df.loc[cond, ['b', 'a']].values
    cables_df.loc[cond, ['Origin','Dest']] = cables_df.loc[cond, ['Dest','Origin']].values
    #    cables_df[cables_df['prev']] = cables_df.apply (lambda row: orig_dest(row), axis=1)
    cables_df['prev'] = 0
    cables_df = cables_df.assign(result=cables_df['Origin'].
                                isin(circuits_df['destionBus']).astype(int))
    cables_df['prev'] = cables_df.apply (lambda row: orig_dest(row), axis=1)
    cables_df['next'] = 0
    cables_df['next']=cables_df['next'].astype(object)
    j=0
    for d,o in fus_list:
        l=cables_df[cables_df['Origin'].isin([d])].id
        if not l.empty:
            cables_df['Origin'][l.index]=o
#           cables_df['next'][l.index]=1
            j += 1
        l=cables_df[cables_df['Dest'].isin([d])].id
        if not l.empty:
            cables_df['Dest'][l.index]=o
#          cables_df['next'][l.index]=2
            j += 1

    for i in range(30):
        df_origin=cables_df[cables_df.result == 1]
        df_remaining=cables_df[cables_df.result == 0]
        cables_df.result[cables_df.result > 0] +=1
        b=set(df_origin['Dest'])
        if not bool(b):
            break
        for x in b:
            p=df_remaining[cables_df['Dest'].isin([x])].id
            if not p.empty:
                temp=cables_df.Origin[p.index]
                cables_df.Origin[p.index]=cables_df.Dest[p.index]
                cables_df.Dest[p.index]=temp
#                y=df_origin[cables_df['Dest'].isin([x])].id
#                for k in p.index:
#                    cables_df.prev[k]=list(y.values)
#                for z in y.index:
#                    cables_df.next[z]=list(p.values)
#
#                cables_df.result[p.index] += 1
            s=df_remaining[cables_df['Origin'].isin([x])].id
            if not s.empty:
                y=df_origin[cables_df['Dest'].isin([x])].id
                for g in s.index:
                    cables_df.prev[g]=list(y.values)
                    cables_df.result[g] += 1
                for m in y.index:
                    cables_df.next[m]=list(s.values)
#            p=df_remaining[cables_df['Dest'].isin([x])].id
#            if not p.empty:
#                y=df_origin[cables_df['Dest'].isin([x])].id
#                for k in p.index:
#                    cables_df.prev[k]=list(y.values)
#                for z in y.index:
#                    cables_df.next[z]=list(p.values)
#                temp=cables_df.Origin[p.index]
#                cables_df.Origin[p.index]=cables_df.Dest[p.index]
#                cables_df.Dest[p.index]=temp
#                cables_df.result[p.index] += 1
    cables_df=cables_df[cables_df.result != 0]
    cables_df=cables_df.drop(columns={'result'})
    cables_df['next'] = cables_df.apply (lambda row: end_dest(row), axis=1)

#%%circuits_df
    connections_df=load_point.rename(columns={"Clave BDI":"id","Nudo Origen":"Origin", "Nudo Destino":"Dest"})
    connections_df=connections_df.drop(columns={"X","Y"})
    connections_df['cablelink']=np.NaN
    for i in range(len(connections_df)):
        if not cables_df[cables_df.Dest==connections_df["Origin"][i]].id.empty:
            #print(i)
            connections_df['cablelink'][i]=cables_df[cables_df.Dest==connections_df["Origin"][i]].id
    connections_df.cablelink=connections_df.cablelink.astype('Int64')
#%% connections_df
    customers_df=load_details[['Referencia','Tipo Contador',
                               'Pot Contratada P1','Pot Contratada P2',
                               'Pot Contratada P3','Fases', 'Acometida',
                               'Clave Ct','Conc_Referencia']]
    customers_df=customers_df.rename(columns={'Referencia':'id','Tipo Contador':'contractType',
                               'Pot Contratada P1':'periodP1','Pot Contratada P2':'periodP2',
                               'Pot Contratada P3':'periodP3','Fases':'connectionPhase', 'Acometida':'connectionId',
                               'Clave Ct':'CTname','Conc_Referencia':'feederMeter'})

 #%% parsing algorithm from database TODO for summer job cleaning the code
    c_nr = 1
    circuit_nr = 1
    for cab_index, cab_row in islice(cabines_df.iterrows(), 0, None): #for c in cabines[1027:]:#[0:10]:
        #print('###### cabine nr: ',c_nr)
        c_id = cab_row["id"]#c["id"]
        #cabine_city = c["city"]
        #cab_trafos = [trafo for trafo in transformers if trafo['cabineId'] == c_id]
        cab_trafos = transformers_df[transformers_df["cabineId"]==c_id]
        for trafo_index, trafo_row in cab_trafos.iterrows(): #for trafo in cab_trafos:
            tr_id = trafo_row["id"]#trafo['id']
            #trafo_circuits = [circ for circ in circuits if circ['transfoId'] == tr_id]
            trafo_circuits = circuits_df[circuits_df["transfoId"]==tr_id]
            for circ_index, circ_row in trafo_circuits.iterrows():#for circuit in trafo_circuits:
                circuit_name = circ_row["name"]#circuit['name']
                #print('circuit name: ' + circuit_name)
                #### lists where grid information should be parsed to
                bus_list = []
                branch_list = []
                devices_dict = dict()
                devices_dict['LVcustomers'] = []
                devices_dict['solarGens'] = []


                ## add slack bus to bus list
                bus_dict = dict()
                bus_dict['busId'] = 0
                bus_dict['slackBus'] = True
                bus_list.append(bus_dict)

                ### find cables in circuit (in correct order)
                circuit_id = circ_row['id']#circuit['id']
                if 'circuitVoltage' in trafo_circuits.columns: #circuit.keys():
                    circuit_voltage = circ_row['circuitVoltage']#circuit['circuitVoltage']
                else:
                    circuit_voltage = 400
                    print('unknown circuit voltage, value not given, 400V assumed')

                #circuit_cables = [cable for cable in cables if cable['circuitId'] == circuit_id]
                circuit_cables = cables_df[cables_df['circuitId']==circuit_id]#[cable for cable in cables if cable['circuitId'] == circuit_id]
                #first cable has empty 'prev' list
                #first_cable = next((cable for cable in circuit_cables if cable["prev"] == []), False)
                first_cable = circuit_cables[circuit_cables["prev"].apply(np.size)==0]
#                 ### to debug eandis case ###
#                 if not first_cable:
#                     #first_cable = next((cable for cable in circuit_cables if cable["prev"] == cable['id']), False)
#                     first_cable = circuit_cables[circuit_cables["prev"]==circuit_cables['id']] ##don't know if this works

                if len(first_cable)>0:
                    bus_list, branch_list, devices_dict = addNextCable(first_cable['id'].values[0], circuit_cables, connections_df, customers_df, bus_list, branch_list, devices_dict,active_cons_dict)
                else:
                    print('no cable present with prev == [] for circuitId: ', circuit_id)
                #print(bus_list)
                #print(branch_list)
                #print(devices_dict)
                #print('circuit done')
                #print('circuit nr:', circuit_nr)
                circuit_nr = circuit_nr + 1

                printGridDataToFiles(bus_list, branch_list, devices_dict, circ_row, trafo_row, cab_row, type='feeder_level')
        c_nr = c_nr + 1
