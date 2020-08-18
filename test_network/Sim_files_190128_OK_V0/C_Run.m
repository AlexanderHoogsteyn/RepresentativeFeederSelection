clear all;
clc;

MatDir  = [pwd filesep 'mat' filesep];

%Variables used for the purpose of monitors only
no_days=20; %no of day of which load shape data is given
no_data_perday=24; %per hr data it will be 96 for per 15 min data

load([MatDir 'linecount.mat']);
load([MatDir 'txno.mat']);
n_monitored=linecount-2*txno;

n_wires=4; %number of columns in PQ monitor of OpenDSS
n_mon_PQ=2*n_wires+2;
n_mon_VI=4*n_wires+2;
%for 4 active power and 4 reactive power and 2 for time stamp FOR 4 wire system 
% 4 each of current, current angle, voltage and voltage angle + 2 for time 

total_loaddata=no_days*no_data_perday+1; %for hourly profile 1 extra point as loadshape start from 00:00hrs of first day and end in 00:00 hrs of last day
total_monitoredbus=linecount; %total no of line segment in the system

n_transformer=txno; %for this case 30 transformers
first_line=2*n_transformer;  % since first 30 integer are MV side of transformer and next 30 integer are LV side of tx

%Connecting the OpenDSS COM server
DSSObj = actxserver('OpenDSSEngine.DSS');
if (~DSSObj.Start(0))
disp('Unable to start the OpenDSS Engine');
return
end
DSSText = DSSObj.Text; % Used for all text interfacing from matlab to opendss
DSSCircuit = DSSObj.ActiveCircuit; % active circuit
DSSSolution=DSSCircuit.Solution;
% Write Path where Master and its associated files are stored and compile as per following command
oldfolder = cd([pwd filesep 'RunDss']);

DSSText.Command='Compile (Master.dss)';
DSSText.Command='batchedit load..* Vmin=0.8'; % Set Vmin lower so that load model property will remain same
DSSTransformers=DSSCircuit.Transformers;

monitor_VI=zeros(total_loaddata,n_mon_VI,total_monitoredbus);
monitor_PQ=zeros(total_loaddata,n_mon_PQ,total_monitoredbus);
%struct monitor
%cd(oldfolder);
notmonitored =[462;463;464;465;1085;2551;2552;2553;1505;2482];
%notmonitored = csvread('notmonitored.txt'); %this file is obtained from the network extraction script and manually updated if you dont want some line to be monitored
for i = 1 : n_monitored
	LineNum = i + first_line;
     k2 = find(any(notmonitored == LineNum ,2));
     if (isempty(k2))
			 try
         %script for rearranging the datastram obtained from OpenDSS
    moni = sprintf('line%d_vi_vs_time',LineNum);
    DSSCircuit.monitors.Name = moni; %Selects the monitor moni 'line%d_vi_vs_time'
    Freqs=DSSCircuit.monitors.ByteStream; %Request the Bytestream from OpenDSS
    iMonitorDataSize= typecast(Freqs(9:12),'int32'); % 4 data streams each of 8 byte are reserved to represent the no of monitors
    VIMonitor = typecast(Freqs(273:end),'single'); %The first 272 entries in data stream is header and remaining are monitors data
    monitor1= reshape(VIMonitor, iMonitorDataSize+2, [])';% reshaping the data in row and column from row matrix
    monitor_VI(:,:,i)=monitor1;
    
    ene = sprintf('line%d_pq_vs_time',LineNum);
    DSSCircuit.monitors.Name = ene; %Selects the monitor ene 'line%d_pq_vs_time'
    Freqs=DSSCircuit.monitors.ByteStream; %Request the Bytestream
    iMonitorDataSize= typecast(Freqs(9:12),'int32'); % To adjust the matrix
    PQMonitor = typecast(Freqs(273:end),'single'); %The first 272 entries in data stream is header
    power1= reshape(PQMonitor, iMonitorDataSize+2, [])';% reshaping the data in row and column from row matrix
    monitor_PQ(:,:,i)=power1;
			 catch
				 fprintf('no monitors data for line(%4d)\n', LineNum);
			 end
		 end
end


cd(oldfolder);

%struct monitor
figure(2)
plot(monitor_VI(:,9,500-first_line)) %Neutral current at bus 500
title('Neutral current at bus 500')
hold off

%%Loop for restructuring the voltage current and power
for k = 1 : total_monitoredbus
    bus(k).name = sprintf('bus %d', k + first_line);
    temp=1;
    for j = 1 : no_days
        for m = 1 : no_data_perday
            for l = 1 : n_wires
                bus(k).day(j).Voltage(m,l)= monitor_VI(temp,(2*l+1),k); %Voltage magnitude are at column 3,5,7 and 9 for 4 wire system
                bus(k).day(j).Current(m,l)=monitor_VI(temp,(9+2*l),k); %Current magnitude are at column 11, 13, 15 and 17 for 4 wire system
                bus(k).day(j).ActivePower(m,l)=monitor_PQ(temp,(2*l+1),k); %Active Power is in column 3,5,7 and 9 of monitor_PQ
                bus(k).day(j).ReactivePower(m,l)=monitor_PQ(temp,(2*l+2),k);% Reactive Power is in column 4, 6, 8 and 10 of monitor_PQ
						end
					temp = temp + 1;
        end
    end
end
save([pwd filesep 'mat' filesep 'bus.mat'], 'bus');

  figure(3)
   plot(monitor_VI(:,3,500-first_line)) %phase A (column 11) voltage in all bus 500
title('phase A (column 11) voltage in all bus 500')