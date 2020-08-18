%shortcut.m
%Set of File to Generate OpenDSS files
file_master_name	= 'master.xlsx';
file_load_name		= 'load.xlsx';

Isolated_Bus = [2722; 1505; 2867; 2482];

OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];
[status,message,messageid] = mkdir(OutDir);

MatDir  = [pwd filesep '..' filesep 'mat' filesep];
[status,message,messageid] = mkdir(MatDir);

copyfile(['SrcDss' filesep '*.*'], OutDir);

GIS_Dir = [pwd filesep '..' filesep 'GIS_data' filesep];
file_master = [GIS_Dir , file_master_name];%master file here
file_load = [GIS_Dir, file_load_name];%load details here
%file_loadshape=('Curva_Carga_Contadores_Pola4CTs.xlsx'); %loadshape excel file name
file_phase = [GIS_Dir, 'phase meters.xls'];


[transformer txID]=xlsread(file_master,'CT - TRAFO');
[Line lnraw]=xlsread(file_master,'Linea BT');	
[Segment Segraw]=xlsread(file_master,'Segmento BT');
buscord=xlsread(file_master,'Coordenadas Segmentos');
loadbus=xlsread(file_master,'Acometidas');
[fuse fuseraw]=xlsread(file_master,'Fusible');
[load_loc load_loc_raw]=xlsread(file_load, 'Load');
[phase phaseraw]=xlsread(file_phase);
txID=txID(:,8);



%[transformer, Line, lnraw, Segment, Segraw, buscord, loadbus,txID, fuse, fuseraw, load_loc, load_loc_raw, phaseraw]=filename;
linecode;%generates line code for existing type of feeder
bus_cord_modified; %Indexing of Bus number before rearrangement
Seg_crazy; %arranges BUS 1 and BUS 2 in proper order
bus_cord_modified; %Indexing of Bus number after rearrangement
line_indexed; %OpenDSS file generator for LV_line
bus_cord_indexed; %OpenDSS file Generator for Bus Coordinate
source_ind; %OpenDSS file Generator for MV line from source to transformer
fuse_indexed; %OpenDSS file Generator for Circuit Breaker in LV network
reactor_neutral; %OpenDSS file generator for Transformer 
transformer_indexed; %OpenDSS file Generator for transformer  
energymeter; %OpenDSS file Generator for Energy Meter
%load_indexed; % FIX missing phases
fclose('all');


%Deva_load;
