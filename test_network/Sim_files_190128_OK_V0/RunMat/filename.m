function [transformer, Line, lnraw, Segment,Segraw, buscord, loadbus,txID, fuse, fuseraw, load_loc, load_loc_raw, phaseraw]=filename

	GIS_Dir =[pwd filesep '..' filesep 'GIS_data' filesep];
	file_master=[GIS_Dir , 'Red_PolaSiero_16mayo18.xlsx'];%master file here
	file_load=[GIS_Dir, 'Inventario_Contadores_Pola_de_Siero.xlsx'];%load details here
	%file_loadshape=('Curva_Carga_Contadores_Pola4CTs.xlsx'); %loadshape excel file name
	file_phase=[GIS_Dir, 'phase meters.xls'];


	[transformer txID]=xlsread(file_master,'CT - TRAFO');
	[Line lnraw]=xlsread(file_master,'Linea BT');
	[Segment Segraw]=xlsread(file_master,'Segmento BT');
	buscord=xlsread(file_master,'Coordenadas Segmentos');
	loadbus=xlsread(file_master,'Acometidas');
	[fuse fuseraw]=xlsread(file_master,'Fusible');
	[load_loc load_loc_raw]=xlsread(file_load,'Inventario_Contadores_Pola_de_S');
	[phase phaseraw]=xlsread(file_phase);
	txID=txID(:,8);

 