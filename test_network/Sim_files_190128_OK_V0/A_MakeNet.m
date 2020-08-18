%run me for network data extraction
%it extracts the network data from the GIS data given by EDP
clear;
oldfolder=cd('RunMat');
ReadGis;
cd(oldfolder);
