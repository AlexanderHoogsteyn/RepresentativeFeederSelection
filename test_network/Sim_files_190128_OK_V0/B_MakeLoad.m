
%run me for load data extraction
%it extracts the load data from the GIS data given by EDP
days = 20; %no of days of load to be manually edited
fileno = 7; % no of loadfile to be manually edited
oldfolder=cd('RunMat');
save_loadshape;
% shortcut_loadshape;
cd(oldfolder);
