OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];
 
lshape = fopen([OutDir 'Loadshape.txt'],'w');
mytext=('New Loadshape.Shape_%d npts=%d minterval=60 mult=(file=day_20_profile\\shape_%d.csv) useactual=true\r\n');
for i = 1 : n_load
    fprintf(lshape,mytext,i,24,i);
end
fclose(lshape);
