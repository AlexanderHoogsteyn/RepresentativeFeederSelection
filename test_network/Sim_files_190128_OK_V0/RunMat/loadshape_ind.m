OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];
n_load = length(shape);
lshape = fopen([OutDir 'Loadshape.txt'],'w');
mytext=('New Loadshape.Shape_%d npts=%d minterval=60 mult=(file=day_%d_profile\\shape_%d.csv) useactual=true\r\n');
for(i=1:n_load)
    fprintf(lshape,mytext,i,days*24+1,days,i);
end
fclose(lshape);

