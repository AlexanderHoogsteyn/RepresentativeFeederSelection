OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];
OutFile = [OutDir 'energymeter_ind.txt'];

 
source =fopen(OutFile,'w');
mytext=('New energymeter.m%d LINE.mv%d 1\r\n');
for i=1:txno
    fprintf(source,mytext,i,i);
end
fclose(source);
