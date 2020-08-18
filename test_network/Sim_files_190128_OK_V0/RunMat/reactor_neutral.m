OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];

reactor=fopen([OutDir 'neutral_ind.txt'],'w'); %name of tx can be replaced by Feeder_transformer
Mytext=('New Reactor.grnd%d phases=1 bus1=%d.4 bus2=%d.0 R=5 X=0.01 \r\n');
for(i=1:(txno-1))
fprintf(reactor,Mytext,i,i+txno-1,i+txno-1);
end
fclose(reactor);
