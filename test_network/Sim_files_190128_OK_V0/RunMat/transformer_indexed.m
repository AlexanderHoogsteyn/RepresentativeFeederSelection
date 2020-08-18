OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];
MatDir  = [pwd filesep '..' filesep 'mat' filesep];

txno=size(txID);
txno=txno(1)-1;
tx=fopen([OutDir 'Transformer_ind.txt'],'w'); %name of tx can be replaced by Feeder_transformer
Mytext=('New Transformer.%s windings=2  Buses=[%d %d.1.2.3.4] Conns=[Delta Wye] kVs=[22 0.420] kVAs=[%d %d] XHL=%.1f sub=y \r\n');
for(i=1:(txno))
    t1=find(any(busindex_new(:,1)==transformer(i,6),2));
    t2=find(any(busindex_new(:,1)==transformer(i,11),2));
    fprintf(tx,Mytext,txID{(i+1),1}, busindex_new(t1,5), busindex_new(t2,5),transformer(i,10),transformer(i,10),4);
end
save([MatDir 'txno.mat'],'txno');

fclose(tx);
