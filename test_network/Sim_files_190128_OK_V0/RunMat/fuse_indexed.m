OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];
OutFile = [OutDir 'fuse_indexed.txt'];

fuseno=size(fuse);
fuseno=fuseno(1);
fuse_line=fopen(OutFile, 'w');
mytext_fc=('New Line.fuse%d Bus1=%d.1.2.3.4 Bus2=%d.1.2.3.4 phases=4 Linecode=%d Length=0.5 Units=m\r\n');
mytext_fo=('!New Line.fuse%d Bus1=%d.1.2.3.4 Bus2=%d.1.2.3.4 phases=4 Linecode=%d Length=0.5 Units=m\r\n');

fuseindex=[];
for(i=1:fuseno)
    if(fuseraw{i+1,7}=='C')
        f1=find(any(busindex_new(:,1)==fuse(i,4),2));
         f2=find(any(busindex_new(:,1)==fuse(i,5),2));
        fprintf(fuse_line,mytext_fc,i, busindex_new(f1,5), busindex_new(f2,5),102);
    else
       f1=find(any(busindex_new(:,1)==fuse(i,4),2));
         f2=find(any(busindex_new(:,1)==fuse(i,5),2));
        fprintf(fuse_line,mytext_fo,i, busindex_new(f1,5), busindex_new(f2,5),102);  
    end
    fuseindex=[fuseindex;fuse(i,1) i];
end
fclose(fuse_line);

