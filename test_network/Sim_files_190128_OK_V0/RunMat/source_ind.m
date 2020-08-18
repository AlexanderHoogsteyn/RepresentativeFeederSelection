OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];

source=fopen([OutDir 'source2txline_ind.txt'],'w');
source_mon=fopen([OutDir 'MVmon.txt'],'w');
mytext=('New Line.mv%1.0f Bus1=Source Bus2=%d phases=3 Linecode=%d Length=5 Units=m\r\n');
mHT_text=('New Monitor.MV%d_PQ_vs_Time Line.mv%d 2 Mode=1 ppolar=0\r\nNew Monitor.MV%d_VI_vs_Time Line.mv%d 2 Mode=0\r\n');
mvline=[];
for i=1:txno-1
    s1=find(any(busindex_new(:,1)==transformer(i,6),2));
    fprintf(source,mytext,i, busindex_new(s1,5),101);
    fprintf(source_mon,mHT_text,i,i,i,i);

    mvline=[mvline; transformer(i,1) i];
end


fclose(source);
fclose(source_mon);
