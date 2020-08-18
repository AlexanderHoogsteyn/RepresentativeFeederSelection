OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];
OutFile = [OutDir 'Buscord_indexed.txt'];

size_cord=[];
for(k=1:size_segment)%loop to find size of coordinate in each segment
    count_cord=nnz(buscord(:,1)==Segment(k,1));
    size_cord=[size_cord count_cord];
end
bscord=fopen(OutFile,'w'); %name can be replaced by *Feedername*_Buscord
mytext_tx = 'tx%d %f %f \r\n';
mytext = '%d	%f %f \r\n';
mytextline = '%d_%d %f %f \r\n';


for(i=1:txno-1) 
    m1=find(any(busindex_new==transformer(i,6),2));
fprintf(bscord,mytext,busindex_new(m1,5),transformer(i,4),transformer(i,5));

end
for(i=1:txno-1)
m2=find(any(busindex_new==transformer(i,11),2));;    
fprintf(bscord,mytext,busindex_new(m2,5),transformer(i,4),transformer(i,5));

end
for(k=1:size_line)
    m3=find(any(busindex_new==Line(k,5),2));
    fprintf(bscord,mytextline,busindex_new(m3,5),Line(k,2),Line(k,7),Line(k,8));
   
end
l=1;

for(i=1:size_segment)
        m4=find(any(busindex_new(:,1)==Segment(i,6),2));
        m4=m4(1);
        fprintf(bscord,mytext,busindex_new(m4,5),buscord(l,7),buscord(l,8));
        l=l+size_cord(i);
end
l=0;


   for(i=1:size_segment)
       l=l+size_cord(i);
       m5=find(any(busindex_new(:,1)==Segment(i,7),2));
        m5=m5(1);
       fprintf(bscord,mytext,busindex_new(m5,5),buscord(l,7),buscord(l,8));
      
   end 
   mytext_cord=('%d_%d %f %f\r\n');
   mytext_cord_m=('%d_m%d %f %f\r\n');
   busseg=[];
   
for(i=1:size_segment)
    m6=find(any(busindex_new(:,1)==Segment(i,7),2));
    m7=find(any(busindex_new(:,1)==Segment(i,6),2));
    m6=m6(1);
    m7=m7(1);
   check1=find(any(busindex_new(m6,5)==busseg,2));
   check2=find(any(busindex_new(m7,5)==busseg,2));
    if(last_cord(i)-first_cord(i)>1)  
        if isempty(check1) 
            for(j=first_cord(i)+1:last_cord(i)-1)
                fprintf(bscord,mytext_cord,busindex_new(m6,5),buscord(j,6),buscord(j,7),buscord(j,8));
            end
            busseg=[busseg;busindex_new(m6,5)];
        elseif isempty(check2)
            for(j=first_cord(i)+1:last_cord(i)-1)
                fprintf(bscord,mytext_cord,busindex_new(m7,5),buscord(j,6),buscord(j,7),buscord(j,8));
            end
            busseg=[busseg;busindex_new(m7,5)]; 
        else
            for(j=first_cord(i)+1:last_cord(i)-1)
                fprintf(bscord,mytext_cord_m,busindex_new(m7,5),buscord(j,6),buscord(j,7),buscord(j,8));
            end
        end   
    end
end  
fclose(bscord);
