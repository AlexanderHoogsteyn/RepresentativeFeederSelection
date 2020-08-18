%creating bus cord
size_cord=[];
size_segment=size(Segment);
size_segment=size_segment(1);
first_cord=zeros(1,size_segment);
first_cord(1)=1;
txno=size(txID);
txno=txno(1);
for(i=1:size_segment-1)  %%loop for first coordinate of the line
    count_cord=nnz(buscord(:,1)==Segment(i,1));
    first_cord(i+1)=first_cord(i)+count_cord;
end
size_buscord=size(buscord);
size_buscord=size_buscord(1);
last_cord=zeros(1,size_segment);
for(i=1:size_segment-1)  %%loop for first coordinate of the line
    count_cord=nnz(buscord(:,1)==Segment(i,1));
    last_cord(i)=first_cord(i+1)-1;
end
last_cord(size_segment)=size_buscord;

for(k=1:size_segment)
    count_cord=nnz(buscord(:,1)==Segment(k,1));
    size_cord=[size_cord count_cord];
end


%bscord=fopen('Pola_Buscord_modified.txt','w'); %name can be replaced by *Feedername*_Buscord
%mytext_tx=('tx%d %f %f \r\n');
%mytext=('%d	%f %f \r\n');
%mytextline=('%d_%d %f %f \r\n');
busindex=[];
buscount=1;

for(i=1:txno-1)
busindex=[busindex;transformer(i,6) 0 transformer(i,4) transformer(i,5)];    
%fprintf(bscord,mytext,transformer(i,6),transformer(i,4),transformer(i,5));
buscount=buscount+1;
end
for(i=1:txno-1)
busindex=[busindex;transformer(i,11) 0 transformer(i,4) transformer(i,5)];    
%fprintf(bscord,mytext,transformer(i,11),transformer(i,4),transformer(i,5));
buscount=buscount+1;
end
    busseg=[];
    count=1;
    size_line=size(Line);
    size_line=size_line(1);
for(k=1:size_line)
    busseg=[busseg;Line(k,5) Line(k,2) Line(k,7) Line(k,8) count]; 
    %fprintf(bscord,mytextline,Line(k,5),Line(k,2),Line(k,7),Line(k,8));
    count=count+1;
end

l=0;
   for(i=1:size_segment)
       l=l+size_cord(i); 
       busindex=[busindex;Segment(i,7) 0 buscord(l,7) buscord(l,8)];
       %fprintf(bscord,mytext,Segment(i,7),buscord(l,7),buscord(l,8));
       buscount=buscount+1;
   end 
   mytext_cord=('%d%d %f %f\r\n');
l=1;

for(i=1:size_segment)
        busindex=[busindex;Segment(i,6) 0 buscord(l,7) buscord(l,8)];
        %fprintf(bscord,mytext,Segment(i,6),buscord(l,7),buscord(l,8));
        l=l+size_cord(i);
        buscount=buscount+1;
end   
for(i=1:size_segment)
    for(j=first_cord(i)+1:last_cord(i)-1)
        busseg=[busseg;Segment(i,7) buscord(j,6) buscord(j,7) buscord(j,8) count];
        %fprintf(bscord,mytext_cord,Segment(i,7),buscord(j,6),buscord(j,7),buscord(j,8));
        count=count+1;
    end
    end
   
%fclose(bscord);

[~,new]=unique(busindex,'rows');
busindex_new=busindex(sort(new),:);
no=size(busindex_new);
no=no(1);
busindex_new(:,5)=1:no;