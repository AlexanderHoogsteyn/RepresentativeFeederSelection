%Arran
Seg=Segment;
bus_cord=buscord;
net_start=Line(:,6);
closed_start=[];
%finding all the starting Point from Breaker
for i=1:size(net_start)
    if (lnraw{i+1,10}=='C')
        closed_start=[closed_start; net_start(i) 0 0];
    end
end
segcode=[Line(:,1) zeros(length(Line(:,1)),1)];
Seg(:,8)=zeros(length(Seg(:,1)),1);
%fuse start points
fuse_start=fuse(:,4:5);
fuse_closed=[];
for(i=1:length(fuse_start))
    if(fuseraw{i+1,7}=='C')
        f1=find(any(busindex_new(:,1)==fuse_start(i,1),2));
        f2=find(any(busindex_new(:,1)==fuse_start(i,2),2));
        fuse_closed=[fuse_closed; busindex_new(f1,5) busindex_new(f2,5)];
    end
end

segcon=[];
seg_crazy=[];
crazy=[];
single=[];
 for(i=1:size_segment) %for fixing the starting coordinate
        start1=find(any(closed_start==Seg(i,7),2));
        start2=find(any(closed_start==Seg(i,6),2));
        f1=find(any(busindex_new(:,1)==Seg(i,7),2));
        f1=f1(1);
        f2=find(any(busindex_new(:,1)==Seg(i,6),2));
        f2=f2(1);
        last1=[];
        last2=[];
        unique1=find(any(Seg(:,6)==Seg(i,6),2));
        unique2=find(any(Seg(:,7)==Seg(i,6),2));
        unique3=find(any(Seg(:,6)==Seg(i,7),2));
        unique4=find(any(Seg(:,7)==Seg(i,7),2));
        last1=[unique1;unique2];
        last2=[unique3;unique4];
        
        if length(start2)==1
            seg1=find(any(segcode(:,1)==Seg(i,4),2));
            segcode(seg1,2)=segcode(seg1,2)+1;
            Seg(i,8)=segcode(seg1,2); %it gives the segment number
            segcon=[segcon; busindex_new(f1,5) i 1 busindex_new(f1,1)];
        elseif length(start1)==1
            seg1=find(any(segcode(:,1)==Seg(i,4),2));
            segcode(seg1,2)=segcode(seg1,2)+1;%count to keep track of total number of segments in a feeder
            Seg(i,8)=segcode(seg1,2);
            segcon=[segcon; busindex_new(f2,5) i 3 busindex_new(f2,1)];
            temp=Seg(i,7);      
            Seg(i,7)=Seg(i,6);
            Seg(i,6)=temp;
            temp=[bus_cord(first_cord(i):last_cord(i),7:8)];
            s=last_cord(i)-first_cord(i)+1;
            for(t=first_cord(i):last_cord(i))
                bus_cord(t,7)=temp(s,1);
                bus_cord(t,8)=temp(s,2);
                s=s-1;
            end
        elseif length(last2)==1&&length(last1)==1
            %single=[single;start1];
        elseif length(last2)==1&&length(unique4)==1
            %segcon=[segcon; busindex_new(f1,5) i 7];
         elseif length(last1)==1&&length(unique1)==1
            %segcon=[segcon; busindex_new(f2,5) i 8];
            temp=Seg(i,7);      
            Seg(i,7)=Seg(i,6);
            Seg(i,6)=temp;
            temp=[bus_cord(first_cord(i):last_cord(i),7:8)];
            s=last_cord(i)-first_cord(i)+1;
            for(t=first_cord(i):last_cord(i))
                bus_cord(t,7)=temp(s,1);
                bus_cord(t,8)=temp(s,2);
                s=s-1;
            end
        end
 end
 
len_diff=1;
 SumVal=10;
 while len_diff>0
    len1=length(segcon);
     seg_crazy=[]; 
     cr=[]; 
      for i=1:size_segment 
        n1=find(any(segcon(:,2)==i,2));
        if isempty(n1)
        start1=find(any(net_start==Seg(i,7),2));
        cr=[cr;start1];
        f1=find(any(busindex_new(:,1)==Seg(i,7),2));
        f1=f1(1);
        f2=find(any(busindex_new(:,1)==Seg(i,6),2));
        f2=f2(1);
        ch1=find(any(segcon(:,1)==busindex_new(f1,5),2));
        ch2=find(any(segcon(:,1)==busindex_new(f2,5),2)); 
       
        if isempty(ch2)&&~isempty(ch1)
            seg1=find(any(segcode==Seg(i,4),2));
            segcode(seg1,2)=segcode(seg1,2)+1;
            Seg(i,8)=segcode(seg1,2);
            seg_crazy=[seg_crazy;  Seg(i,6) Seg(i,7) i];
            segcon=[segcon; busindex_new(f2,5) i SumVal+1 busindex_new(f1,1)];
            temp=Seg(i,7);      
            Seg(i,7)=Seg(i,6);
            Seg(i,6)=temp;
            temp=[bus_cord(first_cord(i):last_cord(i),7:8)];
            s=last_cord(i)-first_cord(i)+1;
            
            for(t=first_cord(i):last_cord(i))
                bus_cord(t,7)=temp(s,1);
                bus_cord(t,8)=temp(s,2);
                s=s-1;
            end
            
        elseif ~isempty(ch2)&&isempty(ch1)
            seg1=find(any(segcode==Seg(i,4),2));
            segcode(seg1,2)=segcode(seg1,2)+1;
            Seg(i,8)=segcode(seg1,2);
            segcon=[segcon; busindex_new(f1,5) i SumVal busindex_new(f2,1)];
        end
        
     end
      end
 for(i=1:length(fuse_closed))
     b1=find(any(busindex_new(:,5)==fuse_closed(i,1),2));
     b2=find(any(busindex_new(:,5)==fuse_closed(i,2),2));
     a1=find(any(segcon(:,1)==fuse_closed(i,1),2));
     a2=find(any(segcon(:,1)==fuse_closed(i,2),2));
     if isempty(a1)&&~isempty(a2)
         segcon=[segcon; busindex_new(b1,5) 0 0 busindex_new(a2,1)];
     elseif ~isempty(a1)&&isempty(a2)
         segcon=[segcon; busindex_new(b2,5) 1 1 busindex_new(a1,1)];
     end
 end
      
 SumVal=SumVal+10;
 len_diff=length(segcon)-len1;
 end
 
 Segment=Seg;
 buscord=bus_cord;