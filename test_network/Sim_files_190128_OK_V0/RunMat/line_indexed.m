SkipNum = [1505; 2482];

OutDir = [pwd filesep '..' filesep 'RunDss' filesep];
MatDir = [pwd filesep '..' filesep 'mat' filesep];

notmonitored = [];
size_line = size(Line);
size_line = size_line(1);
line = fopen([OutDir 'Line_indexed_check.txt'], 'w'); %name of the file can be replaced by Feedername_line
monitor_feeder = fopen([OutDir 'monitor_feeder.txt'], 'w');
monitor_line = fopen([OutDir 'monitor_line.txt'], 'w');

mf_text = ('New Monitor.feeder%d_PQ_vs_Time Line.feeder%d 2 Mode=1 ppolar=0\r\nNew Monitor.feeder%d_VI_vs_Time Line.feeder%d 2 Mode=0\r\n');
ml_text = ('New Monitor.LINE%d_PQ_vs_Time Line.%d 2 Mode=1 ppolar=0\r\nNew Monitor.LINE%d_VI_vs_Time Line.%d 2 Mode=0\r\n');
mytext = ('New Line.feeder%1.0f Bus1=%1.0f.1.2.3.4 Bus2=%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytext_seg = ('New Line.%1.0f Bus1=%1.0f.1.2.3.4 Bus2=%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytextline = ('New Line.feeder%1.0f Bus1=%1.0f_%d.1.2.3.4 Bus2=%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytexttransbus = ('New Line.feeder%1.0f Bus1=%1.0f.1.2.3.4 Bus2=%d_%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytext_cktbk = ('New Line.cktbk%1.0f Bus1=%1.0f_%d.1.2.3.4 Bus2=%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytexttransbusopen = ('!New Line.feeder%1.0f Bus1=%1.0f.1.2.3.4 Bus2=%d_%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytextopenconnection = ('!New Line.feeder%1.0f Bus1=%1.0f_%d.1.2.3.4 Bus2=%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
lineindex = [];
feederindex = [];
linecount = 2 * txno - 1;
feedercount = 0;
%Segment=Seg;
openlineattx = [];
for(i = 1:size_line)

m = find(any(transformer == Line(i, 5), 2));
x = [transformer(m, 4) transformer(m, 5)];
len = sqrt((x(1) - Line(i, 7)) ^ 2 + (x(2) - Line(i, 8)) ^ 2); %length of busbar
if (lnraw{i + 1, 10} == 'C')
    feedercount = feedercount + 1;
    %lineindex=[lineindex; linecount Line(i,1)];
    feederindex = [feederindex; feedercount Line(i, 1)];
    k1 = find(any(busindex_new(:, 1) == Line(i, 5), 2));
 
    fprintf(line, mytexttransbus, feederindex(feedercount, 1), busindex_new(k1, 5), busindex_new(k1, 5), Line(i, 2), 205, len);
    fprintf(monitor_feeder, mf_text, feederindex(feedercount, 1), feederindex(feedercount, 1), feederindex(feedercount, 1), feederindex(feedercount, 1));
 
else
    %fprintf(line,mytexttransbusopen,feederindex(feedercount,1),busindex_new(k1,5),busindex_new(k1,5),Line(i,2),205,1);
    openlineattx = [openlineattx; Line(i, 1)];
end
end

cktbrk_count = 0;
crbindex = [];
for(i = 1:size_line)
m = find(any(transformer == Line(i, 5), 2));
x = [transformer(m, 4) transformer(m, 5)];

len = sqrt((x(1) - Line(i, 7)) ^ 2 + (x(2) - Line(i, 8)) ^ 2); %length of busbar
%if(lnraw{i+1,10}=='C')
k2 = find(any(busindex_new(:, 1) == Line(i, 5), 2));
k3 = find(any(busindex_new(:, 1) == Line(i, 6), 2));
if (lnraw{i + 1, 10} == 'C')
    cktbrk_count = cktbrk_count + 1;
    %lineindex=[lineindex; linecount Line(i,1)];
    crbindex = [crbindex; cktbrk_count Line(i, 1)];
    fprintf(line, mytext_cktbk, crbindex(cktbrk_count, 1), busindex_new(k2, 5), Line(i, 2), busindex_new(k3, 5), 102, 0.5);
 
else
    %fprintf(line,mytextopenconnection,feederindex(feedercount,1),busindex_new(k2,5),Line(i,2),busindex_new(k3,5),205,len);
end
%end

end

%loop for first and last coordinate of each segment
size_segment = size(Segment);
size_segment = size_segment(1);
first_cord = zeros(1, size_segment);
first_cord(1) = 1;
for(i = 1:size_segment - 1) %%loop for first coordinate of the line
count_cord = nnz(buscord(:, 1) == Segment(i, 1));
first_cord(i + 1) = first_cord(i) + count_cord;
end
size_buscord = size(buscord);
size_buscord = size_buscord(1);
last_cord = zeros(1, size_segment);
for(i = 1:size_segment - 1) %%loop for first coordinate of the line
count_cord = nnz(buscord(:, 1) == Segment(i, 1));
last_cord(i) = first_cord(i + 1) - 1;
end
last_cord(size_segment) = size_buscord;

%%line distance
line_length = zeros(size_buscord, 1);
for(i = 1:size_segment)
	for(j = first_cord(i):(last_cord(i) - 1))
		line_length(j) = sqrt((buscord(j, 7) - buscord(j + 1, 7)) ^ 2 + (buscord(j, 8) - buscord(j + 1, 8)) ^ 2);
		if (line_length(j) < 0.001) %replace line segment less than 0.001 with 0.001
			line_length(j) = 0.001;
		end
	end
end

%Ordering the bus1 and bus2 to remove duplicacy in bus 2 for monitoring
%purpose eventhough the ordering is already done by seg_crazy this part of
%code is to check if there is some discrepcancy in seg_crazy

segname = [];
segname_n = [];
mytext_first = ('New Line.%d_%d Bus1=%1.0f.1.2.3.4 Bus2=%d_%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytext_last = ('New Line.%d Bus1=%d_%d.1.2.3.4 Bus2=%1.0f.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytextsegment = ('New Line.%d_%d Bus1=%d_%d.1.2.3.4 Bus2=%1.0f_%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');

mytext_first_m = ('New Line.%d_%d Bus1=%1.0f.1.2.3.4 Bus2=%d_m%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytext_last_m = ('New Line.%d Bus1=%d_m%d.1.2.3.4 Bus2=%1.0f.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');
mytextsegment_m = ('New Line.%d_%d Bus1=%d_m%d.1.2.3.4 Bus2=%1.0f_m%d.1.2.3.4 phases=4 Linecode=%d Length=%0.3f Units=m\r\n');

for(i = 1:size_segment)
lineindex = [lineindex; linecount Segment(i, 1)];
k6 = find(any(busindex_new(:, 1) == Segment(i, 7), 2));
k6 = k6(1);
k7 = find(any(busindex_new(:, 1) == Segment(i, 6), 2));
k7 = k7(1);
check1 = find(any(busindex_new(k6, 5) == segname, 2));
check2 = find(any(busindex_new(k7, 5) == segname, 2));
t1 = find(any(openlineattx == Segment(i, 4), 2));
	SkipMe = any(linecount == SkipNum);
	if (SkipMe)
		linecount = linecount + 1;
		continue;
	end

if isempty(t1)
    if (last_cord(i) - first_cord(i) > 1)
        if isempty(check1)
            for(j = first_cord(i))
							fprintf(line, mytext_first, linecount, buscord(j, 6), busindex_new(k7, 5), busindex_new(k6, 5), buscord(j + 1, 6), code(i), line_length(j));
						end
        for(j = first_cord(i) + 1:last_cord(i) - 2)
					fprintf(line, mytextsegment, linecount, buscord(j, 6), busindex_new(k6, 5), buscord(j, 6), busindex_new(k6, 5), buscord(j + 1, 6), code(i), line_length(j));
				end
    segname = [segname; busindex_new(k6, 5)];
 
    for(j = last_cord(i) - 1)
    fprintf(line, mytext_last, linecount, busindex_new(k6, 5), buscord(j, 6), busindex_new(k6, 5), code(i), line_length(j));
end

elseif isempty(check2)
    for(j = first_cord(i))
    fprintf(line, mytext_first, linecount, buscord(j, 6), busindex_new(k7, 5), busindex_new(k7, 5), buscord(j + 1, 6), code(i), line_length(j));
end

for(j = first_cord(i) + 1:last_cord(i) - 2)
fprintf(line, mytextsegment, linecount, buscord(j, 6), busindex_new(k7, 5), buscord(j, 6), busindex_new(k7, 5), buscord(j + 1, 6), code(i), line_length(j));
end
segname = [segname; busindex_new(k7, 5)];
for(j = last_cord(i) - 1)
fprintf(line, mytext_last, linecount, busindex_new(k7, 5), buscord(j, 6), busindex_new(k6, 5), code(i), line_length(j));
end

else
    for(j = first_cord(i))
    fprintf(line, mytext_first_m, linecount, buscord(j, 6), busindex_new(k7, 5), busindex_new(k7, 5), buscord(j + 1, 6), code(i), line_length(j));
end

for(j = first_cord(i) + 1:last_cord(i) - 2)
fprintf(line, mytextsegment_m, linecount, buscord(j, 6), busindex_new(k7, 5), buscord(j, 6), busindex_new(k7, 5), buscord(j + 1, 6), code(i), line_length(j));
end
segname_n = [segname_n; busindex_new(k7, 5) i];

for(j = last_cord(i) - 1)
fprintf(line, mytext_last_m, linecount, busindex_new(k7, 5), buscord(j, 6), busindex_new(k6, 5), code(i), line_length(j));
end
end
else
    for(j = first_cord(i))
    k9 = find(any(busindex_new(:, 1) == Segment(i, 6), 2));
    k10 = find(any(busindex_new(:, 1) == Segment(i, 7), 2));
    k9 = k9(1);
    k10 = k10(1);
    fprintf(line, mytext_seg, linecount, busindex_new(k9, 5), busindex_new(k10, 5), code(i), line_length(j));

end
end
fprintf(monitor_line, ml_text, linecount, linecount, linecount, linecount);
save([MatDir 'linecount.mat'], 'linecount');
else
    %these line are for those freely floating segments in rawdata
    %which has no connections anywhere
    %k11=find(any(busindex_new(:,1)==Segment(i,6),2));
    % k11=k11(1);
    %k12=find(any(busindex_new(:,1)==Segment(i,7),2));
    %k12=k12(1);
    %fprintf(line,mytextopenconnection,linecount,busindex_new(k11,5),0,busindex_new(k12,5),code(i),0);
    notmonitored = [notmonitored; linecount];
 
end
linecount = linecount + 1;
end


fclose(line);
fclose(monitor_feeder);
fclose(monitor_line);
fclose('all');
% nom = fopen([OutDir 'notmonitored.txt'], 'w');
% fprintf(nom, '%d\r\n ', notmonitored);
% fclose(nom);
