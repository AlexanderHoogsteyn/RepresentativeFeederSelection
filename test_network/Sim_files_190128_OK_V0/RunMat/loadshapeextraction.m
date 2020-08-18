 GisDir  = [pwd filesep '..' filesep 'GIS_data' filesep];

file_load=[GisDir 'load.xlsx'];%load details here
[load_loc load_loc_raw]=xlsread(file_load,'load');
for i=1:length(load_loc)
    shape(i).hrs=zeros(days*24+1,1);
end
load_loc_raw2 = load_loc_raw(2:end, 1);
notfulldata=[];
fulldata=[];
date=[];
SampleCount = days * 24 + 1;
for file = 1 : fileno
filename=sprintf('%sfile%d.xlsx', GisDir, file); %change the name file if the name is else[shape] 
[~,~, text]=xlsread(filename,'C:C');

pow1 = xlsread(filename,'G:G');
pow1 = pow1(2:end);
n = length(pow1);
text = text(2 : n + 1, 1);

for i = 1 : length(load_loc)
	m = strcmp(load_loc_raw2{i}, text);
	% m=find();
%	if ~isempty(m)
		try
		if (sum(m) == SampleCount)
			LoadShape = pow1(m);
			k = LoadShape > 1e3;
			LoadShape(k) = LoadShape(k) * 1e-3;
			shape(i).hrs(1 : SampleCount) = LoadShape;
%        fulldata=[fulldata; i file];
%       for j=1:481
%             if (shape(i).hrs(j))>1000
%                 shape(i).hrs(j)=shape(i).hrs(j)/1000;
%             end
%        end
%     else
%          last_value=length(m);
%         notfulldata=[notfulldata; i file length(m)];
%           %date=[date; text{m(last_value),2}];
		end
			catch
				fprintf(' ');
			end
%	end
end
end
