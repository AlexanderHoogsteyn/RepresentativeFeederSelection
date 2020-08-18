if (~exist('days', 'var'))
	days = 20;
end

if (~exist('fileno', 'var'))
	fileno = 7;
end

loadshapeextraction;
loadshape_ind;


OutDir  = [pwd filesep '..' filesep 'RunDss' filesep 'day_20_profile' filesep];
[status,message,messageid] = mkdir(OutDir);
%cd ('RunDss\day_20_profile');
 for j = 1 : length(load_loc)
		dlmwrite(sprintf('%sshape_%d.csv', OutDir,j),shape(j).hrs)
 end
 
 
