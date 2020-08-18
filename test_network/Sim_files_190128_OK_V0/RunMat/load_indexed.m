%  GisDir  = [pwd filesep '..' filesep 'GIS_data' filesep];
% 
% file_load=[GisDir 'Inventario_Contadores_Pola_de_Siero.xlsx'];%load details here
% [load_loc load_loc_raw]=xlsread(file_load,'Inventario_Contadores_Pola_de_S');
% for i=1:length(load_loc)
%     shape(i).hrs=zeros(days * 24 + 1,1);
% end
% notfulldata=[];
% fulldata=[];
% date=[];
% for file=1:fileno
% filename=sprintf('%sfile%d.xlsx', GisDir, file); %change the name file if the name is else[shape] 
% [~,~,text]=xlsread(filename,'C:E');
% pow1=xlsread(filename,'G:G');
% 
% for i=1:length(load_loc)
%     m=find(strcmp(load_loc_raw{i+1,1},text));
%     if ~isempty(m)
%     if length(m)==days*24+1 
%       shape(i).hrs(1:days*24+1)=pow1(m-1,1);
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
%       end
%     end
% end
% end

GetRandPhase(0); % Reset random number genrator;
OutDir  = [pwd filesep '..' filesep 'RunDss' filesep];

load1=fopen([OutDir 'Load_indexed.txt'], 'w');
phaseerror=fopen([OutDir 'error_phasemissing.txt'],'w'); %Uncomment if you want
%to see the load where there is phase error dont forget to uncomment fclose
%command at the end of this script
acometidaserror=fopen([OutDir 'ac_error.txt'],'w');
%uncomment previous line to see the floating loads 
%dont forget to uncomment fclose
errortext_phase='%s  %s  phase missing random phase=%d \r\n';
erracc='%s  %s  acometidasmissing \r\n';

n_load=size(load_loc);
n_load=n_load(1);
n_loadbus=size(loadbus);
n_loadbus=n_loadbus(1);
size_phase_raw=size(phaseraw);
size_phaseraw=size_phase_raw(1);
bus1=zeros(n_load,1);
myloadtext_1ph='New Load.LOAD%d Phases=1 Bus1=%d.%d.4 kV=0.23 kW=1 PF=0.95  daily=Shape_%d\r\n';
myloadtext_3ph='New Load.LOAD%d Phases=3 Bus1=%d.1.2.3.4 kV=0.4 kW=1 PF=0.95  daily=Shape_%d\r\n';
myloadtext_neg='!New Load.LOAD%d Phases=1 Bus1=%d.%d.4 kV=0.4 kW=1 PF=0.95  daily=Shape_%d\r\n';
myloadtext_3phneg='!New Load.LOAD%d Phases=3 Bus1=%d.1.2.3.4 kV=0.4 kW=1 PF=0.95  daily=Shape_%d\r\n';
load_ind=[];

for(i=1:n_load)
        loc=find(any(load_loc(i,1)==loadbus(:,2),2));
        if ~isempty(loc)    
        bus1(i)=loadbus(loc(1),3);%bus list
  
        else
            bus1(i)=0;
        fprintf(acometidaserror,erracc,load_loc_raw{i+1,1},load_loc_raw{i+1,18});
        end
    phase=0;
    %Indexing load
        %load_loc(i).name=sprintf('%s',load_loc_raw{i+1,1});
     %load_loc(i).index=i;
        
     l1=find(any(busindex_new(:,1)==bus1(i),2));
     if length(l1)>1
     l1=l1(1);
     end
    %checking phase of single phase load
    if load_loc_raw{i+1,11}(1)=='M'
       m=find(strcmp(load_loc_raw{i+1,1},phaseraw));
       if length(m)==1
                    if(phaseraw{m,2}=='R')
                        phase=1;
                    elseif(phaseraw{m,2}=='S')
                        phase=2;
                    elseif(phaseraw{m,2}=='T')
                        phase=3;
                    else
                       phase = GetRandPhase(); 
                    end
       elseif length(m)>1
           for c=1:length(m)
               if(phaseraw{m(c),2}=='R')
                        phase=1;
                        break;
                    elseif(phaseraw{m(c),2}=='S')
                        phase=2;
                        break;
                    elseif(phaseraw{m(c),2}=='T')
                        phase=3;
                        break;
               end
           end
       end
           if phase==0
               phase = GetRandPhase();
                fprintf(phaseerror,errortext_phase,load_loc_raw{i+1,1},load_loc_raw{i+1,20},phase);
           end
                       
       if isempty(l1)
           fprintf(load1,myloadtext_neg,i,0,phase,i);
       else
        fprintf(load1,myloadtext_1ph,i,busindex_new(l1,5),phase,i);
       end     
        
    else
        if isempty(l1)
           fprintf(load1,myloadtext_3phneg,i,0,i);
       else
        fprintf(load1,myloadtext_3ph,i,busindex_new(l1,5),i);
        end
    end
end

fclose(acometidaserror);
fclose(phaseerror);
fclose(load1);  
