% script for evaluating artifact removal across different SNR levels
% N.B. can take between 12 - 48 hours when processing thousands of datasets

%%  ALLEEG1: ground truth datasets
%%  ALLEEG2: artifact-contaminated datasets
%%  ALLEEG3: mixed (ground truth + artifact) datasets

%%  ALLEEG6: raw denoised dataset
%%  ALLEEG7: ASR denoised dataset

%%  ALLEEG9: IClabel denoised dataset
%%  ALLEEG10: MARA denoised dataset
%%  ALLEEG11: GEDAI denoised dataset

% Tomas Ros, CIBM & University of Geneva, 2025

clear all
eeglab nogui

just_testing=false; % for script testing purposes (runs only a few files)
random_number_of_test_files = 50; % Number of test files to run

%% manually set the NOISE parameters below that will be investigated

contaminated_signal_proportion= [100]; % percent of epochs temporally contaminated with artifact e.g. [25 50 75 100]
signal_to_noise_in_db= [-9 ]; % initial data signal-to-noise ratio in decibels e.g. [-9 -6 -3 0 ]


%% automated script begins here %%

%% Read the GROUND TRUTH data
Group1Dir = uigetdir([],'Root directory of CLEAN EEG files (EEGlab set files)');
Group1Index = 1;
DirGroup1 = dir(fullfile(Group1Dir,'*.set'));
FileNamesGroup1 = {DirGroup1.name};
files_in_Group1=numel(FileNamesGroup1);

%% Read the ARTIFACT data
Group2Dir = uigetdir([],'Root directory of ARTIFACT files (EEGlab set files)');
SavePath   = pwd;
Group2Index = 2;
DirGroup2 = dir(fullfile(Group2Dir,'**','*.set'));
FileNamesGroup2 = {DirGroup2.name};
files_in_Group2=numel(FileNamesGroup2);

%% Determine magnetometer channel indices from the first file
if files_in_Group1 > 0
    tempEEG = pop_loadset('filename', FileNamesGroup1{1}, 'filepath', Group1Dir);
    mag_chans = find(strcmp({tempEEG.chanlocs.type}, 'MEG GRAD'));
end

%% initialize tables
varNames = {'clean_EEG_file', 'artifact_file','artifact','temporal_contamination','signal_to_noise','Algorithm','Rsquared'}; 
TABLE_Rsquared_concatenated = table('Size', [0, numel(varNames)], ...
          'VariableTypes', {'string','string', 'string','string','string', 'categorical', 'double'}, ...
          'VariableNames', varNames);

varNames = {'clean_EEG_file','artifact_file','artifact','temporal_contamination','signal_to_noise','Algorithm','RRMSE'}; 
TABLE_RRMSE_concatenated= table('Size', [0, numel(varNames)], ...
          'VariableTypes', {'string','string', 'string','string','string', 'categorical', 'double'}, ...
          'VariableNames', varNames);

varNames = {'clean_EEG_file','artifact_file','artifact','temporal_contamination','signal_to_noise','Algorithm','SNR'}; 
TABLE_SNR_concatenated= table('Size', [0, numel(varNames)], ...
          'VariableTypes', {'string','string', 'string','string','string', 'categorical', 'double'}, ...
          'VariableNames', varNames);


varNames = {'clean_EEG_file','artifact_file','artifact','temporal_contamination','signal_to_noise','Algorithm','time'}; 
TABLE_time_concatenated= table('Size', [0, numel(varNames)], ...
          'VariableTypes', {'string','string', 'string','string','string', 'categorical', 'double'}, ...
          'VariableNames', varNames);



h = waitbar(0,'Please wait...');
for n = 1:length(signal_to_noise_in_db)  % LOOP 1: looping through different SNRs
    waitbar((n-1)/length(signal_to_noise_in_db),h, ...
        sprintf('SNR Loop: %d%% complete',round((n-1)/length(signal_to_noise_in_db)*100))); 
    
    for c = 1:length(contaminated_signal_proportion) % LOOP 2: looping through different TEMPORAL CONTAMINATION levels
    waitbar(((n-1)+ (c-1)/length(contaminated_signal_proportion))/length(signal_to_noise_in_db) ,h, ...
        sprintf('SNR %d/%d, Contam. %d/%d: Initializing...', n, length(signal_to_noise_in_db), c, length(contaminated_signal_proportion)));

    signal_to_noise_linear = 10.^(signal_to_noise_in_db / 20); % signal/noise amplitude ratio  
    noise = 1./signal_to_noise_linear; % noise/signal amplitude ratio  



% bandpass_filter=[1 Inf] % uncomment here and below if additional band-pass filtering is needed


% % Load the GROUND TRUTH data(group 1)
for f = 1:files_in_Group1
    ALLEEG1(f) = pop_loadset('filename', FileNamesGroup1{f}, 'filepath', Group1Dir);
    ALLEEG1(f) = pop_select(ALLEEG1(f), 'channel', mag_chans);
    % ALLEEG1(f) = pop_firws(ALLEEG1(f), 'fcutoff', bandpass_filter, 'ftype', 'bandpass', 'wtype', 'blackman', 'forder', 1408);
   


   ALLEEG1(f).data=reshape(zscore(ALLEEG1(f).data(:)), size(ALLEEG1(f).data) ); % SNR normalise amplitudes of CLEAN DATA

end

% % Load the ARTIFACT data (group 2)
clear keptColumnIndices
clear zeroedColumnIndices
for f = 1:files_in_Group2
   ALLEEG2(f) = pop_loadset('filename', FileNamesGroup2{f}, 'filepath', DirGroup2(f).folder);
   ALLEEG2(f) = pop_select(ALLEEG2(f), 'channel', mag_chans);
    % ALLEEG2(f) = pop_firws(ALLEEG2(f), 'fcutoff', bandpass_filter, 'ftype', 'bandpass', 'wtype', 'blackman', 'forder', 1408);

     min_block_size_in_samples = 1; % default = 1 sample
        max_block_size_in_samples = 1 * ALLEEG2(f).srate; %  default = 1 second

        [ALLEEG2(f).data, keptColumnIndices(f,:), zeroedColumnIndices(f,:)] = retainExactPercentageRandomEEGBlocks(ALLEEG2(f).data, contaminated_signal_proportion(c), min_block_size_in_samples, max_block_size_in_samples);
        temp_data=ALLEEG2(f).data(:,keptColumnIndices(f,:)); 
        ALLEEG2(f).data(:,keptColumnIndices(f,:)) = noise(n)*reshape(zscore(temp_data(:)), size(temp_data) ); % SNR normalise amplitudes of ARTIFACT DATA

end










% % MIX: the GROUND TRUTH and ARTIFACT files
file_mixing_combination=table2array(combinations([1:files_in_Group1],[1:files_in_Group2]));

if just_testing

file_mixing_combination = [randi(files_in_Group1, random_number_of_test_files, 1), ...
                           randi(files_in_Group2, random_number_of_test_files, 1)];          
end









for m=1:length(file_mixing_combination) % LOOP 3: looping through different ARTIFACT TYPES

    ALLEEG3(m)=ALLEEG1(1); % initialize ALLEEG3 dataset
    mixed_dataset_name=[' SNR=' num2str(signal_to_noise_linear(n)) ' contamination=' num2str(contaminated_signal_proportion(c)) ' ' FileNamesGroup1{file_mixing_combination(m,1)} ' + ' FileNamesGroup2{file_mixing_combination(m,2)}]
    ALLEEG3(m) = pop_editset(ALLEEG3(m), 'setname', mixed_dataset_name);
    ALLEEG3(m).data=ALLEEG1(file_mixing_combination(m,1)).data + ALLEEG2(file_mixing_combination(m,2)).data;
    
    % ALLEEG2(m) = pop_saveset( ALLEEG2(m), 'filename',FileNamesGroup2{file_mixing_combination(m,2)},'savemode','onefile'); % optional artifact file save
    % ALLEEG3(m) = pop_saveset( ALLEEG3(m),'filename',mixed_dataset_name, 'savemode','onefile'); % optional MIXED file save
end


%% RUN different denoising methods

% % % RUN RAW (i.e. no denoising)
% for m=1:length(file_mixing_combination)
% 
% tic
% ALLEEG6= ALLEEG3(m);
% time_raw(m,:) = toc;
% 
% denoised_dataset_name=['raw ' ' SNR=' num2str(signal_to_noise_linear(n)) ' contamination=' num2str(contaminated_signal_proportion(c)) ' ' FileNamesGroup1{file_mixing_combination(m,1)} ' + ' FileNamesGroup2{file_mixing_combination(m,2)}]
% % ALLEEG6 = pop_saveset( ALLEEG6,'filename',denoised_dataset_name, 'savemode','onefile'); % optional DENOISED file save
% 
% % % estimate raw denoising quality
% ground_truth_matrix=ALLEEG1(file_mixing_combination(m,1)).data;
% raw_denoised_matrix=ALLEEG6.data;
% 
% raw_Rsquared(m,:)=variance_explained(ground_truth_matrix,raw_denoised_matrix);
% raw_RRMSE(m,:)=relative_RMSE(ground_truth_matrix, raw_denoised_matrix);
% raw_SNR(m,:) = sig_to_noise(ground_truth_matrix,raw_denoised_matrix);
% 
% % vis_artifacts(ALLEEG7,ALLEEG1(file_mixing_combination(m,1)))
% end
% 
% 
% % % RUN ASR 
% 
% for m=1:length(file_mixing_combination)
% 
% tic
% [~, artifact_type, ~] = fileparts(DirGroup2(file_mixing_combination(m,2)).folder)
% if strcmp(artifact_type, 'NOISE') | strcmp(artifact_type, 'NOISE EOG EMG')% additionally use channel rejection for NOISE artifacts
% 
% ASRtemp = pop_clean_rawdata(ALLEEG3(m), 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
% ALLEEG7  = pop_interp(ASRtemp,  ALLEEG1(1).chanlocs, 'spherical');
% time_ASR(m,:) = toc;
% 
% 
% else % no bad channel rejection
% 
% ALLEEG7 = pop_clean_rawdata(ALLEEG3(m), 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off','Highpass','off','BurstCriterion',20,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
% time_ASR(m,:) = toc;
% 
% end
% 
% 
% 
% denoised_dataset_name=['ASR ' ' SNR=' num2str(signal_to_noise_linear(n)) ' contamination=' num2str(contaminated_signal_proportion(c)) ' ' FileNamesGroup1{file_mixing_combination(m,1)} ' + ' FileNamesGroup2{file_mixing_combination(m,2)}]
% % ALLEEG7 = pop_saveset( ALLEEG7,'filename',denoised_dataset_name, 'savemode','onefile'); % optional DENOISED file save
% 
% % % estimate ASR denoising quality
% ground_truth_matrix=ALLEEG1(file_mixing_combination(m,1)).data;
% ASR_denoised_matrix=ALLEEG7.data;
% 
% ASR_Rsquared(m,:)=variance_explained(ground_truth_matrix,ASR_denoised_matrix);
% ASR_RRMSE(m,:)=relative_RMSE(ground_truth_matrix, ASR_denoised_matrix);
% ASR_SNR(m,:) = sig_to_noise(ground_truth_matrix,ASR_denoised_matrix);
% 
% % vis_artifacts(ALLEEG7,ALLEEG1(file_mixing_combination(m,1)))
% end
% 
% 
% 
% 
% % RUN ICA (for IClabel and MARA)
% 
% for m=1:length(file_mixing_combination)
% [~, artifact_type, ~] = fileparts(DirGroup2(file_mixing_combination(m,2)).folder)
% 
% if strcmp(artifact_type, 'NOISE') | strcmp(artifact_type, 'NOISE EOG EMG') % additionally use bad channel rejection for NOISE condition prior to ICA
% 
% ALLEEG8(m) = pop_clean_rawdata(ALLEEG3(m), 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',4,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
% 
% 
% else % no bad channel rejection
% 
% ALLEEG8(m)=ALLEEG3(m);
% end
% 
% tic
% % run Infomax ICA Extended
% ALLEEG8(m) = pop_runica(ALLEEG8(m), 'verbose', 'off','extended',1,'interupt','off'); % run Infomax ICA Extended
% 
% time_ICA(m,:) = toc;
% end
% 
% 
% 
% 
% % RUN IClabel
% for m=1:length(file_mixing_combination)
% 
% ALLEEG9=ALLEEG8(m);
% 
% tic
% ALLEEG9 = pop_iclabel(ALLEEG9, 'default'); % run IClabel
% [maxprob, IC_class]=max(ALLEEG9.etc.ic_classification.ICLabel.classifications');
% 
% [~, artifact_type, ~] = fileparts(DirGroup2(file_mixing_combination(m,2)).folder);
% 
% if strcmp(artifact_type, 'NOISE') | strcmp(artifact_type, 'NOISE EOG EMG') % remove all components except for "brain" or "other"
% 
% keep_ICbrain=find(IC_class==1); % brain ICs
% keep_ICother=find(IC_class==7); % other ICs
% try
% ALLEEG9 = pop_subcomp( ALLEEG9, [keep_ICbrain keep_ICother], 0, 1); %keep brain + other ICs
% catch 
% end
% 
% 
% else % remove only eye or muscle artifacts
% %% EYE and/or MUSCLE ARTIFACTS
% 
% remove_ICeye=find(IC_class==3);
% remove_ICmuscle=find(IC_class==2);
% 
% ALLEEG9 = pop_subcomp( ALLEEG9, [remove_ICeye remove_ICmuscle], 0, 0); %remove eye + muscle ICs
% 
% end
%  %% interpolate bad channels, if any
% ALLEEG9 = pop_interp(ALLEEG9,  ALLEEG1(1).chanlocs, 'spherical'); 
% 
% time_IClabel(m,:) = toc+ time_ICA(m,:);
% 
% denoised_dataset_name=['IClabel ' ' SNR=' num2str(signal_to_noise_linear(n)) ' contamination=' num2str(contaminated_signal_proportion(c)) ' ' FileNamesGroup1{file_mixing_combination(m,1)} ' + ' FileNamesGroup2{file_mixing_combination(m,2)}]
% % ALLEEG9 = pop_saveset( ALLEEG9,'filename',denoised_dataset_name, 'savemode','onefile'); % optional DENOISED file save
% 
% % % estimate IClabel denoising quality
% ground_truth_matrix=ALLEEG1(file_mixing_combination(m,1)).data;
% IClabel_denoised_matrix=ALLEEG9.data;
% 
% IClabel_Rsquared(m,:)=variance_explained(ground_truth_matrix,IClabel_denoised_matrix);
% IClabel_RRMSE(m,:)=relative_RMSE(ground_truth_matrix, IClabel_denoised_matrix);
% IClabel_SNR(m,:) = sig_to_noise(ground_truth_matrix,IClabel_denoised_matrix);
% end
% 
% 
% % % RUN MARA
% for m=1:length(file_mixing_combination)
% 
% 
% ALLEEG10=ALLEEG8(m);
% 
% tic
% 
% [artifact_ICs, info] = MARA(ALLEEG10); % run MARA
% 
%  %% reject the ICs that MARA flagged as artifact
% ALLEEG10 = pop_subcomp( ALLEEG10, artifact_ICs, 0); %remove flagged components
% 
%  %% interpolate bad channels, if any
% ALLEEG10   = pop_interp(ALLEEG10,  ALLEEG1(1).chanlocs, 'spherical'); 
% 
% time_MARA(m,:) = toc + time_ICA(m,:);
% 
% 
% denoised_dataset_name=['MARA ' ' SNR=' num2str(signal_to_noise_linear(n)) ' contamination=' num2str(contaminated_signal_proportion(c)) ' ' FileNamesGroup1{file_mixing_combination(m,1)} ' + ' FileNamesGroup2{file_mixing_combination(m,2)}]
% % ALLEEG10 = pop_saveset( ALLEEG10,'filename',denoised_dataset_name, 'savemode','onefile'); % optional DENOISED file save
% 
% % % estimate MARA denoising quality
% ground_truth_matrix=ALLEEG1(file_mixing_combination(m,1)).data;
% MARA_denoised_matrix=ALLEEG10.data;
% 
% MARA_Rsquared(m,:)=variance_explained(ground_truth_matrix,MARA_denoised_matrix);
% MARA_RRMSE(m,:)=relative_RMSE(ground_truth_matrix, MARA_denoised_matrix);
% MARA_SNR(m,:) = sig_to_noise(ground_truth_matrix,MARA_denoised_matrix);
% end


 % RUN GEDAI

load('Freesurfer_MEG306_leadfield_4_GEDAI');
MEG_gram_matrix=MEG_leadfield_4_GEDAI.Gain*MEG_leadfield_4_GEDAI.Gain';
MEG_gram_matrix = MEG_gram_matrix(mag_chans, mag_chans);


%%
for m=1:length(file_mixing_combination)

    
m 
tic

[EEGout] = GEDAI(ALLEEG3(m), 'auto+', 12, 0.5, MEG_gram_matrix, true, false, inf, 'meg');
% EEGout = GEDAI(ALLEEG3(m),  'auto'); % run GEDAI

 % EEGout = GEDAI_optimizer_topPCs(ALLEEG3(m), 'auto', 12, 0.5, 'precomputed', true, false, inf, false, 5);
% EEGout = GEDAI_optimizer_noise(ALLEEG3(m));

% [ALLEEG11(m,:)] = GEDAI_bayesopt(ALLEEG3(m), 'MaxEvaluations', 10, 'Visualization', false, 'VisualizeResult', false);
% [ALLEEG11(m,:)] =GEDAI_Bayesfield(ALLEEG3(m), 3, 3, false, 'trace');


% ALLEEG11 = GEDAI(ALLEEG3(m),'auto', epoch_size,ref_matrix,true,false); % run GEDAI
toc
time_GEDAI(m,:) = toc;

denoised_dataset_name=['GEDAI ' ' SNR=' num2str(signal_to_noise_linear(n)) ' contamination=' num2str(contaminated_signal_proportion(c)) ' ' FileNamesGroup1{file_mixing_combination(m,1)} ' + ' FileNamesGroup2{file_mixing_combination(m,2)}]
% EEGtemp = pop_saveset( EEGout,'filename',denoised_dataset_name, 'savemode','onefile'); % optional DENOISED file save

% % estimate GEDAI denoising quality
ground_truth_matrix=ALLEEG1(file_mixing_combination(m,1)).data;
GEDAI_denoised_matrix=EEGout.data;

GEDAI_Rsquared(m,:)=variance_explained(ground_truth_matrix,GEDAI_denoised_matrix);
GEDAI_RRMSE(m,:)=relative_RMSE(ground_truth_matrix, GEDAI_denoised_matrix);
GEDAI_SNR(m,:) = sig_to_noise(ground_truth_matrix,GEDAI_denoised_matrix);

% vis_artifacts(ALLEEG11,ALLEEG1(file_mixing_combination(m,1)));
% 
% if GEDAI_Rsquared(m,:) <0.5
%       mixed_dataset_name=[' SNR=' num2str(signal_to_noise_linear(n)) ' contamination=' num2str(contaminated_signal_proportion(c)) ' ' FileNamesGroup1{file_mixing_combination(m,1)} ' + ' FileNamesGroup2{file_mixing_combination(m,2)}]
% 
%     ALLEEG3(m) = pop_saveset( ALLEEG3(m),'filename',mixed_dataset_name, 'savemode','onefile'); % optional MIXED file save
% end





end

clear ALLEEG3
clear ALLEEG11


%%%%%%%   SUMMARY RESULTS  %%%%%%

% define clean EEG files
for m=1:length(file_mixing_combination)
clean_EEG_file_name=DirGroup1(file_mixing_combination(m,1)).name;
clean_EEG_file{m,:}=clean_EEG_file_name;
end

% define artifact files
for m=1:length(file_mixing_combination)
artifact_file_name=DirGroup2(file_mixing_combination(m,2)).name;
artifact_file{m,:}=artifact_file_name;
end

% define temporal contamination variable
contamination=repmat(num2str(contaminated_signal_proportion(c)), size(GEDAI_Rsquared));

% define signal_to_noise variable
signal_to_noise=repmat(num2str(signal_to_noise_in_db(n)), size(GEDAI_Rsquared));

% define artifact variable
for m=1:length(file_mixing_combination)
[~, artifact_name, ~] = fileparts(DirGroup2(file_mixing_combination(m,2)).folder);
artifact{m,:}=artifact_name;
end

% %  Variance Explained table (higher is better)
TABLE_Rsquared=table(clean_EEG_file,artifact_file,artifact,contamination,signal_to_noise, GEDAI_Rsquared,'VariableNames',["clean_EEG_file","artifact_file","artifact","temporal_contamination","signal_to_noise", "GEDAI"]);
TABLE_Rsquared = stack(TABLE_Rsquared,6,'NewDataVariableName','Rsquared','IndexVariableName','Algorithm');
TABLE_Rsquared_concatenated= [TABLE_Rsquared_concatenated; TABLE_Rsquared];
TABLE_Correlation_concatenated=TABLE_Rsquared_concatenated;
TABLE_Correlation_concatenated.Correlation=TABLE_Rsquared_concatenated.Rsquared.^0.5;

% %  Signal to Noise ratio (higher is better)
TABLE_SNR=table(clean_EEG_file,artifact_file,artifact,contamination,signal_to_noise, GEDAI_SNR,'VariableNames',["clean_EEG_file","artifact_file","artifact","temporal_contamination","signal_to_noise","GEDAI"]);
TABLE_SNR = stack(TABLE_SNR,6,'NewDataVariableName','SNR', 'IndexVariableName','Algorithm');
TABLE_SNR_concatenated= [TABLE_SNR_concatenated; TABLE_SNR];

% % % RRMSE table (lower is better)
TABLE_RRMSE=table(clean_EEG_file,artifact_file,artifact,contamination,signal_to_noise,GEDAI_RRMSE,'VariableNames',["clean_EEG_file","artifact_file","artifact","temporal_contamination","signal_to_noise","GEDAI"]);
TABLE_RRMSE = stack(TABLE_RRMSE,6,'NewDataVariableName','RRMSE', 'IndexVariableName','Algorithm');
TABLE_RRMSE_concatenated= [TABLE_RRMSE_concatenated; TABLE_RRMSE];

% % % Computation Time table (lower is better)
TABLE_time=table(clean_EEG_file,artifact_file,artifact,contamination,signal_to_noise,time_GEDAI,'VariableNames',["clean_EEG_file","artifact_file","artifact","temporal_contamination","signal_to_noise","GEDAI"]);
TABLE_time = stack(TABLE_time,6,'NewDataVariableName','time', 'IndexVariableName','Algorithm');
TABLE_time_concatenated= [TABLE_time_concatenated; TABLE_time];

    end

end

%% PLOT across different temporal contamination ratios 
rng('default')

% % % Correlation
grpandplot(TABLE_Correlation_concatenated,"Correlation",yTitle='Correlation AFTER denoising (R)',xFactor="temporal_contamination",cFactor="Algorithm",cOrder=string(unique(TABLE_RRMSE_concatenated.Algorithm)),...
              xOrder=string(categorical(contaminated_signal_proportion)),boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=true);
legend('Location','best')
set(gca,'YGrid','on', 'GridLineStyle', '-')
title(['Artifact type: ALL'  '            Signal to Noise Ratio : ALL'])
xlabel('Temporal Contamination Level (%)');
ylim([0 1.0]);


% % SNR
[tile, statsTable_by_contamination]=grpandplot2(TABLE_SNR_concatenated,"SNR",yTitle='SNR AFTER denoising (dB)',xFactor="temporal_contamination",cFactor="Algorithm",cOrder=string(unique(TABLE_SNR_concatenated.Algorithm)),...
               xOrder=string(categorical(contaminated_signal_proportion)),boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.15,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
legend('Location','best')
set(gca,'YGrid','on', 'GridLineStyle', '-')
yticklabels;
title(['Artifact type: ALL'  '            Signal to Noise Ratio: ALL'])
xlabel('Temporal Contamination Level (%)')



% % % RRMSE
grpandplot(TABLE_RRMSE_concatenated,"RRMSE",yTitle='RRMSE AFTER denoising',xFactor="temporal_contamination",cFactor="Algorithm",cOrder=string(unique(TABLE_RRMSE_concatenated.Algorithm)),...
               xOrder=string(categorical(contaminated_signal_proportion)),boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
legend('Location','best')
set(gca,'YGrid','on', 'GridLineStyle', '-')
title(['Artifact type: ALL'  '            Signal to Noise Ratio: ALL'])
xlabel('Temporal Contamination Level (%)');



% % % Computation time
% grpandplot(TABLE_time_concatenated,"time",yTitle='Computational time (seconds)',xFactor="temporal_contamination",cFactor="Algorithm",cOrder=string(unique(TABLE_time_concatenated.Algorithm)),...
%                boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
% legend('Location','best')
% set(gca,'YGrid','on', 'GridLineStyle', '-')
% yticklabels;
% title(['Artifact type: ALL'  '            Signal to Noise Ratio: ALL'])
% xlabel('Temporal Contamination Level (%)');



%% PLOT across different signal-to-noise ratios 
% rng('default')
% 
% % % Correlation
TABLE_Correlation_concatenated=TABLE_Rsquared_concatenated;
TABLE_Correlation_concatenated.Correlation=TABLE_Rsquared_concatenated.Rsquared.^0.5;
grpandplot(TABLE_Correlation_concatenated,"Correlation",yTitle='Correlation AFTER denoising (R)',xFactor="signal_to_noise",cFactor="Algorithm",cOrder=string(unique(TABLE_RRMSE_concatenated.Algorithm)),...
              xOrder=string(categorical(signal_to_noise_in_db)),boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
legend('Location','best')
set(gca,'YGrid','on', 'GridLineStyle', '-')
title(['Artifact type: ALL'  '            Temporal contamination level: ALL' ])
xlabel('SNR BEFORE denoising (dB)');
ylim([0 1.0]);

% % % SNR
[tile, statsTable_by_SNR]=grpandplot2(TABLE_SNR_concatenated,"SNR",yTitle='SNR AFTER denoising (dB)',xFactor="signal_to_noise",cFactor="Algorithm",cOrder=string(unique(TABLE_SNR_concatenated.Algorithm)),...
               xOrder=string(categorical(signal_to_noise_in_db)),boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
legend('Location','best')
set(gca,'YGrid','on', 'GridLineStyle', '-')
yticklabels;
title(['Artifact type: ALL'  '            Temporal contamination level: ALL'])
xlabel('SNR BEFORE denoising (dB)');
ylim([-15 20]);


% % RRMSE
% grpandplot(TABLE_RRMSE_concatenated,"RRMSE",yTitle='RRMSE AFTER denoising',xFactor="signal_to_noise",cFactor="Algorithm",cOrder=string(unique(TABLE_RRMSE_concatenated.Algorithm)),...
%                xOrder=string(categorical(signal_to_noise_in_db)),boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
% legend('Location','best')
% set(gca,'YGrid','on', 'GridLineStyle', '-')
% title(['Artifact type: ALL'  '            Temporal contamination level: ALL'])
% xlabel('SNR BEFORE denoising (dB)');



% % Computation time
% grpandplot(TABLE_time_concatenated,"time",yTitle='Computational time (seconds)',xFactor="signal_to_noise",cFactor="Algorithm",cOrder=string(unique(TABLE_time_concatenated.Algorithm)),...
%                boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
% legend('Location','best')
% set(gca,'YGrid','on', 'GridLineStyle', '-')
% yticklabels;
% title(['Artifact type: ALL'  '            Temporal contamination level: ALL'])
% xlabel('SNR BEFORE denoising (dB)');


%% PLOT across different artifact types 

% % % Correlation (higher is better)
% grpandplot(TABLE_Correlation_concatenated,"Correlation",yTitle='Correlation AFTER denoising',xFactor="artifact",cFactor="Algorithm",cOrder=string(unique(TABLE_Correlation_concatenated.Algorithm)),...
%                boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
% legend('Location','best')
% set(gca,'YGrid','on', 'GridLineStyle', '-')
% title(['Signal to noise level: ALL'  '            Temporal contamination level: ALL'])
% xlabel('Artifact type')
% ylim([0 1.0]);

% % SNR (higher is better)
[tile, statsTable_by_artifact]=grpandplot2(TABLE_SNR_concatenated,"SNR",yTitle='SNR AFTER denoising (dB)',xFactor="artifact",cFactor="Algorithm",cOrder=string(unique(TABLE_SNR_concatenated.Algorithm)),...
               boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
legend('Location','best')
set(gca,'YGrid','on', 'GridLineStyle', '-')
yticklabels;
title(['Signal to noise level: ALL'  '            Temporal contamination level: ALL'])
xlabel('Artifact type')
ylim([0 21]);


% % RRMSE (lower is better)
% grpandplot(TABLE_RRMSE_concatenated,"RRMSE",yTitle='RRMSE AFTER denoising',xFactor="artifact",cFactor="Algorithm",cOrder=string(unique(TABLE_RRMSE_concatenated.Algorithm)),...
%               xOrder=["EOG" "EMG" "NOISE" "EOG EMG" "NOISE EOG EMG"]' , boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
% legend('Location','best')
% set(gca,'YGrid','on', 'GridLineStyle', '-')
% title(['Signal to noise level: ALL'  '            Temporal contamination level: ALL'])
% xlabel('Artifact type')


% % Computation time
% [tile, statsTable_by_time]=grpandplot2(TABLE_time_concatenated,"time",yTitle='Computational time (seconds)',xFactor="artifact",cFactor="Algorithm",cOrder=string(unique(TABLE_time_concatenated.Algorithm)),...
%               xOrder=["EOG" "EMG" "NOISE" "NOISE EOG EMG"]' , boxAlpha=0.15,showBox=true, showVln=false,vlnAlpha=0.3,showOutlier=false, showLegend=true, showXLine=false,pntSize=22,pntEdgeC='w',pntOnTop=false);
% legend('Location','best')
% set(gca,'YGrid','on', 'GridLineStyle', '-')
% yticklabels;
% title(['Signal to noise level: ALL'  '            Temporal contamination level: ALL'])
% xlabel('Artifact type')



% global parametric STATISTICAL TESTS FOR VISUALISATION ONLY (across all signal_to_noise levels AND artifact types)
%% since reconstruction accuries are NOT normally distributed

% % Correlation
% [p,tbl,stats_Correlation] = anova1(TABLE_Correlation_concatenated.Correlation, TABLE_Correlation_concatenated.Algorithm, 'off');
% figure;multcompare(stats_Correlation, 'CriticalValueType', 'bonferroni');
% title('Correlation: Group Means and 95% Confidence Intervals');
% % 
% % % SNR
[p,tbl,stats_SNR] = anova1(TABLE_SNR_concatenated.SNR, TABLE_SNR_concatenated.Algorithm, 'off');
figure;multcompare(stats_SNR, 'CriticalValueType', 'bonferroni');
title('SNR: Group Means and 95% Confidence Intervals');

%RRMSE
% [p,tbl,stats_RRMSE] = anova1(TABLE_RRMSE_concatenated.RRMSE, TABLE_RRMSE_concatenated.Algorithm, 'off');
% figure;multcompare(stats_RRMSE, 'CriticalValueType', 'bonferroni');
% title('RRMSE: Group Means and 95% Confidence Intervals');

% % Computation time
% [p,tbl,stats_time] = anova1(TABLE_time_concatenated.time, TABLE_time_concatenated.Algorithm, 'off');
% figure;multcompare(stats_time, 'CriticalValueType', 'bonferroni');
% title('Time (seconds): Group Means and 95% Confidence Intervals');



% global non-parametric STATISTICAL TESTS (across all signal_to_noise levels AND artifact types)
%% Friedman test for non-normally distributed, paired reconstruction accuracies 

% TABLE_Correlation_paired = unstack(TABLE_Correlation_concatenated, 'Correlation','Algorithm', 'GroupingVariables', (["clean_EEG_file", "artifact_file","artifact", "temporal_contamination", "signal_to_noise"])); %unstack
% % TABLE_Correlation_paired = TABLE_Correlation_paired(TABLE_Correlation_paired.artifact =="EOG", :)
% ARRAY_Correlation_paired = table2array(TABLE_Correlation_paired(:,7:10)); % exclude raw condition from comparison
% [friedman_p_Correlation,friedman_tbl_Correlation,friedman_stats_Correlation] = friedman(ARRAY_Correlation_paired, 1,'off');
% figure;[friedman_multcompare_Correlation]=multcompare(friedman_stats_Correlation, 'CriticalValueType', 'bonferroni');
% title('Correlation: Friedman test (Bonferroni-corrected)');
% 
% 
% TABLE_SNR_paired = unstack(TABLE_SNR_concatenated, 'SNR','Algorithm', 'GroupingVariables', (["clean_EEG_file", "artifact_file","artifact", "temporal_contamination", "signal_to_noise"])); %unstack
% % TABLE_SNR_paired = TABLE_SNR_paired(TABLE_SNR_paired.artifact =="EOG", :)
% ARRAY_SNR_paired = table2array(TABLE_SNR_paired(:,7:10));
% [friedman_p_SNR,friedman_tbl_SNR,friedman_stats_SNR] = friedman(ARRAY_SNR_paired, 1,'off');
% figure;[friedman_multcompare_SNR]=multcompare(friedman_stats_SNR, 'CriticalValueType', 'bonferroni');
% title('SNR: Friedman test (Bonferroni-corrected)');
% 
% TABLE_RRMSE_paired = unstack(TABLE_RRMSE_concatenated, 'RRMSE','Algorithm', 'GroupingVariables', (["clean_EEG_file", "artifact_file","artifact", "temporal_contamination", "signal_to_noise"])); %unstack
% ARRAY_RRMSE_paired = table2array(TABLE_RRMSE_paired(:,7:10));
% [friedman_p_RRMSE,friedman_tbl_RRMSE,friedman_stats_RRMSE] = friedman(ARRAY_RRMSE_paired, 1,'off');
% figure;[friedman_multcompare_RRMSE]=multcompare(friedman_stats_RRMSE, 'CriticalValueType', 'bonferroni');
% title('RRMSE: Friedman test (Bonferroni-corrected)');
% 
% 
% TABLE_time_paired = unstack(TABLE_time_concatenated, 'time','Algorithm', 'GroupingVariables', (["clean_EEG_file", "artifact_file","artifact", "temporal_contamination", "signal_to_noise"])); %unstack
% ARRAY_time_paired = table2array(TABLE_time_paired(:,7:10));
% [friedman_p_time,friedman_tbl_time,friedman_stats_time] = friedman(ARRAY_time_paired, 1,'off');
% figure;[friedman_multcompare_time]=multcompare(friedman_stats_time, 'CriticalValueType', 'bonferroni');
% title('Time: Friedman test (Bonferroni-corrected)');

%% Global Effect size calculation (between a pair of algorithms)
% algorithm_1="GEDAI";
% algorithm_2="MARA";
% 
% %Effect size: Correlation (higher is better)
% global_x_Correlation=TABLE_Correlation_concatenated.Correlation(TABLE_Correlation_concatenated.Algorithm==algorithm_1);
% global_y_Correlation=TABLE_Correlation_concatenated.Correlation(TABLE_Correlation_concatenated.Algorithm==algorithm_2);
% % effect_size_Correlation=meanEffectSize(x_Correlation,y_Correlation,Paired=true,Effect="cohen") % suitable only for normally distributed data
% [global_Correlation_signed_rank_p,h,global_Correlation_signed_rank_stats] = signrank(global_x_Correlation, global_y_Correlation);
% global_effect_size_Correlation=global_Correlation_signed_rank_stats.zval/sqrt(length(global_x_Correlation))
% 
% 
% %Effect size: SNR (higher is better)
% global_x_SNR=TABLE_SNR_concatenated.SNR(TABLE_SNR_concatenated.Algorithm==algorithm_1);
% global_y_SNR=TABLE_SNR_concatenated.SNR(TABLE_SNR_concatenated.Algorithm==algorithm_2);
% 
% [global_SNR_signed_rank_p,h,global_SNR_signed_rank_stats] = signrank(global_x_SNR, global_y_SNR);
% global_effect_size_SNR=global_SNR_signed_rank_stats.zval/sqrt(length(global_x_SNR))

% %Effect size: RRMSE (lower is better)
% global_x_RRMSE=TABLE_RRMSE_concatenated.RRMSE(TABLE_RRMSE_concatenated.Algorithm==algorithm_1);
% global_y_RRMSE=TABLE_RRMSE_concatenated.RRMSE(TABLE_RRMSE_concatenated.Algorithm==algorithm_2);
% 
% [global_RRMSE_signed_rank_p,h,global_RRMSE_signed_rank_stats] = signrank(global_x_RRMSE, global_y_RRMSE);
% global_effect_size_RRMSE=global_RRMSE_signed_rank_stats.zval/sqrt(length(global_x_RRMSE))







%% denoising quality functions

function SNR = sig_to_noise (ground_truth_matrix, denoised_matrix)
original_signal_power =var(ground_truth_matrix (:)); % mean power of the ground truth (brain) signal over electrodes
residual_noise_power = var(denoised_matrix(:)-ground_truth_matrix(:));  % mean power of the residual noise after processing (denoised - ground truth)
SNR = 10 * log10(original_signal_power/residual_noise_power);
end


function Rsquared=variance_explained(ground_truth_matrix,denoised_matrix)
R=corrcoef(ground_truth_matrix, denoised_matrix);
Rsquared=R(2)*R(2);
end


function RRMSE = relative_RMSE(ground_truth_matrix, denoised_matrix)
    squared_error = (ground_truth_matrix - denoised_matrix).^2;    % Calculate the squared error between all elements
    RMSE = sqrt(mean(squared_error(:)));   % Calculate the Root Mean Square Error (RMSE) across all elements
    rms_ground_truth = rms(ground_truth_matrix(:)); % Calculate the Root Mean Square (RMS) of all elements in the ground truth matrix
    RRMSE = RMSE / rms_ground_truth;  % Calculate the Relative Root-Mean-Square Error
end


function [modifiedMatrix, keptColumnIndices, zeroedColumnIndices, actualBlockSizes, actual_k_numBlocks] = retainExactPercentageRandomEEGBlocks(dataMatrix, percentageToKeep, min_blockSize, max_blockSize)
%retainExactPercentageRandomEEGBlocks Retains NON-OVERLAPPING blocks to meet an exact target percentage.
%
%   [modifiedMatrix, keptColumnIndices, zeroedColumnIndices, actualBlockSizes, actual_k_numBlocks] =
%   retainExactPercentageRandomEEGBlocks(dataMatrix, percentageToKeep, min_blockSize, max_blockSize)
%   Selects non-overlapping blocks of columns. Most blocks have random sizes
%   between min_blockSize and max_blockSize (inclusive). Enough blocks are
%   selected such that the total number of columns kept exactly matches the
%   target percentage (after rounding the target to the nearest integer number
%   of columns). A final block smaller than min_blockSize might be added if
%   needed to meet the exact target count. Data within these selected blocks
%   is retained; all other columns are set to zero.
%
%   Args:
%       dataMatrix:       The input 2D matrix.
%       percentageToKeep: The target percentage (0-100) of total columns to keep.
%       min_blockSize:    The minimum number of consecutive columns for most blocks (positive int).
%       max_blockSize:    The maximum number of consecutive columns in each block (positive int >= min_blockSize).
%
%   Returns:
%       modifiedMatrix:    Matrix with only retained blocks non-zero.
%       keptColumnIndices: Column vector of unique indices of retained columns (sorted).
%                          Length is exactly targetTotalColumns (rounded from percentage).
%       zeroedColumnIndices: Column vector of unique indices of zeroed columns (sorted).
%       actualBlockSizes:  Row vector containing the actual size used for each selected block.
%                          The last block size might be < min_blockSize.
%       actual_k_numBlocks: The actual number of blocks selected.
%
%   Raises:
%       Error if inputs invalid, or if the target percentage cannot be met
%       (e.g., target > 0 but numCols is too small).

% --- Input Validation ---
if ~ismatrix(dataMatrix) || isempty(dataMatrix)
    error('Input data must be a non-empty 2D matrix.');
end
if ~(isscalar(percentageToKeep) && percentageToKeep >= 0 && percentageToKeep <= 100)
    error('percentageToKeep must be a scalar between 0 and 100.');
end
if ~(isscalar(min_blockSize) && min_blockSize > 0 && floor(min_blockSize) == min_blockSize)
    error('Minimum block size min_blockSize must be a positive integer scalar.');
end
if ~(isscalar(max_blockSize) && max_blockSize > 0 && floor(max_blockSize) == max_blockSize)
    error('Maximum block size max_blockSize must be a positive integer scalar.');
end
if min_blockSize > max_blockSize
    error('min_blockSize (%d) cannot be greater than max_blockSize (%d).', min_blockSize, max_blockSize);
end

[rows, numCols] = size(dataMatrix);

% Initialize outputs
keptColumnIndices = [];
zeroedColumnIndices = (1:numCols)'; % Default: all zeroed
actualBlockSizes = [];
actual_k_numBlocks = 0;
modifiedMatrix = zeros(rows, numCols, 'like', dataMatrix); % Start with zeros

% --- Handle percentageToKeep = 0 ---
if percentageToKeep == 0
    return; % Outputs are already initialized correctly
end

% --- Calculate Exact Target Columns ---
targetTotalColumns = round(numCols * percentageToKeep / 100);

% If target is 0 after rounding (but percentage > 0), we keep nothing.
if targetTotalColumns == 0
    warning('Target percentage %.2f%% results in 0 columns to keep after rounding for %d total columns.', percentageToKeep, numCols);
    return; % Return all zeros
end

% Check if target is even possible
if targetTotalColumns > numCols
    error('Target percentage %.2f%% results in %d columns, which is more than the available %d columns.', percentageToKeep, targetTotalColumns, numCols);
end


% --- Iteratively Generate Block Sizes ---
currentTotalColumns = 0;
while true
    remainingTarget = targetTotalColumns - currentTotalColumns;

    % Stop if target is met
    if remainingTarget <= 0
        break;
    end

    % --- Determine size for the next block ---
    % Max possible size for this block without exceeding target OR max_blockSize
    maxPossibleNextSize = min(max_blockSize, remainingTarget);

    % Min possible size for this block (usually min_blockSize, but capped by feasibility)
    % If the remaining target is less than min_blockSize, we might need a smaller final block later.
    minPossibleNextSize = min(min_blockSize, maxPossibleNextSize); % Cannot be larger than maxPossibleNextSize

    % Check if we can add *any* block (even size 1) without exceeding numCols
    if currentTotalColumns + 1 > numCols
        warning('Cannot add more columns; stopping short of the target %d. Current total: %d.', targetTotalColumns, currentTotalColumns);
        break; % Cannot fit even one more column
    end

    % Check if the smallest possible block (minPossibleNextSize) fits within numCols
    if currentTotalColumns + minPossibleNextSize > numCols
         warning('Cannot fit another block respecting constraints without exceeding total columns; stopping short of the target %d. Current total: %d.', targetTotalColumns, currentTotalColumns);
         break; % Cannot fit the minimum required block size
    end

    % If the largest possible block size respecting constraints is less than
    % the minimum required block size, it means we must add a final smaller block.
    if maxPossibleNextSize < min_blockSize
        % Add the remaining amount as the last block
        nextBlockSize = remainingTarget;
        % Ensure this final block doesn't exceed available columns
        if currentTotalColumns + nextBlockSize > numCols
             warning('Cannot fit the final remainder block; stopping short of the target %d. Current total: %d.', targetTotalColumns, currentTotalColumns);
             break;
        end
    else
        % Generate a random block size within the allowed range
        % Range: [minPossibleNextSize, maxPossibleNextSize]
        nextBlockSize = randi([minPossibleNextSize, maxPossibleNextSize], 1, 1);
    end

    % Add the block
    actualBlockSizes(end+1) = nextBlockSize;
    currentTotalColumns = currentTotalColumns + nextBlockSize;
    actual_k_numBlocks = actual_k_numBlocks + 1;

    % Break if we added the exact remainder as the last block
    if nextBlockSize == remainingTarget
        break;
    end
end % End of while loop for block generation

% --- Final check on achieved total ---
if currentTotalColumns ~= targetTotalColumns
    warning('Could not achieve exact target of %d columns. Achieved %d columns due to space constraints.', targetTotalColumns, currentTotalColumns);
    % Proceed with the columns accumulated so far.
    % Update targetTotalColumns to reflect reality for gap calculation
    targetTotalColumns = currentTotalColumns;
end

% If no blocks were selected (e.g., target was > 0 but very small, and min_blockSize was large)
if actual_k_numBlocks == 0 && targetTotalColumns > 0
    warning('No blocks were selected despite a non-zero target. Target percentage might be too low or min_blockSize too large relative to total columns.');
    return; % Return all zeros
elseif actual_k_numBlocks == 0 && targetTotalColumns == 0
    return; % Correctly returning all zeros
end

totalColumnsNeeded = targetTotalColumns; % Use the possibly adjusted target

% --- Core Logic (Non-Overlapping Selection using Gap Method) ---
% 1. Calculate available space for gaps between/around blocks
availableGapSpace = numCols - totalColumnsNeeded;
if availableGapSpace < 0
    error('Internal logic error: Available gap space (%d) is negative. totalNeeded=%d, numCols=%d', availableGapSpace, totalColumnsNeeded, numCols);
end

% 2. Randomly determine the sizes of the k+1 gaps (g0, g1, ..., gk)
k = actual_k_numBlocks; % Use the actual number of blocks selected
if k == 0 % Should be handled above, but safety check
     gapLengths = availableGapSpace; % Only one gap
else
    if availableGapSpace + k > 0 % Ensure randperm has a positive N
        gapDividerIndices = sort(randperm(availableGapSpace + k, k));
        gapLengths = diff([0, gapDividerIndices, availableGapSpace + k + 1]) - 1;
    else % No gap space (means totalColumnsNeeded == numCols)
        gapLengths = zeros(1, k + 1); % All gaps are zero length
    end
end

% 3. Calculate the start/end positions and collect indices for each block
keptColumnIndices = zeros(1, totalColumnsNeeded); % Preallocate
currentIndex = 1; % Index for filling keptColumnIndices
currentColumn = 1; % Starting column position in the matrix

for i = 1:k % Iterate through the actual number of blocks
    % Add the gap *before* the current block (gap g_{i-1})
    currentColumn = currentColumn + gapLengths(i);

    % Determine start and end columns for block i using its specific size
    startCol = currentColumn;
    blockSize = actualBlockSizes(i);
    endCol = currentColumn + blockSize - 1;

    % Boundary check (should not happen with correct gap calculation)
    if endCol > numCols
        error('Internal logic error: Calculated end column %d exceeds matrix dimension %d for block %d.', endCol, numCols, i);
    end
    if blockSize <= 0
         error('Internal logic error: Calculated block size %d is non-positive for block %d.', blockSize, i);
    end

    % Add indices for this block to the list
    blockIndices = startCol:endCol;
    numIndicesInBlock = length(blockIndices); % Should equal blockSize

    % Ensure preallocation size is sufficient (dynamic resizing if error occurs)
    if currentIndex + numIndicesInBlock - 1 > length(keptColumnIndices)
        warning('Resizing keptColumnIndices array; potential performance issue.');
        keptColumnIndices = [keptColumnIndices, zeros(1, numIndicesInBlock)]; % Append space
    end

    keptColumnIndices(currentIndex : currentIndex + numIndicesInBlock - 1) = blockIndices;
    currentIndex = currentIndex + numIndicesInBlock;

    % Advance column position *past* the current block to prepare for the next gap
    currentColumn = endCol + 1;
end

% Trim any unused preallocated space (if totalColumnsNeeded was adjusted downwards)
if length(keptColumnIndices) > totalColumnsNeeded
    keptColumnIndices = keptColumnIndices(1:totalColumnsNeeded);
end

% Sort the collected indices
keptColumnIndices = sort(keptColumnIndices(:)); % Ensure column vector and sorted

% Final Sanity check
if length(keptColumnIndices) ~= totalColumnsNeeded
     error('Fatal internal error: Final number of kept indices (%d) does not match target (%d).', length(keptColumnIndices), totalColumnsNeeded);
end

% 4. Determine column indices to zero out using set difference
allCols = (1:numCols)';
zeroedColumnIndices = setdiff(allCols, keptColumnIndices);

% 5. Create Modified Matrix
if ~isempty(keptColumnIndices)
     modifiedMatrix(:, keptColumnIndices) = dataMatrix(:, keptColumnIndices); % Copy retained data
end

end % End of function




