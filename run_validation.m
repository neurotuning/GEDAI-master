% Set MATLAB path to EEGLAB
eeglab_path = 'C:\Users\Utilisateur\Documents\MATLAB\eeglab-developp';
addpath(eeglab_path);
try
    eeglab nogui;
catch e
    disp('EEGLAB error:');
    disp(e.message);
end

data_path = 'C:\Users\Utilisateur\Documents\MATLAB\eeglab-developp\plugins\GEDAI-master\example data';
try
    EEG = pop_loadset('filename', 'empirical_NOISE_EOG_EMG.set', 'filepath', data_path);
    disp('Running GEDAI with ''auto'' threshold...');
    [EEG_clean, ~, broadband_sensai] = GEDAI(EEG, 'auto', 12, 0.5, 'precomputed', false, false, [], [], true);
    
    % Save all open figures
    figs = findobj('Type', 'figure');
    if isempty(figs)
        disp('No figures generated.');
    end
    for i = 1:numel(figs)
        % Ensure path exists
        out_dir = 'C:\Users\Utilisateur\.gemini\antigravity\brain\6d4fc83d-a89f-47c6-a7b2-8fa6b6d62e54\artifacts';
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end
        file_name = fullfile(out_dir, sprintf('validation_plot_%d.png', i));
        saveas(figs(i), file_name);
        disp(['Saved ', file_name]);
    end
    disp('Validation complete.');
catch e
    disp('Error during processing:');
    disp(e.message);
    for k=1:length(e.stack)
        disp([e.stack(k).name, ' line ', num2str(e.stack(k).line)]);
    end
end
exit;
