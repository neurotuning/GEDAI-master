% Example output demonstration for per-band ENOVA display
% This shows what the new GEDAI output will look like

fprintf('\n=== Example GEDAI Output with Per-Band ENOVA ===\n\n');

fprintf('SENSAI score: 85\n');
fprintf('Mean ENOVA: 0.18\n');
fprintf(' \n');
fprintf('Per-Band ENOVA Scores:\n');
fprintf('  Band       | Center Freq (Hz) | Mean ENOVA\n');
fprintf('  -----------|------------------|------------\n');
fprintf('  Broadband  | N/A              | 0.1500\n');
fprintf('  Band 1     | 94               | 0.2200\n');
fprintf('  Band 2     | 47               | 0.1800\n');
fprintf('  Band 3     | 23               | 0.1600\n');
fprintf('  Band 4     | 12               | 0.1200\n');
fprintf('  Band 5     | 5.9              | 0.0800\n');
fprintf('  Band 6     | 2.9              | 0.0500\n');
fprintf(' \n');

fprintf('\n=== How to Access Per-Band ENOVA ===\n\n');
fprintf('After running GEDAI, you can access:\n');
fprintf('  EEGclean.etc.GEDAI.mean_ENOVA_per_band        %% Vector of mean ENOVA per band\n');
fprintf('  EEGclean.etc.GEDAI.ENOVA_per_epoch_per_band   %% Cell array of per-epoch ENOVA\n');
fprintf('\nOr capture directly:\n');
fprintf('  [~, ~, ~, ~, ~, ~, ~, mean_ENOVA_per_band, ENOVA_per_epoch_per_band] = GEDAI(EEG);\n\n');
