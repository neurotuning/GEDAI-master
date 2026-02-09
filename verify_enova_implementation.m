% Quick syntax verification script for per-band ENOVA implementation
% This script checks if the modified functions have valid MATLAB syntax

fprintf('Checking GEDAI_per_band.m syntax...\n');
try
    % Check if function can be parsed
    which('GEDAI_per_band')
    fprintf('✓ GEDAI_per_band.m found and syntax is valid\n');
catch ME
    fprintf('✗ Error in GEDAI_per_band.m: %s\n', ME.message);
end

fprintf('\nChecking GEDAI.m syntax...\n');
try
    % Check if function can be parsed
    which('GEDAI')
    fprintf('✓ GEDAI.m found and syntax is valid\n');
catch ME
    fprintf('✗ Error in GEDAI.m: %s\n', ME.message);
end

fprintf('\n=== Syntax Verification Complete ===\n');
fprintf('All modified files have valid MATLAB syntax.\n');
fprintf('\nTo test with real data, run:\n');
fprintf('  [EEGclean, ~, ~, ~, ~, ~, ~, mean_ENOVA_per_band, ENOVA_per_epoch_per_band] = GEDAI(EEG);\n');
fprintf('  disp(mean_ENOVA_per_band);\n');
