%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            % 
% Outputs from the accumulation simulator are stored as CSV files.   %
% As long as they are in the same directory as this file,            %
%      everything should run as expected                             % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read in the data, stored as two-column CSV
% First column represents channel number
% Second column is accumulator power response ((in dBm)) 
% Change variable 'file' to change which csv file you are using
file = '15Hz_accum_sim.csv';
dataFrame = readtable(file);
trace = table2array(dataFrame);

% Separate data from columns into spectrum and channel
pwr_spec = trace(2,:);
freqs = trace(1,:);

% slice the data to only examine from 0 - 250KHz
center  = find(freqs==0);
last = length(freqs);
response = pwr_spec(center:last);
resp_freqs = freqs(center:last);

%%%%%%%%%%%%%%%%%%%%
% Useful variables %
%%%%%%%%%%%%%%%%%%%%

% pwr_spec -- the double-sided power spectrum in (dBm) 
%             from -250KHz to 250KHz. Split into 8193 channels based on
%             simulation resolution.
%
% response -- single-sided power spectrum in (dBm) from 0 to 250KHz. Split into
%             4097 channels.
%
% freqs -- 8193-channel frequency span from -250KHz-250KHz
%
% resp_freqs -- 4097 channel frequency span from 0-250KHz

%%%%%%%%%%%%%%%%%%%%
%     Plotting     %
%%%%%%%%%%%%%%%%%%%%

% Right now this plots the single-side response from 0-2KHz
% Bounds can be changed easily by modifying the upper-lim for 'xlim'
plot(resp_freqs, response)
title('Broadband Frequency Response of Accumulator (122Hz Integration)');
xlim([0,2000]) % 0 - 2000Hz
xlabel('Frequency [Hz]') 
ylabel('Power [dBW]') 






