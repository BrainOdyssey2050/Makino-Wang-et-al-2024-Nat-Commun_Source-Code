
% Calculate the strength of phase-amplitude coupling between two LFPs, 
% which is shown by modulation index (Tort et al. 2010 J Neurophysiol) 
% Calculate coupling for each of segmented epochs from the state to analyze
% (whole session, freezing or non-freezing) and take average over all epochs


clc;  clear;


% --- Parameter list ---

csc1 = 'CA1';  % CSC to extract phase - CA1, ACC or BLA
csc2 = 'BLA';  % CSC to extract amplitude (envelope)
state = 'exp';  % state to analyze: 'whole', 'frz' (freezing) or 'exp' (non-freezing)
epoch_length_sec = 1.28;  % unit length of frz/exp period to segment into, in sec
csc1_cutoff_hz = [6,12]; 
csc2_cutoff_hz = [30,50];
bin_size = 10;  % size of phase bins (in degrees), must be a divisor of 360 

sign_re_inversion = 'yes';  % re-invert sign (postiive/negative) of LFP 
srate_hz = 1600; 

% --- End of parameter list ---


num_bin = 360/bin_size;

[~, list_cscpair] = xlsread(strcat('cscpair_list_',csc1,'_',csc2,'.csv'));

f_out = fopen(strcat('cfc','_',csc1,'_',csc2,'_allmice.csv'),'w');

fprintf(f_out, strcat('Region1=',num2str(csc1),'  Region2=',num2str(csc2),...
       '  State=',num2str(state),'  EpochLength=',num2str(epoch_length_sec),...
       'sec  LowCut1=',num2str(csc1_cutoff_hz(1)),'Hz  HighCut1=',num2str(csc1_cutoff_hz(2)),...
       'Hz  LowCut2=',num2str(csc2_cutoff_hz(1)),'Hz  HighCut2=',num2str(csc2_cutoff_hz(2)),... 
       'Hz\n\n'));
   
fprintf(f_out,'Mouse,Session,Region1,Region2,,MI,NumEpoch\n');


for cnt_cscpair = 1:size(list_cscpair,1)
    
    fname_csc1 = list_cscpair{cnt_cscpair,1};
    fname_csc2 = list_cscpair{cnt_cscpair,2};
    
    fprintf('\n=== Processing CSC pair: \n%s\n%s\n', fname_csc1, fname_csc2);

    [s1, s2] = fileparts(fname_csc1);
    name_mouse = s1(3:7);
    name_session = s1(9:11);
    name_csc1 = s2(1:end-6);
    [~, s2] = fileparts(fname_csc2);
    name_csc2 = s2(1:end-6);    
    
    fprintf(f_out, '%s,%s,%s,%s,,', name_mouse, name_session, name_csc1, name_csc2);


    % load timestamps for the beginnings / ends of the specified state
    if strcmp(state, 'whole')
        ev = NlxNevGetEvents(fullfile(s1,'Events.nev'));
        ts_state = [ev(1,2).ts, ev(1,4).ts];  
    else
        ev = NlxNevGetEvents(fullfile(s1,'Events_ff.nev'));
        if strcmp(state, 'frz')
            [beg_ts, end_ts] = NlxNevGetTrial(ev, ev(1).ts, ev(end).ts, 'id', 1, 2);
        else
            [beg_ts, end_ts] = NlxNevGetTrial(ev, ev(1).ts, ev(end).ts, 'id', 3, 4);        
        end
        ts_state = [beg_ts, end_ts];
    end
    
    % proceed to the next cscpair when there is no trial (period) of the state to analyze
    if isempty(ts_state)
        fprintf(f_out, '\n'); 
        continue; 
    end  
          
    % divide each trial into multiple epochs
    beg_ts = [];  beg_multi_id = 1;
    for cnt_trial = 1:size(ts_state,1)
        num_epoch = floor((ts_state(cnt_trial,2)-ts_state(cnt_trial,1)) / (epoch_length_sec*1e6) );
        for cnt_epoch = 1:num_epoch
            beg_ts(beg_multi_id) = ts_state(cnt_trial,1) + (cnt_epoch-1) * (epoch_length_sec*1e6);
            beg_multi_id = beg_multi_id + 1;
        end
    end
    
    % proceed to the next cscpair when there is no epoch extracted
    if isempty(beg_ts)
        fprintf(f_out, '\n'); 
        continue; 
    end  
        
    num_epoch = numel(beg_ts);
    

    % extract the entire LFP for both CSCs 
    [lfp1_data_mV, lfp_ts_usec] = NlxNcsGetAll(fname_csc1);
     lfp2_data_mV = NlxNcsGetAll(fname_csc2);    
    if strcmp(sign_re_inversion, 'yes')
        lfp1_data_mV = -lfp1_data_mV;
        lfp2_data_mV = -lfp2_data_mV;        
    end  
     
    % filter both LFPs
    lfp1_filtered = eegfilt(lfp1_data_mV, srate_hz, csc1_cutoff_hz(1), csc1_cutoff_hz(2));
    lfp2_filtered = eegfilt(lfp2_data_mV, srate_hz, csc2_cutoff_hz(1), csc2_cutoff_hz(2));   
    
    % obtain phase for LFP1
    lfp1_phase = Hilbert2PhaseDeg(hilbert(lfp1_filtered));
    
    % obtain envelope for LFP2
    lfp2_envelope = abs(hilbert(lfp2_filtered));
        
    % for each epoch, bin LFP1 ids based on theta phase   
    lfp1_phase_bin_epoch_id = cell(num_bin,num_epoch);
    for cnt_epoch = 1:num_epoch
        beg_id = find(lfp_ts_usec > beg_ts(cnt_epoch), 1); 
        epoch_id = beg_id: beg_id + epoch_length_sec*srate_hz - 1;
        epoch_id = epoch_id';
        for cnt_bin = 1:num_bin    
            id = find(lfp1_phase(epoch_id) >= (cnt_bin-1)*bin_size ...
                    & lfp1_phase(epoch_id) < cnt_bin*bin_size);
            lfp1_phase_bin_epoch_id{cnt_bin,cnt_epoch} = epoch_id(id);        
        end
    end
 
    
    % calculate modulation index averaged over all epochs
   
    mi_epoch = NaN(1,num_epoch);

    for cnt_epoch = 1:num_epoch
        
        lfp2_envelope_bin = NaN(num_bin,1);
        for cnt_bin = 1:num_bin
            lfp2_envelope_bin(cnt_bin) = mean...
               (lfp2_envelope(lfp1_phase_bin_epoch_id{cnt_bin,cnt_epoch}));
        end

        lfp2_envelope_bin = lfp2_envelope_bin / sum(lfp2_envelope_bin); 

        tmp = lfp2_envelope_bin .* log2(lfp2_envelope_bin);
        tmp(isnan(tmp)) = 0;  % treat special case: 0 * log2(0) == 0
        H = -sum(tmp);  % entropy

        mi_epoch(cnt_epoch) = (log2(num_bin) - H) / log2(num_bin);

    end
            
    mi_mean = mean(mi_epoch);
       
        
    % print
    fprintf(f_out, '%.6f,%.0f\n', mi_mean, num_epoch);     
        
end


fclose('all');

