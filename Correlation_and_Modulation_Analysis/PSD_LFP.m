
% Calculate power spectral density (PSD) of LFP 
% Calculate PSD for each of segmented epochs from the state to analyze
% (whole session, freezing or non-freezing) and take average over all epochs


clc;  clear;


% --- Parameter list ---

csc = 'BLA';  % CA1, ACC or BLA
state = 'exp';  % state to analyze: 'whole', 'frz' (freezing) or 'exp' (non-freezing)
epoch_length_sec = 1.28;  % unit length of frz/exp period to segment into, in sec
cutoff_hz = [1,400];  % for offline filter to remove artifact; e.g. [1,400], [0,0] when not used
nfft = 2048; 
nw = 1.5; 

sign_re_inversion = 'yes';  % re-invert sign (postiive/negative) of LFP 
srate_hz = 1600; 

% --- End of parameter list ---


[~, list_csc] = xlsread(strcat('csc_list_',csc,'.csv'));

f_out = fopen(strcat('psd','_',csc,'_allmice.csv'),'w');

fprintf(f_out, strcat('Region=',num2str(csc),'  State=',num2str(state),...
       '  EpochLength=',num2str(epoch_length_sec),...
       'sec  LowCut=',num2str(cutoff_hz(1)),'Hz  HighCut=',num2str(cutoff_hz(2)),...
       'Hz  NFFT=',num2str(nfft),'  NW=',num2str(nw),'\n\n'));
   
fprintf(f_out,'Mouse,Session,Region,,');

psd_freq = 0: srate_hz/nfft: srate_hz/2;
fprintf(f_out,'%.2f,',psd_freq);
fprintf(f_out,'\n');


for cnt_csc = 1:size(list_csc,1)
    
    fname_csc = list_csc{cnt_csc,1};

    fprintf('\n=== Processing CSC: \n%s\n', fname_csc);

    [s1, s2] = fileparts(fname_csc);
    name_mouse = s1(3:7);
    name_session = s1(9:11);
    name_csc = s2(1:end-6);  
    
    fprintf(f_out, '%s,%s,%s,,', name_mouse, name_session, name_csc);


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
    
    % proceed to the next csc when there is no trial (period) of the state to analyze
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
    
    % proceed to the next csc when there is no epoch extracted
    if isempty(beg_ts)
        fprintf(f_out, '\n'); 
        continue; 
    end  
        
    num_epoch = numel(beg_ts);
    

    % extract the entire LFP
    [lfp_data_mV, lfp_ts_usec] = NlxNcsGetAll(fname_csc);
    if strcmp(sign_re_inversion, 'yes'), lfp_data_mV = -lfp_data_mV; end  
     
    % filter LFP (if necessary)
    if cutoff_hz(1) ~= 0
        flt_order = 3; 
        [B, A] = butter(flt_order, cutoff_hz / (srate_hz/2), 'bandpass'); 
        hflt_filter = dfilt.df2(B,A);
        lfp_data_mV = filtfilt(hflt_filter.Numerator, hflt_filter.Denominator, lfp_data_mV);
    end


    % calculate PSD averaged over all epochs

    psd_epoch = NaN(num_epoch, numel(psd_freq));

    for cnt_epoch = 1:num_epoch
        beg_id = find(lfp_ts_usec > beg_ts(cnt_epoch), 1); 
        epoch_id = beg_id: beg_id + epoch_length_sec*srate_hz - 1;
        psd_epoch(cnt_epoch,:) = pmtm(lfp_data_mV(epoch_id), nw, nfft, srate_hz);
    end

    psd_mean = mean(psd_epoch, 1);
       
        
    % print

    fprintf(f_out,'%.10f,',psd_mean);
    fprintf(f_out,'\n');
        
end


fclose('all');

