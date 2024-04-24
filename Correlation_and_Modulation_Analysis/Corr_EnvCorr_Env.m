
% Calculate correlation between: 
%   LFP envelope correlation between Regions 1 and 2, and 
%   LFP mean envelope at Region 3
% for each of whole session, freezing and exploration (non-freezing) periods


clc;  clear;


% --- Parameter list ---

csc1 = 'CA1';  % region to calcualte envelope correlation (CA1/ACC/BLA)
csc2 = 'ACC';  % region to calcualte envelope correlation
csc3 = 'BLA';  % region to calcualte envelope
csc12_cutoff_hz = [6,12]; 
csc3_cutoff_hz = [6,12];
window_size_pt = 6400;  % in # DATA POINTS, size of windows to calculate 
                        % LFP1-LFP2 envelope correlation and LFP3 mean envelope
slide_size_pt = 3200;  % in # DATA POINTS, sliding step of windows
                       % equal to window_size_pt when not using sliding windows

sign_re_inversion = 'yes';  % re-invert sign (postiive/negative) of LFP
srate_hz = 1600; 

% --- End of parameter list ---


window_size_sec = window_size_pt / srate_hz;
slide_size_sec = slide_size_pt / srate_hz;

name_state = {'', 'whole-session', 'freezing periods', 'non-freezing periods'};


[~, list_csctrio] = xlsread(strcat('csctrio_list_',csc1,'_',csc2,'_',csc3,'.csv'));

f_out = fopen(strcat('trireg_corr_envcorr_',csc1,'_',csc2,'_env_',csc3,'_allmice.csv'),'w');
fprintf(f_out,'%s\n',mfilename);
fprintf(f_out, strcat('Region1=',num2str(csc1),'  Region2=',num2str(csc2),'  Region3=',num2str(csc3),...
        '  LowCut12=',num2str(csc12_cutoff_hz(1)),'Hz  HighCut12=',num2str(csc12_cutoff_hz(2)),...
        'Hz  LowCut3=',num2str(csc3_cutoff_hz(1)),'Hz  HighCut3=',num2str(csc3_cutoff_hz(2)),...        
        'Hz  WindowSize=',num2str(window_size_sec),'sec  SlideSize=',num2str(slide_size_sec),'sec\n\n'));
fprintf(f_out,'Mouse,Session,Region1,Region2,Region3,,');
fprintf(f_out,'WholeCorrNWindow,WholeCorrR,,');
fprintf(f_out,'FrzCorrNWindow,FrzCorrR,,');
fprintf(f_out,'ExpCorrNWindow,ExpCorrR\n');

DATA_ALL = cell(size(list_csctrio,1), 4);


for cnt_csctrio = 1:size(list_csctrio,1)
    
    fname_csc1 = list_csctrio{cnt_csctrio,1};
    fname_csc2 = list_csctrio{cnt_csctrio,2};
    fname_csc3 = list_csctrio{cnt_csctrio,3};
    
    fprintf('\n=== Processing CSC trio: \n%s\n%s\n%s\n', fname_csc1, fname_csc2, fname_csc3);

    [s1, s2] = fileparts(fname_csc1);
    name_mouse = s1(3:7);
    name_session = s1(9:11);
    name_csc1 = s2(1:end-6);
    [~, s2] = fileparts(fname_csc2);
    name_csc2 = s2(1:end-6);
    [~, s2] = fileparts(fname_csc3);
    name_csc3 = s2(1:end-6);    
     
    fprintf(f_out, '%s,%s,%s,%s,%s,,', name_mouse, name_session, name_csc1, name_csc2, name_csc3);

    
    timestamp = cell(1,4);  % 2: whole session, 3: freezing, 4: exploration
    
    % load timestamps of the beginning and the end of session
    ev = NlxNevGetEvents(fullfile(s1,'Events.nev'));
    timestamp{2} = [ev(1,2).ts, ev(1,4).ts];  % 'ts_whole' in some scripts
    
    % load timestamps for freezing and exploration events
    ev = NlxNevGetEvents(fullfile(s1,'Events_ff.nev'));
    [fB_ts_usec, fE_ts_usec] = NlxNevGetTrial(ev, ev(1).ts, ev(end).ts, 'id', 1, 2);
    [xB_ts_usec, xE_ts_usec] = NlxNevGetTrial(ev, ev(1).ts, ev(end).ts, 'id', 3, 4);   
    timestamp{3} = [fB_ts_usec, fE_ts_usec];  % 'ts_frz' in some scripts
    timestamp{4} = [xB_ts_usec, xE_ts_usec];  % 'ts_exp' in some scripts
        
    
    % extract the entire LFP for all CSCs 
    [lfp1_data_mV, lfp_ts_usec] = NlxNcsGetAll(fname_csc1);
     lfp2_data_mV = NlxNcsGetAll(fname_csc2);
     lfp3_data_mV = NlxNcsGetAll(fname_csc3);     
    if strcmp(sign_re_inversion, 'yes')
        lfp1_data_mV = -lfp1_data_mV;
        lfp2_data_mV = -lfp2_data_mV;
        lfp3_data_mV = -lfp3_data_mV;
    end     
     
    % filter all LFPs
    lfp1_filtered = eegfilt(lfp1_data_mV, srate_hz, csc12_cutoff_hz(1), csc12_cutoff_hz(2));
    lfp2_filtered = eegfilt(lfp2_data_mV, srate_hz, csc12_cutoff_hz(1), csc12_cutoff_hz(2));
    lfp3_filtered = eegfilt(lfp3_data_mV, srate_hz, csc3_cutoff_hz(1), csc3_cutoff_hz(2));
    
    % Hilbert transform / obtain envelope for all LFPs
    lfp1_envelope = abs(hilbert(lfp1_filtered));
    lfp2_envelope = abs(hilbert(lfp2_filtered));
    lfp3_envelope = abs(hilbert(lfp3_filtered));    
    
    
    for cnt_state = 2:4
        
        fprintf('    === Processing %s\n', name_state{cnt_state});
        
        % skip when there is no trial (period) for that state
        if isempty(timestamp{cnt_state}) 
            fprintf(f_out, 'NaN,NaN,,');
            continue;         
        end  
        
        % divide each state into multiple windows     
        beg_ts = [];  beg_id = []; 
        beg_multi_id = 1;
        for cnt_trial = 1:size(timestamp{cnt_state},1)
            trial_length_sec = (timestamp{cnt_state}(cnt_trial,2) - timestamp{cnt_state}(cnt_trial,1)) / 1e6; 
            num_window = floor( (trial_length_sec - window_size_sec) / slide_size_sec ) + 1;
            if num_window < 0,  num_window = 0;  end
            for cnt_window = 1:num_window
                beg_ts = timestamp{cnt_state}(cnt_trial,1) + (cnt_window-1) * (slide_size_sec*1e6);
                beg_id(beg_multi_id) = find(lfp_ts_usec > beg_ts, 1, 'first'); 
                beg_multi_id = beg_multi_id + 1;
            end
            % this for-loop is skipped when trial_length_sec is shorter than window_size_sec
            % (num_window = 0)
        end
        
        % skip if the number of windows is smaller than 2 (cannot calculate correlation)
        if numel(beg_id) < 2
            fprintf(f_out, 'NaN,NaN,,');   
            continue;  
        end  
        
        num_window = numel(beg_id);

        lfp1_lfp2_envcorr_window = NaN(1,num_window); 
        lfp3_envmean_window = NaN(1,num_window); 
        
        
        % for each window, calculate LFP1-LFP2 envelope correlation and LFP3 mean envelope
        
        for cnt_window = 1:num_window
            
            lfp_id = beg_id(cnt_window): beg_id(cnt_window)+window_size_pt-1;
            
            % calculate LFP1-LFP2 envelope correlation
            corr_matrix = corrcoef(lfp1_envelope(lfp_id), lfp2_envelope(lfp_id));   
            lfp1_lfp2_envcorr_window(cnt_window) = corr_matrix(1,2);         
            
            % calculate LFP3 mean envelope
            lfp3_envmean_window(cnt_window) = mean(lfp3_envelope(lfp_id));
            
        end
        
        
        % calculate correlation between LFP1-LFP2 envcorr and LFP3 envelope
        r_matrix = corrcoef(lfp1_lfp2_envcorr_window, lfp3_envmean_window);        
        corr_r = r_matrix(1,2);
        
        % print
        fprintf(f_out, '%.0f,%.3f,,', num_window, corr_r);    
   
    end
    
    fprintf(f_out, '\n');
    
end


fclose('all');

