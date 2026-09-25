function [h_old,h_new] = vis_artifacts(new,old,varargin)
% vis_artifacts(NewEEG,OldEEG,Options...)
% Display the artifact rejections done by any of the artifact cleaning functions.
%
% Keyboard Shortcuts:
%   [n] : display just the new time series
%   [o] : display just the old time series
%   [b] : display both time series super-imposed
%   [d] : display the difference between both time series
%   [+] : increase signal scale
%   [-] : decrease signal scale
%   [*] : expand time range
%   [/] : reduce time range
%   [h] : show/hide slider
%   [e] : toggle events
%   [l] : toggle event legend
%
% In:
%   NewEEG     : cleaned continuous EEG data set
%   OldEEG     : original continuous EEG data set
%   Options... : name-value pairs specifying the options, with names:
%                'YRange' : y range of the figure that is occupied by the signal plot
%                'YScaling' : distance of the channel time series from each other in std. deviations
%                'WindowLength : window length to display
%                'NewColor' : color of the new (i.e., cleaned) data
%                'OldColor' : color of the old (i.e., uncleaned) data
%                'HighpassOldData' : whether to high-pass the old data if not already done
%                'ScaleBy' : the data set according to which the display should be scaled, can be
%                            'old', 'new' or 'noscale' (default: 'new')
%                'ChannelSubset' : optionally a channel subset to display
%                'TimeSubet' : optionally a time subrange to display
%                'DisplayMode' : what should be displayed: 'both', 'new', 'old', 'diff'
%                'ShowSetname' : whether to display the dataset name in the title
%                'EqualizeChannelScaling' : optionally equalize the channel scaling
%                See also code for more options.
%
% Notes:
%   This function is primarily meant for testing purposes and is not a particularly high-quality
%   implementation. If you are a MATLAB expert and have an interest in this function we would
%   appreciate it if you chose to do a clean rewrite. :)
%
% Examples:
%  vis_artifacts(clean,raw)
%
%  % display only a subset of channels
%  vis_artifacts(clean,raw,'ChannelSubset',1:4:raw.nbchan);
%
%
%                                Christian Kothe, Swartz Center for Computational Neuroscience, UCSD
%                                2010-07-06
%
%                                relies on the findjobj() function by Yair M. Altman.
%
% 07/27/2018 Makoto. Addressed the issue of ResizeFcn behavior change started in R2014b. https://www.mathworks.com/help/matlab/graphics_transition/why-has-the-behavior-of-resizefcn-changed.html

% Copyright (C) Christian Kothe, SCCN, 2012, ckothe@ucsd.edu
%
% This program is free software; you can redistribute it and/or modify it under the terms of the GNU
% General Public License as published by the Free Software Foundation; either version 2 of the
% License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
% even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program; if not,
% write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
% USA

have_signallegend = false;
have_eventlegend = false;

if nargin < 2
    old = new;
elseif ischar(old)
    varargin = [{old} varargin];
    old = new;
end

% parse options
opts = hlp_varargin2struct(varargin, ...
    {'yrange','YRange'}, [0.05 0.95], ...       % y range of the figure occupied by the signal plot
    {'yscaling','YScaling'}, 3.5, ...           % distance of the channel time series from each other in std. deviations
    {'wndlen','WindowLength'}, 10, ...          % window length to display
    {'newcol','NewColor'}, [0 0 0.5], ...       % color of the new (i.e., cleaned) data
    {'oldcol','OldColor'}, [1 0 0], ...         % color of the old (i.e., uncleaned) data
    {'highpass_old','HighpassOldData'},true, ...% whether to high-pass the old data if not already done
    {'show_removed_portions','ShowRemovedPortions'},true, ...% whether to show removed data portions (if only one set is passed in)
    {'show_events','ShowEvents'},true, ...      % whether to show events
    {'show_eventlegend','ShowEventLegend'},false, ...  % whether to show a legend for the currently visible events
    {'scale_by','ScaleBy'},'allnew',...         % the data set according to which the display should be scaled (can be allold, allnew, wndold, wndnew or noscale)
    {'channel_subset','ChannelSubset'},[], ...  % optionally a channel subset to display
    {'time_subset','TimeSubset'},[],...         % optionally a time subrange to display
    {'display_mode','DisplayMode'},'both',...   % what should be displayed: 'both', 'new', 'old', 'diff'
    {'show_setname','ShowSetname'},true,...     % whether to display the dataset name in the title
    {'line_spec','LineSpec'},'-',...            % line style for plotting
    {'line_width','LineWidth'},0.5,...          % line width
    {'add_legend','AddLegend'},false,...        % add a signal legend
    {'equalize_channel_scaling','EqualizeChannelScaling'},false);  % optionally equalize the channel scaling

% ensure that the data are not epoched (but do NOT expand rejections to save memory)
if opts.show_removed_portions
    new = expand_rejections(to_continuous(new));
    old = expand_rejections(to_continuous(old));
end
new.chanlocs = old.chanlocs;

% note: filter delay is now corrected on-the-fly inside window slicing functions

% make sure that the old data is high-passed the same way as the new data
if opts.highpass_old && isfield(new.etc,'clean_drifts_kernel') && ~isfield(old.etc,'clean_drifts_kernel')
    old.data = old.data';
    for c=1:old.nbchan
        old.data(:,c) = filtfilt_fast(new.etc.clean_drifts_kernel,1,old.data(:,c)); end
    old.data = old.data';
end

if isscalar(opts.line_width)
    opts.line_width = [opts.line_width opts.line_width]; end

% optionally pick a subrange to work on
if ~isempty(opts.channel_subset)
    old = pop_select(old,'channel',opts.channel_subset);
    new = pop_select(new,'channel',opts.channel_subset);
end

if ~isempty(opts.time_subset)
    old = pop_select(old,'time',opts.time_subset);
    new = pop_select(new,'time',opts.time_subset);
end

if opts.equalize_channel_scaling
    rescale = 1./mad(old.data,[],2);
    new.data = bsxfun(@times,new.data,rescale);
    old.data = bsxfun(@times,old.data,rescale);
end

% generate event colormap - fixed for numeric event type by Makoto 5/15/14
if ~isempty(old.event)
    if isnumeric(old.event(1).type)
        for tmpIdx = 1:length(old.event)
            old.event(tmpIdx).type = num2str(old.event(tmpIdx).type);
        end
    end
    opts.event_colormap = gen_colormap(old.event,'jet');
end

% calculate whole-data scale
old_iqr = 2*mad(quantile(old.data',1000),1)';
old_iqr(isnan(old_iqr)) = deal(mean(old_iqr(~isnan(old_iqr))));
new_iqr = 2*mad(quantile(new.data',1000),1)';
new_iqr(isnan(new_iqr)) = deal(mean(new_iqr(~isnan(new_iqr))));

% Get actual dimensions of expanded channel and sample spaces
if isfield(new.etc, 'clean_channel_mask') && isfield(new.etc, 'clean_sample_mask') && ...
   islogical(new.etc.clean_channel_mask) && sum(new.etc.clean_channel_mask) == size(new.data, 1) && ...
   islogical(new.etc.clean_sample_mask) && sum(new.etc.clean_sample_mask) == size(new.data, 2)
    new_nbchan = length(new.etc.clean_channel_mask);
    new_pnts = length(new.etc.clean_sample_mask);
else
    new_nbchan = size(new.data, 1);
    new_pnts = size(new.data, 2);
end

% Precalculate filter delays in samples to shift on-the-fly
new_delay_samples = 0;
if isfield(new.etc, 'filter_delay')
    new_delay_samples = round(new.etc.filter_delay * new.srate);
end
old_delay_samples = 0;
if isfield(old.etc, 'filter_delay')
    old_delay_samples = round(old.etc.filter_delay * old.srate);
end

% create figure & slider
lastPos = 0;
%hFig = figure('ResizeFcn',@on_window_resized,'KeyPressFcn',@(varargin)on_key(varargin{2}.Key)); hold; axis();
hFig = figure('KeyPressFcn',@(varargin)on_key(varargin{2}.Key)); hold on; axis();
hAxis = gca;

% Set axis limits and ticks/labels once during initialization
xlim(hAxis, [0 1]);
ylim(hAxis, [0 1]);
ylr = [opts.yrange(1) opts.yrange(2)];
channel_y = (ylr(2):(ylr(1)-ylr(2))/(new_nbchan-1):ylr(1))';
if isfield(old.chanlocs,'labels')
    set(hAxis,'ytick',channel_y(end:-1:1));
    labels = {old.chanlocs.labels};
    set(hAxis,'yticklabel',labels(end:-1:1));
end

% Pre-create line objects to update in-place for fast rendering
h_old_line = plot(hAxis, NaN, NaN, 'Color', opts.oldcol, 'LineWidth', opts.line_width(1), 'LineStyle', opts.line_spec);
h_new_line = plot(hAxis, NaN, NaN, 'Color', opts.newcol, 'LineWidth', opts.line_width(end), 'LineStyle', opts.line_spec);
h_bad_line = plot(hAxis, NaN, NaN, 'Color', [1 0 1], 'LineWidth', opts.line_width(end), 'LineStyle', opts.line_spec);
h_diff_line = plot(hAxis, NaN, NaN, 'Color', opts.newcol, 'LineWidth', opts.line_width(1), 'LineStyle', opts.line_spec);
h_event_lines = [];

hSlider = uicontrol('style','slider','KeyPressFcn',@(varargin)on_key(varargin{2}.Key)); on_resize();
set(hSlider, 'callback', @on_update);
addlistener(hSlider, 'ContinuousValueChange', @(~,~) on_update());
% jSlider = findjobj(hSlider);
% jSlider.AdjustmentValueChangedCallback = @on_update;

annotation(hFig,'textbox', [0 0.07 1 0],...
    'String', {'Keyboard shortcuts: [n] new data, [o] old data, [b] both data, [d] difference, [+] increase amp scale, [-] decrease amp scale, [*] shrink time scale, [/] expand time scale, [h] show/hide slider.'},...
    'HorizontalAlignment','center', 'FontSize',14, 'FitBoxToText','off', 'LineStyle','none');

% Implementing a delay fuse (07/27/2017 Makoto).
set(hFig, 'ResizeFcn', @on_window_resized);

% do the initial update
%on_update();


    function repaint(relPos,moved)
        % repaint the current data
        
        % if this happens, we are maxing out MATLAB's graphics pipeline: let it catch up
        if relPos == lastPos && moved
            return; end
        
        % compute pixel range from axis properties
        xl = [0 1];
        yl = [0 1];
        fp = get(hFig,'Position');
        ap = get(hAxis,'Position');
        pixels = (fp(3))*(ap(3)-ap(1));
        ylr = [opts.yrange(1) opts.yrange(2)];
        
        % compute sample range
        wndsamples = opts.wndlen * new.srate;
        pos = floor((new_pnts-wndsamples)*relPos);
        wndindices = 1 + floor(0:wndsamples/pixels:(wndsamples-1));
        wndrange = pos+wndindices;
        wndrange(wndrange > new_pnts) = new_pnts;
        
        % Get data on-the-fly while accounting for filter delay shifts
        oldwnd = get_raw_window(old.data, wndrange + old_delay_samples);
        newwnd = get_window_data(new, wndrange + new_delay_samples);
        
        switch opts.scale_by
            case 'allnew'
                iqrange = new_iqr;
            case 'allold'
                iqrange = old_iqr;
            case {'wndnew','new'}
                iqrange = mad(newwnd',1)';
                iqrange(isnan(iqrange)) = mad(oldwnd(isnan(iqrange),:)',1)';
            case {'wndold','old'}
                iqrange = mad(oldwnd',1)';
            case 'noscale'
                iqrange = ones(new_nbchan,1);
            otherwise
                error('Unsupported scale_by option.');
        end
        scale = ((ylr(2)-ylr(1))/new_nbchan) ./ (opts.yscaling*iqrange); scale(~isfinite(scale)) = 0;
        scale(scale>median(scale)*3) = median(scale);
        scale = scale(:);
        scale = repmat(scale,1,length(wndindices));
        
        % draw
        if opts.show_setname
            tit = sprintf('%s - ',[old.filepath filesep old.filename]);
        else
            tit = '';
        end
        
        if ~isempty(wndrange)
            tit = [tit sprintf('[%.1f - %.1f]',new.xmin + (wndrange(1)-1)/new.srate, new.xmin + (wndrange(end)-1)/new.srate)];
        end
        
        xrange = xl(1):(xl(2)-xl(1))/(length(wndindices)-1):xl(2);
        yoffset = repmat(channel_y,1,length(wndindices));
        
        % Fast plotting optimization: vectorize lines with NaN separators
        insert_nans = @(Y) reshape([Y, NaN(size(Y,1),1)]', 1, []);
        x_nan = insert_nans(repmat(xrange, size(oldwnd,1), 1));
        
        % Hide all line objects first
        set(h_old_line, 'Visible', 'off');
        set(h_new_line, 'Visible', 'off');
        set(h_bad_line, 'Visible', 'off');
        set(h_diff_line, 'Visible', 'off');
        
        switch opts.display_mode
            case 'both'
                title(hAxis, [tit '; superposition'],'Interpreter','none');
                set(h_old_line, 'XData', x_nan, 'YData', insert_nans(yoffset + scale.*oldwnd), 'Visible', 'on');
                
                % Check for bad channels to highlight in red
                if isfield(new, 'etc') && isfield(new.etc, 'GEDAI') && isfield(new.etc.GEDAI, 'bad_channels_removed') && ~isempty(new.etc.GEDAI.bad_channels_removed)
                    bad_chans = new.etc.GEDAI.bad_channels_removed;
                    good_chans = setdiff(1:size(newwnd,1), bad_chans);
                    
                    % Plot good channels in newcol
                    newwnd_good = nan(size(newwnd));
                    newwnd_good(good_chans,:) = newwnd(good_chans,:);
                    set(h_new_line, 'XData', x_nan, 'YData', insert_nans(yoffset + scale.*newwnd_good), 'Visible', 'on');
                    
                    % Plot bad channels in magenta to distinguish from original red data
                    newwnd_bad = nan(size(newwnd));
                    newwnd_bad(bad_chans,:) = newwnd(bad_chans,:);
                    set(h_bad_line, 'XData', x_nan, 'YData', insert_nans(yoffset + scale.*newwnd_bad), 'Visible', 'on');
                else
                    set(h_new_line, 'XData', x_nan, 'YData', insert_nans(yoffset + scale.*newwnd), 'Visible', 'on');
                end
            case 'new'
                title(hAxis, [tit '; cleaned'],'Interpreter','none');
                set(h_new_line, 'XData', x_nan, 'YData', insert_nans(yoffset + scale.*newwnd), 'Visible', 'on');
            case 'old'
                title(hAxis, [tit '; original'],'Interpreter','none');
                set(h_old_line, 'XData', x_nan, 'YData', insert_nans(yoffset + scale.*oldwnd), 'Visible', 'on');
            case 'diff'
                title(hAxis, [tit '; difference'],'Interpreter','none');
                set(h_diff_line, 'XData', x_nan, 'YData', insert_nans(yoffset + scale.*(oldwnd-newwnd)), 'Visible', 'on');
        end
        
        % Assign output variables in outer scope
        h_old = h_old_line;
        h_new = h_new_line;
        
        % Clear existing event lines
        if ~isempty(h_event_lines)
            delete(h_event_lines(ishghandle(h_event_lines)));
            h_event_lines = [];
        end
        
        % also plot events
        if opts.show_events && ~isempty(old.event)
            evtlats = [old.event.latency];
            evtindices = find(evtlats>wndrange(1) & evtlats<wndrange(end));
            if ~isempty(evtindices)
                evtpos = xl(1) + (evtlats(evtindices)-wndrange(1))/wndsamples*(xl(2)-xl(1));
                evttypes = {old.event(evtindices).type};
                % for each visible type
                visible_types = unique(evttypes);
                handles = [];
                labels = {};
                for ty = visible_types(:)'
                    if isnumeric(ty)
                        curtype = ty(1);
                    else
                        curtype = ty{1};
                    end
                    curcolor = opts.event_colormap.values(strcmp(opts.event_colormap.keys,curtype),:);
                    matchpos = strcmp(evttypes,curtype);
                    h = line(hAxis, [evtpos(matchpos);evtpos(matchpos)],repmat([0;1],1,nnz(matchpos)),'Color',curcolor);
                    h_event_lines = [h_event_lines; h(:)];
                    handles(end+1) = h(1);
                    labels{end+1} = curtype;
                end
                if opts.show_eventlegend
                    legend(hAxis, handles,labels,'Location','NorthWest');
                    have_eventlegend = true;
                elseif have_eventlegend
                    legend(hAxis, 'off');
                    have_eventlegend = false;
                end
            end
        end
        axis(hAxis, [0 1 0 1]);
        
        if opts.add_legend && ~have_signallegend
            legend(hAxis, [h_old_line; h_new_line],'Original','Corrected');
            have_signallegend = 1;
        end
        try
            drawnow limitrate;
        catch
            drawnow;
        end
        
        lastPos = relPos;
    end


    function on_update(varargin)
        % slider moved
        repaint(get(hSlider,'Value'),~isempty(varargin));
    end

    function on_resize(varargin)
        % adapt/set the slider's size
        wPos = get(hFig,'Position');
        if ~isempty(hSlider)
            try
                set(hSlider,'Position',[20,20,wPos(3)-40,20]);
            catch,end
            on_update;
        end
    end

    function on_window_resized(varargin)
        % window resized
        on_resize();
    end

    function EEG = to_continuous(EEG)
        % convert an EEG set to continuous if currently epoched
        if ndims(EEG.data) == 3
            EEG.data = EEG.data(:,:);
            [EEG.nbchan,EEG.pnts,EEG.trials] = size(EEG.data);
        end
    end

    function EEG = expand_rejections(EEG)
        if ~isfield(EEG.etc,'clean_channel_mask') || ~islogical(EEG.etc.clean_channel_mask) || sum(EEG.etc.clean_channel_mask) ~= size(EEG.data,1)
            EEG.etc.clean_channel_mask = true(1, size(EEG.data,1));
        end
        if ~isfield(EEG.etc,'clean_sample_mask') || ~islogical(EEG.etc.clean_sample_mask) || sum(EEG.etc.clean_sample_mask) ~= size(EEG.data,2)
            EEG.etc.clean_sample_mask = true(1, size(EEG.data,2));
        end
        EEG.etc.cumsum_mask = cumsum(EEG.etc.clean_sample_mask);
    end

    function on_key(key)
        switch lower(key)
            case {'add','+'}
                % decrease datascale
                opts.yscaling = opts.yscaling*0.9;
            case {'subtract','-'}
                % increase datascale
                opts.yscaling = opts.yscaling*1.1;
            case {'multiply','*'}
                % increase timerange
                opts.wndlen = opts.wndlen*1.1;
            case {'divide','/'}
                % decrease timerange
                opts.wndlen = opts.wndlen*0.9;
            % case 'pagedown'
            %     % shift display page offset down
            %     opts.pageoffset = opts.pageoffset+1;
            % case 'pageup'
            %     % shift display page offset up
            %     opts.pageoffset = opts.pageoffset-1;
            case 'n'
                opts.display_mode = 'new';
            case 'o'
                opts.display_mode = 'old';
            case 'b'
                opts.display_mode = 'both';
            case 'd'
                opts.display_mode = 'diff';
            case 'l'
                opts.show_eventlegend = ~opts.show_eventlegend;
            case 'e'
                opts.show_events = ~opts.show_events;
            case 'h'
                if strcmp(get(hSlider,'Visible'),'on')
                    set(hSlider,'Visible','off')
                else
                    set(hSlider,'Visible','on')
                end
        end
        on_update();
    end
end

% create a mapping from event types onto colors
function map = gen_colormap(eventstruct,mapname)
map.keys = unique({eventstruct.type});
if isscalar(map.keys)
    tmp = feval(mapname, 256);
    map.values = tmp(round(end/2),:);
elseif ~isempty(map.keys)
    tmp = feval(mapname, 256);
    map.values = tmp(1+floor((0:length(map.keys)-1)/(length(map.keys)-1)*(length(tmp)-1)),:);
else
    map.values = [];
end
end

function wnd_data = get_window_data(EEG, wndrange)
    if isfield(EEG.etc, 'clean_sample_mask') && isfield(EEG.etc, 'cumsum_mask') && ...
       isfield(EEG.etc, 'clean_channel_mask') && ...
       islogical(EEG.etc.clean_channel_mask) && sum(EEG.etc.clean_channel_mask) == size(EEG.data, 1) && ...
       islogical(EEG.etc.clean_sample_mask) && sum(EEG.etc.clean_sample_mask) == size(EEG.data, 2)
        wnd_data = nan(length(EEG.etc.clean_channel_mask), length(wndrange), 'like', EEG.data);
        max_len = length(EEG.etc.clean_sample_mask);
        in_bounds = (wndrange >= 1) & (wndrange <= max_len);
        if any(in_bounds)
            clipped_range = wndrange(in_bounds);
            is_kept = EEG.etc.clean_sample_mask(clipped_range);
            out_cols = find(in_bounds);
            out_cols = out_cols(is_kept);
            compact_indices = EEG.etc.cumsum_mask(clipped_range(is_kept));
            if ~isempty(compact_indices)
                wnd_data(EEG.etc.clean_channel_mask, out_cols) = EEG.data(:, compact_indices);
            end
        end
    else
        wnd_data = get_raw_window(EEG.data, wndrange);
    end
end

function wnd_data = get_raw_window(data, wndrange)
    wnd_data = nan(size(data, 1), length(wndrange), 'like', data);
    max_len = size(data, 2);
    in_bounds = (wndrange >= 1) & (wndrange <= max_len);
    if any(in_bounds)
        wnd_data(:, in_bounds) = data(:, wndrange(in_bounds));
    end
end
