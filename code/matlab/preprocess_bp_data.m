% preprocess_bp_data.m
% This script processes PPG and blood pressure data from VitalDB
% to prepare training and testing datasets stored in HDF5 format.
% Make sure the data files are placed in the "data/" directory.

%% Preparation
clear; clc;
data_dir = '././data/';
output_dir = '././output/';
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

subject_idx = h5read(fullfile(data_dir, 'vitaldb.h5'), '/subject_idx');
sub_idx = subject_idx;
sub_no = unique(subject_idx);
bp = h5read(fullfile(data_dir, 'vitaldb.h5'), '/label');
ppg = h5read(fullfile(data_dir, 'vitaldb.h5'), '/ppg');
sbp = bp(1, :);
ppgfea = h5read(fullfile(data_dir, 'trn_ppgfea.h5'), '/feature');

%% Select subjects with sufficient data
allPositions = arrayfun(@(x) find(ismember(sub_idx, sub_no(x))), 1:length(sub_no), 'UniformOutput', false);
lenPositions = arrayfun(@(x) length(allPositions{x}), 1:length(sub_no));
validSubjects = sub_no(lenPositions >= 360);

%% Training subject selection
rng(42);
selectSubjects = sort(validSubjects(randperm(length(validSubjects), 1000)));
validPositions = arrayfun(@(x) find(ismember(sub_idx, selectSubjects(x))), 1:length(selectSubjects), 'UniformOutput', false);

%% Calculate SBP change
calcBPC = cellfun(@(pos) arrayfun(@(n) single(sbp(pos(n)+(1:(length(pos(n:end))-1))) - sbp(pos(n))), 1:length(pos)-1, 'UniformOutput', false), validPositions, 'UniformOutput', false);
flattenedArray = cell2mat(cellfun(@(x) x(:)', calcBPC, 'UniformOutput', false));

concatIdx = cellfun(@(pos, idx) arrayfun(@(n) repmat(idx, 1, length(pos(n:end))-1), 1:length(pos)-1, 'UniformOutput', false), validPositions, num2cell(1:length(validPositions)), 'UniformOutput', false);
flattenedIdx = cell2mat(cellfun(@(x) x(:)', concatIdx, 'UniformOutput', false));

calcPos = cellfun(@(pos) arrayfun(@(n) [single(pos(n)+(1:(length(pos(n:end))-1))); single(pos(n)*ones(1,length(pos(n:end))-1))], 1:length(pos)-1, 'UniformOutput', false), validPositions, 'UniformOutput', false);
flattenedPos = cell2mat(cellfun(@(x) x(:)', calcPos, 'UniformOutput', false));

%% Balanced sampling by SBP change range
ranges = {[-Inf, -50], [-50, -40], [-40, -30], [-30, -20], [-20, -10], [-10, 0], [0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, Inf]};
sample_sizes = [900000, 900000, 900000, 450000, 450000, 450000, 450000, 450000, 450000, 900000, 900000, 900000];
selectedPositions = [];
for i = 1:length(ranges)
    filteredIndices = find(flattenedArray >= ranges{i}(1) & flattenedArray < ranges{i}(2));
    randomIndices = sort(randperm(numel(filteredIndices), sample_sizes(i)));
    selectedPositions = sort([selectedPositions, filteredIndices(randomIndices)]);
end

selectedDeltaBP = flattenedArray(selectedPositions);
selectedIdx = flattenedIdx(selectedPositions);
selectedPos = flattenedPos(:, selectedPositions);

%% Save training data in parts
uniqueSubjects = unique(selectedIdx);
numSubjects = length(uniqueSubjects);
numFiles = 6;
groupSize = ceil(numSubjects / numFiles);
file_prefix = fullfile(output_dir, 'sbp_trn_30_part'); % for example

for i = 1:numFiles
    startIdx = (i-1) * groupSize + 1;
    endIdx = min(i * groupSize, numSubjects);
    subjectsInGroup = uniqueSubjects(startIdx:endIdx);
    filterIdx = ismember(selectedIdx, subjectsInGroup);

    fileName = sprintf('%s%d.h5', file_prefix, i);
    h5create(fileName, '/label', [3, sum(filterIdx)], 'Datatype', 'single');
    h5create(fileName, '/subject_idx', [1, sum(filterIdx)], 'Datatype', 'single');
    h5create(fileName, '/inputs', [1751, sum(filterIdx)], 'Datatype', 'single');
    h5create(fileName, '/features', [10, sum(filterIdx)], 'Datatype', 'single');
end

%% Write inputs
for i = 1:numFiles
    startIdx = (i-1) * groupSize + 1;
    endIdx = min(i * groupSize, numSubjects);
    subjectsInGroup = uniqueSubjects(startIdx:endIdx);
    filterIdx = ismember(selectedIdx, subjectsInGroup);
    selePos = selectedPos(:, filterIdx);

    inputs = arrayfun(@(x) [ppg(:, selePos(1, x)); ppg(:, selePos(2, x)); sbp(selePos(2, x))], 1:size(selePos,2), 'UniformOutput', false);
    inputs = cell2mat(inputs);

    fileName = sprintf('%s%d.h5', file_prefix, i);
    h5write(fileName, '/inputs', inputs);
end

%% Write features and labels
features = arrayfun(@(x) [ppgfea(:, selectedPos(1, x)); ppgfea(:, selectedPos(2, x))], 1:size(selectedPos,2), 'UniformOutput', false);
features = cell2mat(features);

sbpcValues = [-Inf, -30, 30, Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues), selectedDeltaBP);
labels = sbpcCategories(:, categoryIndex);

for i = 1:numFiles
    startIdx = (i-1) * groupSize + 1;
    endIdx = min(i * groupSize, numSubjects);
    subjectsInGroup = uniqueSubjects(startIdx:endIdx);
    filterIdx = ismember(selectedIdx, subjectsInGroup);

    fileName = sprintf('%s%d.h5', file_prefix, i);
    h5write(fileName, '/label', labels(:, filterIdx));
    h5write(fileName, '/subject_idx', selectedIdx(:, filterIdx));
    h5write(fileName, '/features', features(:, filterIdx));
end

fprintf('Training data generation completed.\n');