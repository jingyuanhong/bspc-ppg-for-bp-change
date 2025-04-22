% preprocess_bp_data.m
% This script processes PPG and blood pressure data from VitalDB
% to prepare training and testing datasets stored in HDF5 format.
% Make sure the data files are placed in the "data/" directory.

%% Preparation - training
clear; clc;
data_dir = '../../data/';
output_dir = '../../output/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir); 
end
subject_idx = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/subject_idx');
sub_idx = subject_idx;
sub_no = unique(subject_idx);
bp = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/label');
ppg = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/ppg');
sbp = bp(1, :);

%% Select subjects with sufficient data
allPositions = arrayfun(@(x) find(ismember(sub_idx, sub_no(x))), 1:length(sub_no), 'UniformOutput', false);
lenPositions = arrayfun(@(x) length(allPositions{x}), 1:length(sub_no));
validSubjects = sub_no(lenPositions >= 360);

%% Training subject selection
rng(42);
selectSubjects = sort(validSubjects(randperm(length(validSubjects), floor(0.8*length(validSubjects)))));
validPositions = arrayfun(@(x) find(ismember(sub_idx, selectSubjects(x))), 1:length(selectSubjects), 'UniformOutput', false);

%% Calculate SBP change
calcBPC = cellfun(@(pos) arrayfun(@(n) ...
    single(sbp(pos(n)+ (1:(length(pos(n:end)) - 1))) - sbp(pos(n))), 1:length(pos)-1, 'UniformOutput', false), ...
    validPositions, 'UniformOutput', false);

flattenedArray = cellfun(@(x) x(:)', calcBPC, 'UniformOutput', false);
flattenedArray = [flattenedArray{:}];
flattenedArray = [flattenedArray{:}];

concatIdx = cellfun(@(pos, idx) ...
    arrayfun(@(n) repmat(idx, 1, (length(pos(n:end)) - 1)), ...
    1:(length(pos) - 1), 'UniformOutput', false), ...
    validPositions, num2cell(1:length(validPositions)), ...
    'UniformOutput', false);

flattenedIdx = cellfun(@(x) x(:)', concatIdx, 'UniformOutput', false);
flattenedIdx = [flattenedIdx{:}];
flattenedIdx = [flattenedIdx{:}];
%
calcPos = cellfun(@(pos) arrayfun(@(n) ...
    [single(pos(n)+ (1:(length(pos(n:end)) - 1))); single(pos(n)*ones(1,length(pos(n:end))-1))], 1:length(pos)-1, 'UniformOutput', false), ...
    validPositions, 'UniformOutput', false);

flattenedPos = cellfun(@(x) x(:)', calcPos, 'UniformOutput', false);
flattenedPos = [flattenedPos{:}];
flattenedPos = [flattenedPos{:}];

%% Balanced sampling by SBP change range
ranges = {[-Inf, -50], [-50, -40], [-40, -30], [-30, -20], [-20, -10], [-10, 0], [0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, Inf]};
sample_sizes = [9000, 9000, 9000, 4500, 4500, 4500, 4500, 4500, 4500, 9000, 9000, 9000];
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
numFiles = 1;
groupSize = ceil(numSubjects / numFiles);
file_prefix = fullfile(output_dir, 'sbp_trn_30_part'); % for example

for i = 1:numFiles
    startIdx = (i-1) * groupSize + 1;
    endIdx = min(i * groupSize, numSubjects);
    subjectsInGroup = uniqueSubjects(startIdx:endIdx);
    filterIdx = ismember(selectedIdx, subjectsInGroup);

    fileName = sprintf('%s%d.h5', file_prefix, i);
    h5create(fileName, '/label', [3, length(selectedIdx(:, filterIdx))], 'Datatype', 'single');
    h5create(fileName, '/subject_idx', [1, sum(filterIdx)], 'Datatype', 'single');
    h5create(fileName, '/inputs', [1751, sum(filterIdx)], 'Datatype', 'single');
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

%% Write labels
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
end

fprintf('Training data generation completed.\n');

%% Preparation - Test-I
clear; clc;
data_dir = '../../data/';
output_dir = '../../output/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir); 
end
subject_idx = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/subject_idx');
sub_idx = subject_idx;
sub_no = unique(subject_idx);
bp = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/label');
ppg = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/ppg');
sbp = bp(1, :);

%% Select subjects with sufficient data
allPositions = arrayfun(@(x) find(ismember(sub_idx, sub_no(x))), 1:length(sub_no), 'UniformOutput', false);
lenPositions = arrayfun(@(x) length(allPositions{x}), 1:length(sub_no));
validSubjects = sub_no(lenPositions >= 360);

%% Test-I subject selection
rng(42)
selectSubjects = sort(validSubjects(randperm(length(validSubjects),floor(0.8*length(validSubjects)))));
remainingSubjects = setdiff(validSubjects, selectSubjects);
tstSubjects = sort(remainingSubjects(randperm(length(remainingSubjects),ceil(0.5*length(remainingSubjects)))));
validPositions = arrayfun(@(x) find(ismember(sub_idx, tstSubjects(x))), 1:length(tstSubjects), 'UniformOutput', false);

%% Calculate SBP change
calcBPC = cellfun(@(pos) arrayfun(@(n) ...
    single(sbp(pos(n)+ (1:(length(pos(n:end)) - 1))) - sbp(pos(n))), 1:length(pos)-1, 'UniformOutput', false), ...
    validPositions, 'UniformOutput', false);

flattenedArray = cellfun(@(x) x(:)', calcBPC, 'UniformOutput', false);
flattenedArray = [flattenedArray{:}];
flattenedArray = [flattenedArray{:}];

concatIdx = cellfun(@(pos, idx) ...
    arrayfun(@(n) repmat(idx, 1, (length(pos(n:end)) - 1)), ...
    1:(length(pos) - 1), 'UniformOutput', false), ...
    validPositions, num2cell(1:length(validPositions)), ...
    'UniformOutput', false);

flattenedIdx = cellfun(@(x) x(:)', concatIdx, 'UniformOutput', false);
flattenedIdx = [flattenedIdx{:}];
flattenedIdx = [flattenedIdx{:}];
%
calcPos = cellfun(@(pos) arrayfun(@(n) ...
    [single(pos(n)+ (1:(length(pos(n:end)) - 1))); single(pos(n)*ones(1,length(pos(n:end))-1))], 1:length(pos)-1, 'UniformOutput', false), ...
    validPositions, 'UniformOutput', false);

flattenedPos = cellfun(@(x) x(:)', calcPos, 'UniformOutput', false);
flattenedPos = [flattenedPos{:}];
flattenedPos = [flattenedPos{:}];

%% Balanced sampling by SBP change range
ranges = {[-Inf, -50], [-50, -40], [-40, -30], [-30, -20], [-20, -10], [-10, 0], [0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, Inf]};
sample_sizes = [240, 240, 240, 120, 120, 120, 120, 120, 120, 240, 240, 240];
selectedPositions = [];
for i = 1:length(ranges)
    filteredIndices = find(flattenedArray >= ranges{i}(1) & flattenedArray < ranges{i}(2));
     if length(filteredIndices) < sample_sizes(i)
        sample_sizes(i) = length(filteredIndices);
    else
    end
    randomIndices = sort(randperm(numel(filteredIndices), sample_sizes(i)));
    selectedPositions = sort([selectedPositions, filteredIndices(randomIndices)]);
end
selectedDeltaBP = flattenedArray(selectedPositions);
selectedIdx = flattenedIdx(selectedPositions);
selectedPos = flattenedPos(:,selectedPositions);

%% Extracted selected inputs
inputs = arrayfun(@(x) [ppg(:,selectedPos(1,x));ppg(:,selectedPos(2,x));sbp(selectedPos(2,x))],1:length(selectedPos),'UniformOutput', false);
inputs = cell2mat(inputs);
sbpcValues =  [-Inf, -30, 30,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
labels = sbpcCategories(:,categoryIndex);

%% Write Test-I inputs and labels
file_name = fullfile(output_dir, 'sbp_tst_i_30');
h5create(file_name,'/label',size(labels),'Datatype','single');
h5create(file_name, '/subject_idx',size(selectedIdx),'Datatype','single');
h5create(file_name, '/inputs',size(inputs),'Datatype','single');
h5write(file_name,'/label',labels);
h5write(file_name, '/subject_idx',selectedIdx);
h5write(file_name,'/inputs',inputs);
fprintf('Test-I data generation completed.\n');

%% Preparation - Test-II
clear; clc;
data_dir = '../../data/';
output_dir = '../../output/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir); 
end
subject_idx = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/subject_idx');
sub_idx = subject_idx;
sub_no = unique(subject_idx);
bp = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/label');
ppg = h5read(fullfile(data_dir, 'vitaldb_sample.h5'), '/ppg');
sbp = bp(1, :);

%% Select subjects with sufficient data
allPositions = arrayfun(@(x) find(ismember(sub_idx, sub_no(x))), 1:length(sub_no), 'UniformOutput', false);
lenPositions = arrayfun(@(x) length(allPositions{x}), 1:length(sub_no));
validSubjects = sub_no(lenPositions >= 360);

%% Test-II subject selection
rng(42)
selectSubjects = sort(validSubjects(randperm(length(validSubjects),floor(0.8*length(validSubjects)))));
remainingSubjects = setdiff(validSubjects, selectSubjects);
tstSubjects = sort(remainingSubjects(randperm(length(remainingSubjects),ceil(0.5*length(remainingSubjects)))));
remainingtstSubjects = setdiff(remainingSubjects, tstSubjects);
validPositions = arrayfun(@(x) find(ismember(sub_idx, remainingtstSubjects(x))), 1:length(remainingtstSubjects), 'UniformOutput', false);

%% Calculate SBP change
calcBPC = cellfun(@(pos) arrayfun(@(n) ...
    single(sbp(pos(n)+ (1:(length(pos(n:end)) - 1))) - sbp(pos(n))), 1:length(pos)-1, 'UniformOutput', false), ...
    validPositions, 'UniformOutput', false);

flattenedArray = cellfun(@(x) x(:)', calcBPC, 'UniformOutput', false);
flattenedArray = [flattenedArray{:}];
flattenedArray = [flattenedArray{:}];

concatIdx = cellfun(@(pos, idx) ...
    arrayfun(@(n) repmat(idx, 1, (length(pos(n:end)) - 1)), ...
    1:(length(pos) - 1), 'UniformOutput', false), ...
    validPositions, num2cell(1:length(validPositions)), ...
    'UniformOutput', false);

flattenedIdx = cellfun(@(x) x(:)', concatIdx, 'UniformOutput', false);
flattenedIdx = [flattenedIdx{:}];
flattenedIdx = [flattenedIdx{:}];

calcPos = cellfun(@(pos) arrayfun(@(n) ...
    [single(pos(n)+ (1:(length(pos(n:end)) - 1))); single(pos(n)*ones(1,length(pos(n:end))-1))], 1:length(pos)-1, 'UniformOutput', false), ...
    validPositions, 'UniformOutput', false);

flattenedPos = cellfun(@(x) x(:)', calcPos, 'UniformOutput', false);
flattenedPos = [flattenedPos{:}];
flattenedPos = [flattenedPos{:}];

selectedDeltaBP = flattenedArray;
selectedIdx = flattenedIdx;
selectedPos = flattenedPos;

%% Extracted selected inputs
inputs = arrayfun(@(x) [ppg(:,selectedPos(1,x));ppg(:,selectedPos(2,x));sbp(selectedPos(2,x))],1:length(selectedPos),'UniformOutput', false);
inputs = cell2mat(inputs);

% Transfer delta BP to label vectors
sbpcValues =  [-Inf, -5, 5,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label5 = sbpcCategories(:,categoryIndex);

sbpcValues =  [-Inf, -10, 10,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label10 = sbpcCategories(:,categoryIndex);

sbpcValues =  [-Inf, -15, 15,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label15 = sbpcCategories(:,categoryIndex);

sbpcValues =  [-Inf, -20, 20,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label20 = sbpcCategories(:,categoryIndex);

sbpcValues =  [-Inf, -25, 25,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label25 = sbpcCategories(:,categoryIndex);

sbpcValues =  [-Inf, -30, 30,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label30 = sbpcCategories(:,categoryIndex);

sbpcValues =  [-Inf, -35, 35,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label35 = sbpcCategories(:,categoryIndex);

sbpcValues =  [-Inf, -40, 40,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label40 = sbpcCategories(:,categoryIndex);

sbpcValues =  [-Inf, -45, 45,Inf];
sbpcCategories = eye(3, 'single');
categoryIndex = arrayfun(@(x) discretize(x, sbpcValues),selectedDeltaBP);
label45 = sbpcCategories(:,categoryIndex);
%% Write Test-II inputs and labels
file_name = fullfile(output_dir, 'sbp_tst_ii');
h5create(file_name,'/label5',size(label5),'Datatype','single');
h5create(file_name,'/label10',size(label10),'Datatype','single');
h5create(file_name,'/label15',size(label15),'Datatype','single');
h5create(file_name,'/label20',size(label20),'Datatype','single');
h5create(file_name,'/label25',size(label25),'Datatype','single');
h5create(file_name,'/label30',size(label30),'Datatype','single');
h5create(file_name,'/label35',size(label35),'Datatype','single');
h5create(file_name,'/label40',size(label40),'Datatype','single');
h5create(file_name,'/label45',size(label45),'Datatype','single');
h5create(file_name, '/subject_idx',size(selectedIdx),'Datatype','single');
h5create(file_name, '/inputs',size(inputs),'Datatype','single');

h5write(file_name,'/label5',label5);
h5write(file_name,'/label10',label10);
h5write(file_name,'/label15',label15);
h5write(file_name,'/label20',label20);
h5write(file_name,'/label25',label25);
h5write(file_name,'/label30',label30);
h5write(file_name,'/label35',label35);
h5write(file_name,'/label40',label40);
h5write(file_name,'/label45',label45);
h5write(file_name, '/subject_idx',selectedIdx);
h5write(file_name,'/inputs',inputs);
fprintf('Test-II data generation completed.\n');