function im2 = generate_spliced_img(imgPath)
% download natsortfiles library before using the code.
% https://ww2.mathworks.cn/matlabcentral/fileexchange/47434-natural-order-filename-sort
% unzip it in your directory.
addpath('natsortfiles');
imgDir  = [ dir([imgPath '*.png']); dir([imgPath '*.jpeg']); dir([imgPath '*.jpg'])];
% S = dir(fullfile(D,'*.txt')); % get list of files in directory
[~,ndx] = natsortfiles({imgDir.name}); % indices of correct order
imgDir = imgDir(ndx); % sort structure using indices
im = cell(1, length(imgDir));
for i = 1:length(imgDir)
    im{1,i} = imread([imgPath imgDir(i).name]);
end
im2 = [];
for  i = 1:length(imgDir)
    im2 = [im2 im{1,i}];
end
