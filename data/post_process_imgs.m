clear
clc
close all

addpath('natsortfiles');

imgPath = './';

imgPathfirstrow = './firstrow/';
imfirstrow = generate_spliced_img(imgPathfirstrow);

imgPathsecondrow = './secondrow/';
imsecondrow = generate_spliced_img(imgPathsecondrow);
combImg=cat(1, imfirstrow,imsecondrow);
% imwrite(combImg, sprintf([imgPath, 'Fig4_no_text.jpg']));

size_sliced_image = size(combImg);
whiteImage = 255 * ones(40, size_sliced_image(2), 'uint8');

position = [];
for i =1:5
    position = [position;224/2+(i-1)*224 20];
end

% text_str = cell(5,1);
% indexes = [139 477 426 43 37];
% value = [0.98 0.85 0.83 0.75 0.7];
% for i =1:5
%     text_str{i} = [num2str(indexes(i)) ': ' num2str(value(i))];
% end

value = [7.45 4.03 2.96 2.26 1.45];
value = value*-1;
box_color = 'red';
whiteImage_withtext = insertText(whiteImage,position,value,'Font','Times New Roman',...
    'FontSize',36, 'TextColor','black', 'BoxColor',box_color,...
    'BoxOpacity',0, 'AnchorPoint','Center');
% figure
% imshow(whiteImage_withtext),title('whiteImage_withtext');
three_row_img=cat(1, combImg,whiteImage_withtext);
imwrite(three_row_img, sprintf([imgPath, 'Fig10c_advblock_negcat.jpg']));
print('stop');



% position = [];
% for i =1:4
%     position = [position;224/2+(i-1)*224 20];
% end
% 
% text_str = cell(4,1);
% indexes = [425 77 520 399];
% value = [0.58 0.47 0.46 0.45];
% for i =1:4
%     text_str{i} = [num2str(indexes(i)) ': ' num2str(value(i))];
% end

%{
% names = imgDir(10:15).name;
name1 = names;

im1 = imread([imgPath imgDir(2).name]);
im2 = imread([imgPath imgDir(1).name]);
im3 = [im1,im2];
imwrite(im3, sprintf([imgPath, 'figure2_compare_with_guided.jpeg']));
% im0 = imread([imgPath imgDir(3).name]);
im = cell(1, length(imgDir));
for i = 1:length(imgDir)
    im{1,i} = imread([imgPath imgDir(i).name]);
end
im2 = [];
for  i = 1:length(imgDir)
    im2 = [im2; im{1,i}];
end
% im2 = imread([imgPath imgDir(2).name]);
% im3 = [im1, im2];
imwrite(im2, sprintf([imgPath, 'VGG-ResNet50-Lyr5-green-blue-kernels.jpeg']));
sz = [size(im1,1) size(im1,2)] ;
sc = 1344/max(sz) ;
sz = round(sc*sz) ;
im1 = imresize(im1, sz, 'bicubic') ;
im1 = single(im1) ;



im3 = [im1; im2];
imwrite(im3, sprintf([imgPath, 'figure33.jpeg']));
[im0, scalingFactor] = ksresize(im0);
im0 = resizencrop(im0, [224,224]);
blank_img = zeros(size(im0), 'single');
final_image_vgg16_TTT = [im0; blank_img];
final_image_vgg16_TTT = [final_image_vgg16_TTT, im1, im1];
imwrite(final_image_vgg16_TTT, sprintf([imgPath, 'aaa.jpeg']));
%}