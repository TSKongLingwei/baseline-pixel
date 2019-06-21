clear all
clc
rand('seed',0);
folder = fullfile(pwd);
subFolder_org = struct2cell(dir([folder '/DUT-train/DUT-train-Image/']))';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=3:length(subFolder_org)
    i
    filename_img = fullfile(folder,'/DUT-train/DUT-train-Image/',subFolder_org{i});
    img = imread(filename_img);
    img = imresize(img,[416,416]);
    imwrite(img,fullfile(folder,'/DUT-train-416/DUT-train-Image/',subFolder_org{i}))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    filename_label = fullfile(folder,'/DUT-train/DUT-train-Mask/',[subFolder_org{i}(1:end-4) '.png']);
    label = imread(filename_label);
    label = imresize(label,[416,416]);
    im = uint8(label/255);
    imwrite(im,fullfile(folder,'/DUT-train-416/DUT-train-Mask/',[subFolder_org{i}(1:end-4) '.png']))
end