clear all
clc
in_dir=dir('./DUT-train/DUT-train-Image');
imgnum=length(in_dir);
 for i=1:imgnum-2
        i
        imgname=in_dir(i+2).name;
        img_name=['./DUT-train/DUT-train-Image/' imgname];
        I = imread(img_name);
        I = imresize(I,[384,384]);
        %imshow(I)
        imwrite(I,img_name,'jpg');
 end
% in_dir=dir('./DUT-train/DUT-train-Mask');
% imgnum=length(in_dir);
%  for i=1:imgnum-2
%         i
%         imgname=in_dir(i+2).name;
%         img_name=['./DUT-train/DUT-train-Mask/' imgname];
%         I = imread(img_name);
%         I = imresize(I,[384,384]);
%         %imshow(I*255)
%         imwrite(I,img_name,'png');
%  end