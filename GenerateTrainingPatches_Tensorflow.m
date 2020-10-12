
%%% Generate the training data.

clear;close all;
clear all;


batchSize      = 128;        %%% batch size
dataName      = 'TrainingPatches_Tensorflow';
folder        = 'image';
% subfolder = {'T1','T2','T3','image04','image02','image06','image07','S2','S3','S4','T7',...
%     'T8','T9','T10','T11','T13'};
subfolder = {'T1','T2','T7','T8','T9','T10','T11','image04','image02','image06','image07'};
imagename = {'0.bmp','45.bmp','90.bmp','135.bmp'};
patchsize     = 40;
stride        = 10;
step          = 0;

count   = 0;



%% count the number of extracted patches
scales  = [1 0.9 0.8 0.7];
for i = 1 : length(subfolder)
    image = imread(fullfile(folder,subfolder{i},imagename{1})); % uint8
    if size(image,3)==3
        image = rgb2gray(image);
    end
    %[~, name, exte] = fileparts(filepaths(i).name); 
    for s = 1:4
        image_re = imresize(image,scales(s),'bicubic');
        [hei,wid,~] = size(image_re);
        for x = 1+step : stride : (hei-patchsize+1)
            for y = 1+step :stride : (wid-patchsize+1)
                count = count+1;
            end
        end
    end
end

numPatches = ceil(count/batchSize)*batchSize;

disp([numPatches,batchSize,numPatches/batchSize]);

%pause;

inputs  = zeros(numPatches,patchsize, patchsize, 2, 'single'); % this is fast



tic;
count = 0;
for i = 1 : length(subfolder)
        image_ori_0 =double(imread(fullfile(folder,subfolder{i},imagename{1}))); % uint8
        image_ori_45 = double(imread(fullfile(folder,subfolder{i},imagename{2}))); % uint8
        image_ori_90 = double(imread(fullfile(folder,subfolder{i},imagename{3}))); % uint8
        image_ori_135 = double(imread(fullfile(folder,subfolder{i},imagename{4}))); % uint8
        
        S0 = double(image_ori_0+image_ori_45+image_ori_90+image_ori_135)*0.5;
        S1 = double(image_ori_0-image_ori_90);
        S2 = double(image_ori_45-image_ori_135);
        DoLP = sqrt(S1.^2+S2.^2)./S0;
        for s = 1:4
            image_s0 = imresize(S0,scales(s),'bicubic');
            image_dolp = imresize(DoLP,scales(s),'bicubic');
            image_s0 = uint8(GrayStretch(image_s0));
            image_dolp =uint8(GrayStretch(image_dolp));
            
            im_s0  = im2single(image_s0); % single
            im_dolp  = im2single(image_dolp); % single
            [hei,wid,~] = size(im_s0);
            
            for x = 1+step : stride : (hei-patchsize+1)
                for y = 1+step :stride : (wid-patchsize+1)
                    count       = count+1
                    inputs(count,:, :, 1)   = im_s0(x : x+patchsize-1, y : y+patchsize-1,:);
                    inputs(count,:, :, 2)   = im_dolp(x : x+patchsize-1, y : y+patchsize-1,:);
                end
            end
        end
end
toc;

if count<numPatches
    pad = numPatches - count;
    inputs(count+1:end,:,:,:) = inputs(1:pad,:,:,:);
end

disp('-------Datasize-------')
disp([size(inputs,1),batchSize,size(inputs,1)/batchSize]);

if ~exist(dataName,'file')
    mkdir(dataName);
end

%%% save data
save(fullfile(dataName,['imdb_',num2str(patchsize),'_',num2str(batchSize)]), 'inputs','-v7.3')

