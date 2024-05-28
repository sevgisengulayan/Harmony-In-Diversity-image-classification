%clear all
%close all
imds = imageDatastore("step1_data",'IncludeSubfolders',true, 'LabelSource','foldernames');
numTrainFiles = 0.80;

[imdsTrain2, imdsValidation2] = splitEachLabel(imds, numTrainFiles,'randomize');
inputSize = [224 224 3];
numClasses = unique(imds.Labels);


imdsTrain = augmentedImageDatastore(inputSize,imdsTrain2);

imdsValidation = augmentedImageDatastore(inputSize,imdsValidation2);


net=googlenet;

inputSize = net.Layers(1).InputSize

lgraph = layerGraph(net); 

numClasses = numel(categories(imdsTrain2.Labels))

newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);
    
lgraph = replaceLayer(lgraph,'loss3-classifier',newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',2, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(imdsTrain,lgraph,options);

save("netTransfer.mat", "netTransfer");
