clear;
cur = cd;
addpath(genpath(cur));

% load training data
load('Data/Train_data.mat');

tic

% Filter input images and compute highpass images
npd = 16;
fltlmbd = 1;
[hl, hh] = lowpass(train_data, fltlmbd, npd);

% Construct initial dictionary
filter_size = 10;
filter_num = 32;
D0 = zeros(filter_size,filter_size,filter_num);
D0(:,:,:) = single(randn(filter_size,filter_size,filter_num));

% Set up cbpdndl parameters
lambda = 0.005;
DLOpts = [];
DLOpts.Verbose = 1;
DLOpts.MaxMainIter = 1000;
DLOpts.rho = 100 * lambda + 1;
DLOpts.sigma = size(train_data,3);
DLOpts.AutoRho = 1;
DLOpts.AutoRhoPeriod = 10;
DLOpts.AutoSigma = 1;
DLOpts.AutoSigmaPeriod = 10;
DLOpts.XRelaxParam = 1.8;
DLOpts.DRelaxParam = 1.8;

% Do dictionary learning
[D, X, DLOptsinf] = cbpdndl_gpu(D0, hh, lambda, DLOpts);
toc

% save filters
path1 = strcat('Filter/mayo_filtersize_',num2str(filter_size),'_num_',num2str(filter_num),'_lambda_',num2str(lambda),'.mat');
save(path1,'D');

% Display learned dictionary
figure;
imdisp(tiledict(D));

% Plot functional value evolution
figure;
plot(DLOptsinf.itstat(:,2));
xlabel('Iterations');
ylabel('Functional value');

