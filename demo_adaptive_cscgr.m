% These MATLAB programs implement the Sparse-view CT reconstruction method as described in paper:
%   
% Title: Convolutional Sparse Coding for Compressed Sensing CT Reconsturection 
% Author: Peng Bao, Wenjun Xia, Kang Yang, Weiyan Chen, Mianyi Chen, Yan Xi,
%         Shanzhou Niu, Jiliu Zhou, He Zhang, Huaiqiang Sun, Zhangyang Wang, and Yi Zhang
%   
% -------------------------------------------------------------------------------------------------------
% The software implemented by MatLab 2018a are included in this package.

% ------------------------------------------------------------------
% Author: Peng Bao
% Email:  pengbao7598@gmail.com
% Last modified by Peng Bao, March 2019

clear
cur = cd;
addpath(genpath(cur));

tic

% load the test image
ImgNo = 1;
switch ImgNo
    case 1
        imageName = 'abdominal';
    case 2
        imageName = 'thoracic';
end
load(strcat('Test_Images/', imageName, '.mat'));
[rows, cols] = size(original_image);

% load the system matrix
img_size = rows;
pro_views = 64;
bins = 512;
sysmat = strcat('SysMat_ImgSize_', num2str(img_size), ...
    '_Views_', num2str(pro_views), ...
    '_Bins_', num2str(bins));
load(strcat('Data/', sysmat, '.mat'));

% get measurement;
proj = SystemMatrix * original_image(:);

% initialize filters (adaptive updating in each iteration)
filter_size = 10;
filter_num = 32;
filter = strcat('mayo_filtersize_',num2str(filter_size), ...
    '_num_',num2str(filter_num));
D = zeros(filter_size,filter_size,filter_num);
D(:,:,:) = single(randn(filter_size,filter_size,filter_num));
D_initial = D;
figure;
imdisp(tiledict(D));
    

% params of pwls
beta = 0.005;
pwls_iter = 20;
pwls = zeros(size(original_image));
reconstruction = zeros(size(original_image));

% params of cscgr
lambda = 0.005;
rho = 100 * lambda + 1;
tau = 0.06;
iter = 2000;

% create folder to save the results
filename = strcat(num2str(pro_views),'_alpha_',num2str(beta),'_lambda_',num2str(lambda),'_rho_',num2str(rho),'_mu_',num2str(tau));
path = strcat('Result\adaptive\',filter,'\',imageName,'\',filename);
if ~exist(path,'dir')
    mkdir(path);
end

% the file used to record the psnrs
if exist(strcat(path,'\result.txt'),'file')
    delete(strcat(path,'\result.txt'));
end
result = fopen(strcat(path,'\result.txt'),'a+');

pre_pwls = zeros(size(original_image));

for i = 1 : iter
    % pwls reconstruction
    if i < 40 % a little trick to accelerate the convergence
        pwls = split_hscg(reconstruction, proj,reconstruction, SystemMatrix, beta, pwls_iter);
    else
        pwls = split_hscg(pwls, proj, reconstruction, SystemMatrix, beta, pwls_iter);
    end
    pwls(pwls<0) = 0;
    fprintf('iter:%d, pwls = %f,', i, psnr(original_image,pwls));
    fprintf(result, 'iter:%d, pwls = %f,', i, psnr(original_image,pwls));
    
    % Highpass filter test image
    npd = 16;
    fltlmbd = 1;  
    [ll, lh] = lowpass(pwls, fltlmbd, npd);
    
    % update filters
    training_lambda = 0.005;
    DLOpts = [];
    DLOpts.Verbose = 0;
    DLOpts.MaxMainIter = 500;
    DLOpts.rho = 100 * training_lambda + 1;
    DLOpts.sigma = size(lh,3);
    DLOpts.AutoRho = 1;
    DLOpts.AutoRhoPeriod = 10;
    DLOpts.AutoSigma = 1;
    DLOpts.AutoSigmaPeriod = 10;
    DLOpts.XRelaxParam = 1.8;
    DLOpts.DRelaxParam = 1.8;
    
    [D, X, DLOptsinf] = cbpdndl_gpu(D, lh, training_lambda, DLOpts);
    if i == 1
        D_first = D;
        figure;
        imdisp(tiledict(D));
    end
    if i == 250
        D_middle = D;
        figure;
        imdisp(tiledict(D));
    end
    
    
    % Compute representation
    SCOpts = [];
    SCOpts.Verbose = 0;
    SCOpts.MaxMainIter = 300;
    SCOpts.rho = rho;
    SCOpts.AuxVarObj = 0;
    SCOpts.HighMemSolve = 1;
    
    [X, optinf2] = cbpdngr_gpu(D, lh, lambda, tau, SCOpts);
%     [X, optinf2] = cbpdn_gpu(D, lh, lambda, SCOpts); 
    DX = ifft2(sum(bsxfun(@times, fft2(D, size(X,1), size(X,2)), fft2(X)),3), ...
        'symmetric');
    
    reconstruction = double(DX+ll);
    reconstruction(reconstruction<0) = 0;
    fprintf('recon_psnr = %f\n',psnr(original_image,reconstruction));
    fprintf(result,'recon_psnr = %f\r\n',psnr(original_image,reconstruction));
    
    % stop criteria
    diff = (pre_pwls-pwls).^2;
    if sqrt(sum(diff(:))) < 5e-4
        break;
    end

    pre_pwls = pwls;
end

toc
fclose('all');

% show the results
switch ImgNo
    case 1
        disp_win = [850/3000 1250/3000];
    case 2
        disp_win = [0/3000 1250/3000];
end

figure;
subplot(1, 3, 1);
imshow(original_image, disp_win, 'border', 'tight');
subplot(1, 3, 2);
imshow(pwls,disp_win, 'border', 'tight');
subplot(1, 3, 3);
imshow(reconstruction,disp_win, 'border', 'tight');

% save the result
path1 = strcat(path, '/', pro_views, '_pwls_psnr_', num2str(psnr(original_image, pwls)));
path2 = strcat(path, '/',pro_views, '_csc_psnr_', num2str(psnr(original_image, reconstruction)));
save(strcat(path,'/',filter,'.mat'),'D');
save(strcat(path1, '.mat'), 'pwls');
save(strcat(path2, '.mat'), 'reconstruction');

disp('End of PWLS-CSCGR');

