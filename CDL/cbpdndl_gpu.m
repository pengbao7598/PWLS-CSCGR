function [D, Y, DLOptsinf] = cbpdndl_gpu(D0, S, lambda, DLOpts)

% cbpdndl_gpu -- Convolutional BPDN Dictionary Learning (GPU version)
%
%         argmin_{x_m,d_m} (1/2) \sum_k ||\sum_m d_m * x_k,m - s_k||_2^2 +
%                           lambda \sum_k \sum_m ||x_k,m||_1
%
%         The solution is computed using Augmented Lagrangian methods
%         (see boyd-2010-distributed) with efficient solution of the
%         main linear systems (see wohlberg-2016-efficient).
%
% Usage:
%       [D, X, DLOptsinf] = cbpdndl_gpu(D0, S, lambda, DLOpts)
%
% Input:
%       D0          Initial dictionary
%       S           Input images
%       lambda      Regularization parameter
%       DLOpts         DLOptsions/algorithm parameters structure (see below)
%
% Output:
%       D           Dictionary filter set (3D array)
%       X           Coefficient maps (3D array)
%       DLOptsinf      Details of DLOptsimisation
%
%
% DLOptsions structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, functional value,
%                    data fidelity term, l1 regularisation term, and
%                    primal and dual residuals (see Sec. 3.3 of
%                    boyd-2010-distributed). The values of rho and sigma
%                    are also displayed if DLOptsions request that they are
%                    automatically adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStDLOptsol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStDLOptsol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   L1Weight         Weight array for L1 norm
%   Y0               Initial value for Y
%   U0               Initial value for U
%   G0               Initial value for G (overrides D0 if specified)
%   H0               Initial value for H
%   rho              Augmented Lagrangian penalty parameter
%   AutoRho          Flag determining whether rho is automatically updated
%                    (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoRhoPeriod    Iteration period on which rho is updated
%   RhoRsdlRatio     Primal/dual residual ratio in rho update test
%   RhoScaling       Multiplier applied to rho when updated
%   AutoRhoScaling   Flag determining whether RhoScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, RhoScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier
%   sigma            Augmented Lagrangian penalty parameter
%   AutoSigma        Flag determining whether sigma is automatically
%                    updated (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoSigmaPeriod  Iteration period on which sigma is updated
%   SigmaRsdlRatio   Primal/dual residual ratio in sigma update test
%   SigmaScaling     Multiplier applied to sigma when updated
%   AutoSigmaScaling Flag determining whether SigmaScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, SigmaScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier.
%   StdResiduals     Flag determining whether standard residual definitions
%                    (see Sec 3.3 of boyd-2010-distributed) are used instead
%                    of normalised residuals (see wohlberg-2015-adaptive)
%   XRelaxParam      Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed) for X update
%   DRelaxParam      Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed) for D update
%   LinSolve         Linear solver for main problem: 'SM' or 'CG'
%   MaxCGIter        Maximum CG iterations when using CG solver
%   CGTol            CG tolerance when using CG solver
%   CGTolAuto        Flag determining use of automatic CG tolerance
%   CGTolFactor      Factor by which primal residual is divided to obtain CG
%                    tolerance, when automatic tolerance is active
%   NoBndryCross     Flag indicating whether all solution coefficients
%                    corresponding to filters crossing the image boundary
%                    should be forced to zero.
%   DictFilterSizes  Array of size 2 x M where each column specifies the
%                    filter size (rows x columns) of the corresponding
%                    dictionary filter
%   ZeroMean         Force learned dictionary entries to be zero-mean
%
%
% Authors: Brendt Wohlberg <brendt@lanl.gov>
%          Ping-Keng Jao <jpk7656@gmail.com>
% Modified: 2015-12-18
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

gS = gpuArray(S);
gD = gpuArray(D0);
glambda = gpuArray(lambda);

if nargin < 4,
  DLOpts = [];
end
checkopt(DLOpts, defaultDLOptss([]));
DLOpts = defaultDLOptss(DLOpts);

% Set up status display for verbose operation
hstr = ['Itn   Fnc       DFid      l1        Cnstr     '...
        'r(X)      s(X)      r(D)      s(D) '];
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 84;
if DLOpts.AutoRho,
  hstr = [hstr '     rho  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if DLOpts.AutoSigma,
  hstr = [hstr '     sigma  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if DLOpts.Verbose && DLOpts.MaxMainIter > 0,
  disp(hstr);
  disp(char('-' * ones(1,nsep)));
end

% Collapsing of trailing singleton dimensions greatly complicates
% handling of both SMV and MMV cases. The simplest approach would be
% if S could always be reshaped to 4d, with dimensions consisting of
% image rows, image cols, a single dimensional placeholder for number
% of filters, and number of measurements, but in the single
% measurement case the third dimension is collapsed so that the array
% is only 3d.
if size(S,3) > 1,
  xsz = [size(S,1) size(S,2) size(D0,3) size(S,3)];
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  gS = gpuArray(reshape(gS, [size(S,1) size(S,2) 1 size(S,3)]));
else
  xsz = [size(S,1) size(S,2) size(D0,3) 1];
end
gNx = gpuArray(prod(xsz));
gNd = gpuArray(prod(xsz(1:2))*size(D0,3));
gcgt = gpuArray(DLOpts.CGTol);

% Dictionary size may be specified when learning multiscale
% dictionary
if isempty(DLOpts.DictFilterSizes),
  dsz = [size(D0,1) size(D0,2)];
else
  dsz = DLOpts.DictFilterSizes;
end

% Mean removal and normalisation projections
Pzmn = @(x) bsxfun(@minus, x, mean(mean(x,1),2));
Pnrm = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(x.^2, 1), 2)));

% Projection of filter to full image size and its transpose
% (zero-pad and crop respectively)
Pzp = @(x) zpad(x, xsz(1:2));
PzpT = @(x) bcrop(x, dsz);

% Projection of dictionary filters onto constraint set
if DLOpts.ZeroMean,
  Pcn = @(x) Pnrm(Pzp(Pzmn(PzpT(x))));
else
  Pcn = @(x) Pnrm(Pzp(PzpT(x)));
end

% Start timer
tstart = tic;

% Project initial dictionary onto constraint set
% D = Pnrm(D0);
gD = Pnrm(gD);

% Compute signal in DFT domain
gSf = fft2(gS);

% Set up algorithm parameters and initialise variables
grho = gpuArray(DLOpts.rho);
if isempty(grho), grho = 50*glambda+1; end;
if DLOpts.AutoRho,
  asgr = DLOpts.RhoRsdlRatio;
  asgm = DLOpts.RhoScaling;
end
gsigma = gpuArray(DLOpts.sigma);
if isempty(gsigma), gsigma = gpuArray(size(S,3)); end;
if DLOpts.AutoSigma,
  asdr = DLOpts.SigmaRsdlRatio;
  asdm = DLOpts.SigmaScaling;
end
DLOptsinf = struct('itstat', [], 'DLOpts', DLOpts);
grx = gpuArray(Inf);
gsx = gpuArray(Inf);
grd = gpuArray(Inf);
gsd = gpuArray(Inf);
geprix = gpuArray(0);
geduax = gpuArray(0);
geprid = gpuArray(0);
geduad = gpuArray(0);

% Initialise main working variables
% X = [];
if isempty(DLOpts.Y0),
  gY = gpuArray.zeros(xsz, class(S));
else
  gY = gpuArray(DLOpts.Y0);
end
gYprv = gY;
if isempty(DLOpts.U0),
  if isempty(DLOpts.Y0),
    gU = gpuArray.zeros(xsz, class(S));
  else
    gU = (glambda/grho)*sign(gY);
  end
else
  gU = gpuArray(DLOpts.U0);
end
% Df = [];
if isempty(DLOpts.G0),
  gG = Pzp(gD);
else
  gG = gpuArray(DLOpts.G0);
end
gGprv = gG;
if isempty(DLOpts.H0),
  if isempty(DLOpts.G0),
    gH = gpuArray.zeros(size(gG), class(S));
  else
    gH = gG;
  end
else
  gH = gpuArray(DLOpts.H0);
end
gGf = fft2(gG, size(S,1), size(S,2));
gGSf = bsxfun(@times, conj(gGf), gSf);


% Main loop
k = 1;
while k <= DLOpts.MaxMainIter & (grx > geprix | gsx > geduax | ...
                              grd > geprid | gsd >geduad),

  % Solve X subproblem. It would be simpler and more efficient (since the
  % DFT is already available) to solve for X using the main dictionary
  % variable D as the dictionary, but this appears to be unstable. Instead,
  % use the projected dictionary variable G
  gXf = solvedbi_sm(gGf, grho, gGSf + grho*fft2(gY - gU));
  gX = ifft2(gXf, 'symmetric');
  clear gXf gGf gGSf;

  % See pg. 21 of boyd-2010-distributed
  if DLOpts.XRelaxParam == 1,
    gXr = gX;
  else
    gXr = DLOpts.XRelaxParam*gX + (1-DLOpts.XRelaxParam)*gY;
  end

  % Solve Y subproblem
  gY = shrink(gXr + gU, (glambda/grho)*DLOpts.L1Weight);
  if DLOpts.NoBndryCross,
    gY((end-size(gD,1)+2):end,:,:,:) = 0;
    gY(:,(end-size(gD,1)+2):end,:,:) = 0;
  end
  gYf = fft2(gY);
  gYSf = sum(bsxfun(@times, conj(gYf), gSf), 4);

  % Update dual variable corresponding to X, Y
  gU = gU + gXr - gY;
  clear gXr;

  % Compute primal and dual residuals and stopping thresholds for X update
  gnX = norm(gX(:)); gnY = norm(gY(:)); gnU = norm(gU(:));
  if DLOpts.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    grx = norm(vec(gX - gY));
    gsx = norm(vec(grho*(gYprv - gY)));
    geprix = sqrt(gNx)*DLOpts.AbsStDLOptsol+max(gnX,gnY)*DLOpts.RelStDLOptsol;
    geduax = sqrt(gNx)*DLOpts.AbsStDLOptsol+grho*gnU*DLOpts.RelStDLOptsol;
  else
    % See wohlberg-2015-adaptive
    grx = norm(vec(gX - gY))/max(gnX,gnY);
    gsx = norm(vec(gYprv - gY))/gnU;
    geprix = sqrt(gNx)*DLOpts.AbsStDLOptsol/max(gnX,gnY)+DLOpts.RelStDLOptsol;
    geduax = sqrt(gNx)*DLOpts.AbsStDLOptsol/(grho*gnU)+DLOpts.RelStDLOptsol;
  end
  clear gX;

  % Compute l1 norm of Y
  gJl1 = sum(abs(vec(DLOpts.L1Weight .* gY)));

  % Update record of previous step Y
  gYprv = gY;


  % Solve D subproblem. Similarly, it would be simpler and more efficient to
  % solve for D using the main coefficient variable X as the coefficients,
  % but it appears to be more stable to use the shrunk coefficient variable Y
  if strcmp(DLOpts.LinSolve, 'SM'),
    gDf = solvemdbi_ism_gpu(gYf, gsigma, gYSf + gsigma*fft2(gG - gH));
  else
    [gDf, gcgst] = solvemdbi_cg(gYf, gsigma, gYSf + gsigma*fft2(gG - gH), ...
                              gcgt, DLOpts.MaxCGIter, gDf(:));
  end
  clear YSf;
  gD = ifft2(gDf, 'symmetric');
  if strcmp(DLOpts.LinSolve, 'SM'), clear gDf; end

  % See pg. 21 of boyd-2010-distributed
  if DLOpts.DRelaxParam == 1,
    gDr = gD;
  else
    gDr = DLOpts.DRelaxParam*gD + (1-DLOpts.DRelaxParam)*gG;
  end

  % Solve G subproblem
  gG = Pcn(gDr + gH);
  gGf = fft2(gG);
  gGSf = bsxfun(@times, conj(gGf), gSf);

  % Update dual variable corresponding to D, G
  gH = gH + gDr - gG;
  clear gDr;

  % Compute primal and dual residuals and stopping thresholds for D update
  gnD = norm(gD(:)); gnG = norm(gG(:)); gnH = norm(gH(:));
  if DLOpts.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    grd = norm(vec(gD - gG));
    gsd = norm(vec(sigma*(gGprv - gG)));
    geprid = sqrt(gNd)*DLOpts.AbsStDLOptsol+max(gnD,gnG)*DLOpts.RelStDLOptsol;
    geduad = sqrt(gNd)*DLOpts.AbsStDLOptsol+gsigma*gnH*DLOpts.RelStDLOptsol;
  else
    % See wohlberg-2015-adaptive
    grd = norm(vec(gD - gG))/max(gnD,gnG);
    gsd = norm(vec(gGprv - gG))/gnH;
    geprid = sqrt(gNd)*DLOpts.AbsStDLOptsol/max(gnD,gnG)+DLOpts.RelStDLOptsol;
    geduad = sqrt(gNd)*DLOpts.AbsStDLOptsol/(gsigma*gnH)+DLOpts.RelStDLOptsol;
  end

  % Apply CG auto tolerance policy if enabled
  if DLOpts.CGTolAuto && (grd/DLOpts.CGTolFactor) < gcgt,
    gcgt = grd/DLOpts.CGTolFactor;
  end

  % Compute measure of D constraint violation
  gJcn = norm(vec(Pcn(gD) - gD));
  clear gD;

  % Update record of previous step G
  gGprv = gG;


  % Compute data fidelity term in Fourier domain (note normalisation)
  gJdf = sum(vec(abs(sum(bsxfun(@times,gGf,gYf),3)-gSf).^2))/(2*xsz(1)*xsz(2));
  clear gYf;
  gJfn = gJdf + glambda*gJl1;


  % Record and display iteration details
  tk = toc(tstart);
  DLOptsinf.itstat = [DLOptsinf.itstat;...
       [k gather(gJfn) gather(gJdf) gather(gJl1) gather(grx) gather(gsx)...
       gather(grd) gather(gsd) gather(geprix) gather(geduax) gather(geprid)...
       gather(geduad) gather(grho) gather(gsigma) tk]];
  if DLOpts.Verbose,
    dvc = [k, gather(gJfn), gather(gJdf), gather(gJl1) gather(gJcn), ...
           gather(grx), gather(gsx), gather(grd), gather(gsd)];
    if DLOpts.AutoRho,
      dvc = [dvc gather(grho)];
    end
    if DLOpts.AutoSigma,
      dvc = [dvc gather(gsigma)];
    end
    disp(sprintf(sfms, dvc));
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if DLOpts.AutoRho,
    if k ~= 1 && mod(k, DLOpts.AutoRhoPeriod) == 0,
      if DLOpts.AutoRhoScaling,
        grhomlt = sqrt(grx/gsx);
        if grhomlt < 1, grhomlt = 1/grhomlt; end
        if grhomlt > DLOpts.RhoScaling, grhomlt = gpuArray(DLOpts.RhoScaling); end
      else
        grhomlt = gpuArray(DLOpts.RhoScaling);
      end
      grsf = 1;
      if grx > DLOpts.RhoRsdlRatio*gsx, grsf = grhomlt; end
      if gsx > DLOpts.RhoRsdlRatio*grx, grsf = 1/grhomlt; end
      grho = grsf*grho;
      gU = gU/grsf;
    end
  end
  if DLOpts.AutoSigma,
    if k ~= 1 && mod(k, DLOpts.AutoSigmaPeriod) == 0,
      if DLOpts.AutoSigmaScaling,
        gsigmlt = sqrt(grd/gsd);
        if gsigmlt < 1, gsigmlt = 1/gsigmlt; end
        if gsigmlt > DLOpts.SigmaScaling, gsigmlt = gpuArray(DLOpts.SigmaScaling); end
      else
        gsigmlt = gpuArray(DLOpts.SigmaScaling);
      end
      gssf = gpuArray(1);
      if grd > DLOpts.SigmaRsdlRatio*gsd, gssf = gsigmlt; end
      if gsd > DLOpts.SigmaRsdlRatio*grd, gssf = 1/gsigmlt; end
      gsigma = gssf*gsigma;
      gH = gH/gssf;
    end
  end


  k = k + 1;

end

gD = PzpT(gG);

% Record run time and working variables
DLOptsinf.runtime = toc(tstart);
DLOptsinf.Y = gather(gY);
DLOptsinf.U = gather(gU);
DLOptsinf.G = gather(gG);
DLOptsinf.H = gather(gH);
DLOptsinf.lambda = gather(glambda);
DLOptsinf.rho = gather(grho);
DLOptsinf.sigma = gather(gsigma);
DLOptsinf.cgt = gather(gcgt);
if exist('gcgst'), DLOptsinf.cgst = gather(gcgst); end
D = gather(gD);
Y = DLOptsinf.Y;
if DLOpts.Verbose && DLOpts.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = shrink(v, lambda)

  u = sign(v).*max(0, abs(v) - lambda);

return


function u = zpad(v, sz)

%   u = zeros(sz(1), sz(2), size(v,3), size(v,4), class(v));
  u = gpuArray.zeros(sz(1), sz(2), size(v,3), size(v,4));
  u(1:size(v,1), 1:size(v,2),:,:) = v;

return


function u = bcrop(v, sz)

  if numel(sz) <= 2,
    if numel(sz) == 1
      cs = [sz sz];
    else
      cs = sz;
    end
    u = v(1:cs(1), 1:cs(2), :);
  else
    if size(sz,1) < size(sz,2), sz = sz'; end
    cs = max(sz);
%     u = zeros(cs(1), cs(2), size(v,3), class(v));
    u = gpuArray.zeros(cs(1), cs(2), size(v,3));
    for k = 1:size(v,3),
      u(1:sz(k,1), 1:sz(k,2), k) = v(1:sz(k,1), 1:sz(k,2), k);
    end
  end

return


function DLOpts = defaultDLOptss(DLOpts)

  if ~isfield(DLOpts,'Verbose'),
    DLOpts.Verbose = 0;
  end
  if ~isfield(DLOpts,'MaxMainIter'),
    DLOpts.MaxMainIter = 1000;
  end
  if ~isfield(DLOpts,'AbsStDLOptsol'),
    DLOpts.AbsStDLOptsol = 1e-6;
  end
  if ~isfield(DLOpts,'RelStDLOptsol'),
    DLOpts.RelStDLOptsol = 1e-4;
  end
  if ~isfield(DLOpts,'L1Weight'),
    DLOpts.L1Weight = 1;
  end
  if ~isfield(DLOpts,'Y0'),
    DLOpts.Y0 = [];
  end
  if ~isfield(DLOpts,'U0'),
    DLOpts.U0 = [];
  end
  if ~isfield(DLOpts,'G0'),
    DLOpts.G0 = [];
  end
  if ~isfield(DLOpts,'H0'),
    DLOpts.H0 = [];
  end
  if ~isfield(DLOpts,'rho'),
    DLOpts.rho = [];
  end
  if ~isfield(DLOpts,'AutoRho'),
    DLOpts.AutoRho = 0;
  end
  if ~isfield(DLOpts,'AutoRhoPeriod'),
    DLOpts.AutoRhoPeriod = 10;
  end
  if ~isfield(DLOpts,'RhoRsdlRatio'),
    DLOpts.RhoRsdlRatio = 10;
  end
  if ~isfield(DLOpts,'RhoScaling'),
    DLOpts.RhoScaling = 2;
  end
  if ~isfield(DLOpts,'AutoRhoScaling'),
    DLOpts.AutoRhoScaling = 0;
  end
  if ~isfield(DLOpts,'sigma'),
    DLOpts.sigma = [];
  end
  if ~isfield(DLOpts,'AutoSigma'),
    DLOpts.AutoSigma = 0;
  end
  if ~isfield(DLOpts,'AutoSigmaPeriod'),
    DLOpts.AutoSigmaPeriod = 10;
  end
  if ~isfield(DLOpts,'SigmaRsdlRatio'),
    DLOpts.SigmaRsdlRatio = 10;
  end
  if ~isfield(DLOpts,'SigmaScaling'),
    DLOpts.SigmaScaling = 2;
  end
  if ~isfield(DLOpts,'AutoSigmaScaling'),
    DLOpts.AutoSigmaScaling = 0;
  end
  if ~isfield(DLOpts,'StdResiduals'),
    DLOpts.StdResiduals = 0;
  end
  if ~isfield(DLOpts,'XRelaxParam'),
    DLOpts.XRelaxParam = 1;
  end
  if ~isfield(DLOpts,'DRelaxParam'),
    DLOpts.DRelaxParam = 1;
  end
  if ~isfield(DLOpts,'LinSolve'),
    DLOpts.LinSolve = 'SM';
  end
  if ~isfield(DLOpts,'MaxCGIter'),
    DLOpts.MaxCGIter = 1000;
  end
  if ~isfield(DLOpts,'CGTol'),
    DLOpts.CGTol = 1e-3;
  end
  if ~isfield(DLOpts,'CGTolAuto'),
    DLOpts.CGTolAuto = 0;
  end
  if ~isfield(DLOpts,'CGTolAutoFactor'),
    DLOpts.CGTolFactor = 50;
  end
  if ~isfield(DLOpts,'NoBndryCross'),
    DLOpts.NoBndryCross = 0;
  end
  if ~isfield(DLOpts,'DictFilterSizes'),
    DLOpts.DictFilterSizes = [];
  end
  if ~isfield(DLOpts,'ZeroMean'),
    DLOpts.ZeroMean = 0;
  end

return
