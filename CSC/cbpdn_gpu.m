function [Y, SCOptsinf] = cbpdn_gpu(D, S, lambda, SCOpts)

% cbpdn_gpu -- Convolutional Basis Pursuit DeNoising (GPU version)
%
%         argmin_{x_m} (1/2)||\sum_m d_m * x_m - s||_2^2 +
%                           lambda \sum_m ||x_m||_1
%
%         The solution is computed using an ADMM approach (see
%         boyd-2010-distributed) with efficient solution of the main
%         linear systems (see wohlberg-2016-efficient).
%
% Usage:
%       [Y, SCOptsinf] = cbpdn_gpu(D, S, lambda, SCOpts);
%
% Input:
%       D           Dictionary filter set (3D array)
%       S           Input image
%       lambda      Regularization parameter
%       SCOpts         Algorithm parameters structure
%
% Output:
%       Y           Dictionary coefficient map set (3D array)
%       SCOptsinf      Details of SCOptsimisation
%
%
% SCOptsions structure fields:
%   Verbose          Flag determining whether iteration status is displayed.
%                    Fields are iteration number, functional value,
%                    data fidelity term, l1 regularisation term, and
%                    primal and dual residuals (see Sec. 3.3 of
%                    boyd-2010-distributed). The value of rho is also
%                    displayed if SCOptsions request that it is automatically
%                    adjusted.
%   MaxMainIter      Maximum main iterations
%   AbsStSCOptsol       Absolute convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   RelStSCOptsol       Relative convergence tolerance (see Sec. 3.3.1 of
%                    boyd-2010-distributed)
%   L1Weight         Weighting array for coefficients in l1 norm of X
%   Y0               Initial value for Y
%   U0               Initial value for U
%   rho              ADMM penalty parameter
%   AutoRho          Flag determining whether rho is automatically updated
%                    (see Sec. 3.4.1 of boyd-2010-distributed)
%   AutoRhoPeriod    Iteration period on which rho is updated
%   RhoRsdlRatio     Primal/dual residual ratio in rho update test
%   RhoScaling       Multiplier applied to rho when updated
%   AutoRhoScaling   Flag determining whether RhoScaling value is
%                    adaptively determined (see wohlberg-2015-adaptive). If
%                    enabled, RhoScaling specifies a maximum allowed
%                    multiplier instead of a fixed multiplier.
%   RhoRsdlTarget    Residual ratio targeted by auto rho update policy.
%   StdResiduals     Flag determining whether standard residual definitions
%                    (see Sec 3.3 of boyd-2010-distributed) are used instead
%                    of normalised residuals (see wohlberg-2015-adaptive)
%   RelaxParam       Relaxation parameter (see Sec. 3.4.3 of
%                    boyd-2010-distributed)
%   NoBndryCross     Flag indicating whether all solution coefficients
%                    corresponding to filters crossing the image boundary
%                    should be forced to zero.
%   AuxVarObj        Flag determining whether objective function is computed
%                    using the auxiliary (split) variable
%   HighMemSolve     Use more memory for a slightly faster solution
%
%
% Authors: Brendt Wohlberg <brendt@lanl.gov>
%          Ping-Keng Jao <jpk7656@gmail.com>
% Modified: 2015-12-28
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.

gS = gpuArray(S);
gD = gpuArray(D);
glambda = gpuArray(lambda);

if nargin < 4,
  SCOpts = [];
end
checkopt(SCOpts, defaultSCOptss([]));
SCOpts = defaultSCOptss(SCOpts);

% Set up status display for verbose operation
hstr = 'Itn   Fnc       DFid      l1        r         s      ';
sfms = '%4d %9.2e %9.2e %9.2e %9.2e %9.2e';
nsep = 54;
if SCOpts.AutoRho,
  hstr = [hstr '   rho  '];
  sfms = [sfms ' %9.2e'];
  nsep = nsep + 10;
end
if SCOpts.Verbose && SCOpts.MaxMainIter > 0,
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
if size(S,3) > 1 && size(S,4) == 1,
  xsz = [size(S,1) size(S,2) size(D,3) size(S,3)];
  % Insert singleton 3rd dimension (for number of filters) so that
  % 4th dimension is number of images in input s volume
  gS = reshape(gS, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [size(S,1) size(S,2) size(D,3) size(S,4)];
end

% Start timer
tstart = tic;

% Compute filters in DFT domain
gDf = fft2(gD, size(S,1), size(S,2));
% Convolve-sum and its Hermitian transpose
gDop = @(x) sum(bsxfun(@times, gDf, x), 3);
gDHop = @(x) bsxfun(@times, conj(gDf), x);
% Compute signal in DFT domain
gSf = fft2(gS);
% S convolved with all filters in DFT domain
gDSf = gDHop(gSf);

% Default lambda is 1/10 times the lambda value beyond which the
% solution is a zero vector
if nargin < 3 | isempty(lambda),
  gb = ifft2(DHop(gSf), 'symmetric');
  glambda = 0.1*max(vec(abs(gb)));
end

% Set up algorithm parameters and initialise variables
grho = gpuArray(SCOpts.rho);
if isempty(grho), grho = 50*glambda+1; end;
if isempty(SCOpts.RhoRsdlTarget),
  if SCOpts.StdResiduals,
    SCOpts.RhoRsdlTarget = 1;
  else
    SCOpts.RhoRsdlTarget = 1 + (18.3).^(log10(gather(glambda)) + 1);
  end
end
if SCOpts.HighMemSolve,
  gC = bsxfun(@rdivide, gDf, sum(gDf.*conj(gDf), 3) + grho);
else
  gC = [];
end
gNx = prod(gpuArray(xsz));
SCOptsinf = struct('itstat', [], 'SCOpts', SCOpts);
gr = gpuArray(Inf);
gs = gpuArray(Inf);
gepri = gpuArray(0);
gedua = gpuArray(0);

% Initialise main working variables
% X = [];
if isempty(SCOpts.Y0),
%   gY = zeros(xsz, 'gpuArray');
    gY = gpuArray.zeros(xsz);
else
  gY = gpuArray(SCOpts.Y0);
end
gYprv = gY;
if isempty(SCOpts.U0),
  if isempty(SCOpts.Y0),
%     gU = zeros(xsz, 'gpuArray');
    gU = gpuArray.zeros(xsz);
  else
    gU = (glambda/grho)*sign(gY);
  end
else
  gU = gpuArray(SCOpts.U0);
end

% Main loop
k = 1;
while k <= SCOpts.MaxMainIter & (gr > gepri | gs > gedua),

  % Solve X subproblem
  gXf = solvedbi_sm(gDf, grho, gDSf + grho*fft2(gY - gU), gC);
  gX = ifft2(gXf, 'symmetric');

  % See pg. 21 of boyd-2010-distributed
  if SCOpts.RelaxParam == 1,
    gXr = gX;
  else
    gXr = SCOpts.RelaxParam*gX + (1-SCOpts.RelaxParam)*gY;
  end

  % Solve Y subproblem
  gY = shrink(gXr + gU, (glambda/grho)*SCOpts.L1Weight);
  if SCOpts.NoBndryCross,
    gY((end-size(gD,1)+2):end,:,:,:) = 0;
    gY(:,(end-size(gD,2)+2):end,:,:) = 0;
  end

  % Update dual variable
  gU = gU + gXr - gY;

  % Compute data fidelity term in Fourier domain (note normalisation)
  if SCOpts.AuxVarObj,
    gYf = fft2(gY); % This represents unnecessary computational cost
    gJdf = sum(vec(abs(sum(bsxfun(@times,gDf,gYf),3)-gSf).^2)) / ...
           (2*xsz(1)*xsz(2));
    gJl1 = sum(abs(vec(bsxfun(@times, SCOpts.L1Weight, gY))));
  else
    gJdf = sum(vec(abs(sum(bsxfun(@times,gDf,gXf),3)-gSf).^2)) / ...
           (2*xsz(1)*xsz(2));
    gJl1 = sum(abs(vec(bsxfun(@times, SCOpts.L1Weight, gX))));
  end
  gJfn = gJdf + glambda*gJl1;

  gnX = norm(gX(:)); gnY = norm(gY(:)); gnU = norm(gU(:));
  if SCOpts.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    gr = norm(vec(gX - gY));
    gs = norm(vec(grho*(gYprv - gY)));
    gepri = sqrt(gNx)*SCOpts.AbsStSCOptsol+max(gnX,gnY)*SCOpts.RelStSCOptsol;
    gedua = sqrt(gNx)*SCOpts.AbsStSCOptsol+grho*gnU*SCOpts.RelStSCOptsol;
  else
    % See wohlberg-2015-adaptive
    gr = norm(vec(gX - gY))/max(gnX,gnY);
    gs = norm(vec(gYprv - gY))/gnU;
    gepri = sqrt(gNx)*SCOpts.AbsStSCOptsol/max(gnX,gnY)+SCOpts.RelStSCOptsol;
    gedua = sqrt(gNx)*SCOpts.AbsStSCOptsol/(grho*gnU)+SCOpts.RelStSCOptsol;
  end

  % Record and display iteration details
  tk = toc(tstart);
  SCOptsinf.itstat = [SCOptsinf.itstat; [k gather(gJfn) gather(gJdf) gather(gJl1) ...
                   gather(gr) gather(gs) gather(gepri) gather(gedua) grho tk]];
  if SCOpts.Verbose,
    if SCOpts.AutoRho,
      disp(sprintf(sfms, k, gather(gJfn), gather(gJdf), gather(gJl1), ...
                   gather(gr), gather(gs), grho));
    else
      disp(sprintf(sfms, k, gather(gJfn), gather(gJdf), gather(gJl1), ...
                   gather(gr), gather(gs)));
    end
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if SCOpts.AutoRho,
    if k ~= 1 && mod(k, SCOpts.AutoRhoPeriod) == 0,
      if SCOpts.AutoRhoScaling,
        grhomlt = sqrt(gr/(gs*SCOpts.RhoRsdlTarget));
        if grhomlt < 1, grhomlt = 1/grhomlt; end
        if grhomlt > SCOpts.RhoScaling, grhomlt = SCOpts.RhoScaling; end
      else
        grhomlt = SCOpts.RhoScaling;
      end
      grsf = 1;
      if gr > SCOpts.RhoRsdlTarget*SCOpts.RhoRsdlRatio*gs, grsf = grhomlt; end
      if gs > (SCOpts.RhoRsdlRatio/SCOpts.RhoRsdlTarget)*gr, grsf = 1/grhomlt; end
      grho = grsf*grho;
      gU = gU/grsf;
      if SCOpts.HighMemSolve && grsf ~= 1,
        gC = bsxfun(@rdivide, gDf, sum(gDf.*conj(gDf), 3) + grho);
%         gC = bsxfun(@rdivide, gDf, sum(gDf.*conj(gDf), 3) + grho*1.05);
      end
    end
  end

  gYprv = gY;
  k = k + 1;

end

% Record run time and working variables
SCOptsinf.runtime = toc(tstart);
SCOptsinf.X = gather(gX);
SCOptsinf.Xf = gather(gXf);
SCOptsinf.Y = gather(gY);
SCOptsinf.U = gather(gU);
SCOptsinf.lambda = gather(glambda);
SCOptsinf.rho = gather(grho);
Y = gather(gY);
% End status display for verbose operation
if SCOpts.Verbose && SCOpts.MaxMainIter > 0,
  disp(char('-' * ones(1,nsep)));
end

return


function u = vec(v)

  u = v(:);

return


function u = shrink(v, lambda)

  if isscalar(lambda),
    u = sign(v).*max(0, abs(v) - lambda);
  else
    u = sign(v).*max(0, bsxfun(@minus, abs(v), lambda));
  end

return


function SCOpts = defaultSCOptss(SCOpts)

  if ~isfield(SCOpts,'Verbose'),
    SCOpts.Verbose = 0;
  end
  if ~isfield(SCOpts,'MaxMainIter'),
    SCOpts.MaxMainIter = 1000;
  end
  if ~isfield(SCOpts,'AbsStSCOptsol'),
    SCOpts.AbsStSCOptsol = 0;
  end
  if ~isfield(SCOpts,'RelStSCOptsol'),
    SCOpts.RelStSCOptsol = 1e-4;
  end
  if ~isfield(SCOpts,'L1Weight'),
    SCOpts.L1Weight = 1;
  end
  if ~isfield(SCOpts,'Y0'),
    SCOpts.Y0 = [];
  end
  if ~isfield(SCOpts,'U0'),
    SCOpts.U0 = [];
  end
  if ~isfield(SCOpts,'rho'),
    SCOpts.rho = [];
  end
  if ~isfield(SCOpts,'AutoRho'),
    SCOpts.AutoRho = 1;
  end
  if ~isfield(SCOpts,'AutoRhoPeriod'),
    SCOpts.AutoRhoPeriod = 1;
  end
  if ~isfield(SCOpts,'RhoRsdlRatio'),
    SCOpts.RhoRsdlRatio = 1.2;
  end
  if ~isfield(SCOpts,'RhoScaling'),
    SCOpts.RhoScaling = 100;
  end
  if ~isfield(SCOpts,'AutoRhoScaling'),
    SCOpts.AutoRhoScaling = 1;
  end
  if ~isfield(SCOpts,'RhoRsdlTarget'),
    SCOpts.RhoRsdlTarget = [];
  end
  if ~isfield(SCOpts,'StdResiduals'),
    SCOpts.StdResiduals = 0;
  end
  if ~isfield(SCOpts,'RelaxParam'),
    SCOpts.RelaxParam = 1.8;
  end
  if ~isfield(SCOpts,'NoBndryCross'),
    SCOpts.NoBndryCross = 0;
  end
  if ~isfield(SCOpts,'AuxVarObj'),
    SCOpts.AuxVarObj = 0;
  end
  if ~isfield(SCOpts,'HighMemSolve'),
    SCOpts.HighMemSolve = 0;
  end

return
