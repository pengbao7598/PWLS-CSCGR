function [D, Y, DLOptsinf] = cbpdndl(D0, S, lambda, DLOpts)

% cbpdndl -- Convolutional BPDN Dictionary Learning
%
%         argmin_{x_m,d_m} (1/2) \sum_k ||\sum_m d_m * x_k,m - s_k||_2^2 +
%                           lambda \sum_k \sum_m ||x_k,m||_1
%
%         The solution is computed using Augmented Lagrangian methods
%         (see boyd-2010-distributed) with efficient solution of the
%         main linear systems (see wohlberg-2016-efficient).
%
% Usage:
%       [D, Y, DLOptsinf] = cbpdndl(D0, S, lambda, DLOpts)
%
% Input:
%       D0          Initial dictionary
%       S           Input images
%       lambda      Regularization parameter
%       DLOpts         DLOptsions/algorithm parameters structure (see below)
%
% Output:
%       D           Dictionary filter set (3D array)
%       X           Coefficient maps (4D array)
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
%   NonNegCoef       Flag indicating whether solution should be forced to
%                    be non-negative
%   ZeroMean         Force learned dictionary entries to be zero-mean
%
%
% Author: Brendt Wohlberg <brendt@lanl.gov>  Modified: 2015-12-18
%
% This file is part of the SPORCO library. Details of the copyright
% and user license can be found in the 'License' file distributed with
% the library.


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
  S = reshape(S, [size(S,1) size(S,2) 1 size(S,3)]);
else
  xsz = [size(S,1) size(S,2) size(D0,3) 1];
end
Nx = prod(xsz);
Nd = prod(xsz(1:2))*size(D0,3);
cgt = DLOpts.CGTol;

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
D = Pnrm(D0);

% Compute signal in DFT domain
Sf = fft2(S);

% Set up algorithm parameters and initialise variables
rho = DLOpts.rho;
if isempty(rho), rho = 50*lambda+1; end;
if DLOpts.AutoRho,
  asgr = DLOpts.RhoRsdlRatio;
  asgm = DLOpts.RhoScaling;
end
sigma = DLOpts.sigma;
if isempty(sigma), sigma = size(S,3); end;
if DLOpts.AutoSigma,
  asdr = DLOpts.SigmaRsdlRatio;
  asdm = DLOpts.SigmaScaling;
end
DLOptsinf = struct('itstat', [], 'DLOpts', DLOpts);
rx = Inf;
sx = Inf;
rd = Inf;
sd = Inf;
eprix = 0;
eduax = 0;
eprid = 0;
eduad = 0;

% Initialise main working variables
X = [];
if isempty(DLOpts.Y0),
  Y = zeros(xsz, class(S));
else
  Y = DLOpts.Y0;
end
Yprv = Y;
if isempty(DLOpts.U0),
  if isempty(DLOpts.Y0),
    U = zeros(xsz, class(S));
  else
    U = (lambda/rho)*sign(Y);
  end
else
  U = DLOpts.U0;
end
Df = [];
if isempty(DLOpts.G0),
  G = Pzp(D);
else
  G = DLOpts.G0;
end
Gprv = G;
if isempty(DLOpts.H0),
  if isempty(DLOpts.G0),
    H = zeros(size(G), class(S));
  else
    H = G;
  end
else
  H = DLOpts.H0;
end
Gf = fft2(G, size(S,1), size(S,2));
GSf = bsxfun(@times, conj(Gf), Sf);


% Main loop
k = 1;
while k <= DLOpts.MaxMainIter && (rx > eprix|sx > eduax|rd > eprid|sd >eduad),

  % Solve X subproblem. It would be simpler and more efficient (since the
  % DFT is already available) to solve for X using the main dictionary
  % variable D as the dictionary, but this appears to be unstable. Instead,
  % use the projected dictionary variable G
  Xf = solvedbi_sm(Gf, rho, GSf + rho*fft2(Y - U));
  X = ifft2(Xf, 'symmetric');
  clear Xf Gf GSf;

  % See pg. 21 of boyd-2010-distributed
  if DLOpts.XRelaxParam == 1,
    Xr = X;
  else
    Xr = DLOpts.XRelaxParam*X + (1-DLOpts.XRelaxParam)*Y;
  end

  % Solve Y subproblem
  Y = shrink(Xr + U, (lambda/rho)*DLOpts.L1Weight);
  if DLOpts.NonNegCoef,
    Y(Y < 0) = 0;
  end
  if DLOpts.NoBndryCross,
    %Y((end-max(dsz(1,:))+2):end,:,:,:) = 0;
    Y((end-size(D0,1)+2):end,:,:,:) = 0;
    %Y(:,(end-max(dsz(2,:))+2):end,:,:) = 0;
    Y(:,(end-size(D0,2)+2):end,:,:) = 0;
  end
  Yf = fft2(Y);
  YSf = sum(bsxfun(@times, conj(Yf), Sf), 4);

  % Update dual variable corresponding to X, Y
  U = U + Xr - Y;
  clear Xr;

  % Compute primal and dual residuals and stopping thresholds for X update
  nX = norm(X(:)); nY = norm(Y(:)); nU = norm(U(:));
  if DLOpts.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    rx = norm(vec(X - Y));
    sx = norm(vec(rho*(Yprv - Y)));
    eprix = sqrt(Nx)*DLOpts.AbsStDLOptsol+max(nX,nY)*DLOpts.RelStDLOptsol;
    eduax = sqrt(Nx)*DLOpts.AbsStDLOptsol+rho*nU*DLOpts.RelStDLOptsol;
  else
    % See wohlberg-2015-adaptive
    rx = norm(vec(X - Y))/max(nX,nY);
    sx = norm(vec(Yprv - Y))/nU;
    eprix = sqrt(Nx)*DLOpts.AbsStDLOptsol/max(nX,nY)+DLOpts.RelStDLOptsol;
    eduax = sqrt(Nx)*DLOpts.AbsStDLOptsol/(rho*nU)+DLOpts.RelStDLOptsol;
  end
  clear X;

  % Compute l1 norm of Y
  Jl1 = sum(abs(vec(bsxfun(@times, DLOpts.L1Weight, Y))));

  % Update record of previous step Y
  Yprv = Y;


  % Solve D subproblem. Similarly, it would be simpler and more efficient to
  % solve for D using the main coefficient variable X as the coefficients,
  % but it appears to be more stable to use the shrunk coefficient variable Y
  if strcmp(DLOpts.LinSolve, 'SM'),
    Df = solvemdbi_ism(Yf, sigma, YSf + sigma*fft2(G - H));
  else
    [Df, cgst] = solvemdbi_cg(Yf, sigma, YSf + sigma*fft2(G - H), ...
                              cgt, DLOpts.MaxCGIter, Df(:));
  end
  clear YSf;
  D = ifft2(Df, 'symmetric');
  if strcmp(DLOpts.LinSolve, 'SM'), 
      clear Df; 
  end

  % See pg. 21 of boyd-2010-distributed
  if DLOpts.DRelaxParam == 1,
    Dr = D;
  else
    Dr = DLOpts.DRelaxParam*D + (1-DLOpts.DRelaxParam)*G;
  end

  % Solve G subproblem
  G = Pcn(Dr + H);
  Gf = fft2(G);
  GSf = bsxfun(@times, conj(Gf), Sf);

  % Update dual variable corresponding to D, G
  H = H + Dr - G;
  clear Dr;

  % Compute primal and dual residuals and stopping thresholds for D update
  nD = norm(D(:)); nG = norm(G(:)); nH = norm(H(:));
  if DLOpts.StdResiduals,
    % See pp. 19-20 of boyd-2010-distributed
    rd = norm(vec(D - G));
    sd = norm(vec(sigma*(Gprv - G)));
    eprid = sqrt(Nd)*DLOpts.AbsStDLOptsol+max(nD,nG)*DLOpts.RelStDLOptsol;
    eduad = sqrt(Nd)*DLOpts.AbsStDLOptsol+sigma*nH*DLOpts.RelStDLOptsol;
  else
    % See wohlberg-2015-adaptive
    rd = norm(vec(D - G))/max(nD,nG);
    sd = norm(vec(Gprv - G))/nH;
    eprid = sqrt(Nd)*DLOpts.AbsStDLOptsol/max(nD,nG)+DLOpts.RelStDLOptsol;
    eduad = sqrt(Nd)*DLOpts.AbsStDLOptsol/(sigma*nH)+DLOpts.RelStDLOptsol;
  end

  % Apply CG auto tolerance policy if enabled
  if DLOpts.CGTolAuto && (rd/DLOpts.CGTolFactor) < cgt,
    cgt = rd/DLOpts.CGTolFactor;
  end

  % Compute measure of D constraint violation
  Jcn = norm(vec(Pcn(D) - D));
  clear D;

  % Update record of previous step G
  Gprv = G;


  % Compute data fidelity term in Fourier domain (note normalisation)
  Jdf = sum(vec(abs(sum(bsxfun(@times,Gf,Yf),3)-Sf).^2))/(2*xsz(1)*xsz(2));
  clear Yf;
  Jfn = Jdf + lambda*Jl1;


  % Record and display iteration details
  tk = toc(tstart);
  DLOptsinf.itstat = [DLOptsinf.itstat;...
       [k Jfn Jdf Jl1 rx sx rd sd eprix eduax eprid eduad rho sigma tk]];
  if DLOpts.Verbose,
    dvc = [k, Jfn, Jdf, Jl1, Jcn, rx, sx, rd, sd];
    if DLOpts.AutoRho,
      dvc = [dvc rho];
    end
    if DLOpts.AutoSigma,
      dvc = [dvc sigma];
    end
    disp(sprintf(sfms, dvc));
  end

  % See wohlberg-2015-adaptive and pp. 20-21 of boyd-2010-distributed
  if DLOpts.AutoRho,
    if k ~= 1 && mod(k, DLOpts.AutoRhoPeriod) == 0,
      if DLOpts.AutoRhoScaling,
        rhomlt = sqrt(rx/sx);
        if rhomlt < 1, rhomlt = 1/rhomlt; end
        if rhomlt > DLOpts.RhoScaling, rhomlt = DLOpts.RhoScaling; end
      else
        rhomlt = DLOpts.RhoScaling;
      end
      rsf = 1;
      if rx > DLOpts.RhoRsdlRatio*sx, rsf = rhomlt; end
      if sx > DLOpts.RhoRsdlRatio*rx, rsf = 1/rhomlt; end
      rho = rsf*rho;
      U = U/rsf;
    end
  end
  if DLOpts.AutoSigma,
    if k ~= 1 && mod(k, DLOpts.AutoSigmaPeriod) == 0,
      if DLOpts.AutoSigmaScaling,
        sigmlt = sqrt(rd/sd);
        if sigmlt < 1, sigmlt = 1/sigmlt; end
        if sigmlt > DLOpts.SigmaScaling, sigmlt = DLOpts.SigmaScaling; end
      else
        sigmlt = DLOpts.SigmaScaling;
      end
      ssf = 1;
      if rd > DLOpts.SigmaRsdlRatio*sd, ssf = sigmlt; end
      if sd > DLOpts.SigmaRsdlRatio*rd, ssf = 1/sigmlt; end
      sigma = ssf*sigma;
      H = H/ssf;
    end
  end


  k = k + 1;

end

D = PzpT(G);

% Record run time and working variables
DLOptsinf.runtime = toc(tstart);
DLOptsinf.Y = Y;
DLOptsinf.U = U;
DLOptsinf.G = G;
DLOptsinf.H = H;
DLOptsinf.lambda = lambda;
DLOptsinf.rho = rho;
DLOptsinf.sigma = sigma;
DLOptsinf.cgt = cgt;
if exist('cgst'), DLOptsinf.cgst = cgst; end

if DLOpts.Verbose && DLOpts.MaxMainIter > 0,
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


function u = zpad(v, sz)

  u = zeros(sz(1), sz(2), size(v,3), size(v,4), class(v));
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
    cs = max(sz,[],2);
    u = zeros(cs(1), cs(2), size(v,3), class(v));
    for k = 1:size(v,3),
      u(1:sz(1,k), 1:sz(2,k), k) = v(1:sz(1,k), 1:sz(2,k), k);
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
  if ~isfield(DLOpts,'NonNegCoef'),
    DLOpts.NonNegCoef = 0;
  end
  if ~isfield(DLOpts,'ZeroMean'),
    DLOpts.ZeroMean = 0;
  end

return
