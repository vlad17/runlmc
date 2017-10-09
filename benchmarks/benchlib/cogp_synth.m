% copied from various parts of COGP paper code
% original repo: https://github.com/trungngv/cogp
% repo with a argument count bug corrected:
% https://github.com/vlad17/cogp

% Uses similar assumptions to cogp_fx2007: datadir and runs required
% datadir should go the csv directory
% requires maxiter, M, runs and datadir to be defined

addpath(genpath([tempdir 'cogp']));
format long;

x = csvread([datadir 'xss.csv']);
y = csvread([datadir 'yss.csv']);

% test on upper quadrant only, for last output
nlast = sum(~isnan(y(:, end)));
sel = all(x>=0.5, 2);
sel(1:end - nlast + 1) = 0;
ytest = y;
xtest = x;
y(sel, end) = nan;

rng(1234,'twister');

[y,ymean,ystd] = standardize(y,[],[]);

cf.covfunc_g  = 'covSEard';
cf.covfunc_h = 'covSEard';
cf.lrate      = 1e-2;
cf.momentum   = 0.9;
cf.lrate_hyp  = 1e-5;
cf.lrate_beta = 1e-4;
cf.momentum_w = 0.9;
cf.lrate_w    = 1e-6;
cf.learn_z    = true;
cf.momentum_z = 0.0;
cf.lrate_z    = 1e-4;
cf.maxiter = maxiter;
cf.nbatch = nbatch;
cf.beta = 1/0.1;
cf.initz = 'random';
cf.w = ones(size(y,2),2);
cf.monitor_elbo = 50;
cf.fix_first = true;
Q = 2;

smses = zeros(runs,1);
MM.g = M; MM.h = M;
nlpds = zeros(runs,1);
times = zeros(runs,1);
for r=1:runs
  par.g = cell(Q,1); par.task = cell(size(y,2),1);
  tic;
  [elbo,par] = slfm_learn(x,y,MM,par,cf);
  times(r) = toc;
  [mu,vaar,mu_g,var_g] = slfm_predict(cf.covfunc_g, cf.covfunc_h,par,xtest(sel,:));
  mu = mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);
  fvar = vaar.*repmat(ystd.^2,size(mu,1),1);
  smses(r) = mysmse(ytest(sel,end),mu(:,end),ymean(end));
  nlpds(r) = mynlpd(ytest(sel,end),mu(:,end),fvar(:,end));  
end
disp('mean/stderr times')
disp(mean(times))
disp(std(times) / sqrt(length(times)))
disp('mean/stderr smses')
disp(mean(smses))
disp(std(smses) / sqrt(length(smses)))
disp('mean/stderr nlpds')
disp(mean(nlpds))
disp(std(nlpds) / sqrt(length(nlpds)))
