% See cogp_fx2007.m for the assumptions this file makes

addpath(genpath([tempdir 'cogp']));
format long;
y = csvread('../data/fx/fx33k_train.csv');
y = 1./y; % usd / currency
y(y == -1) = nan; % missing data
x = (1:size(y,1))';
nout = size(y, 2);
xtest = cell(nout, 1);
ytest = cell(nout, 1);
for i = 1:nout
    % subtract 1 because python is 0 indexed
    xtest{i} = csvread(['../data/fx/fx33k_test/x', num2str(i - 1)]);
    ytest{i} = csvread(['../data/fx/fx33k_test/y', num2str(i - 1)]);
end

rng(1234,'twister');

[y,ymean,ystd] = standardize(y,[],[]);

cf.covfunc_g  = 'covSEard';
cf.lrate      = 1e-2;
cf.momentum   = 0.9;
cf.lrate_hyp  = 1e-5;
cf.lrate_beta = 1e-4;
cf.momentum_w = 0.9;
cf.lrate_w    = 1e-5;
cf.learn_z    = false;
cf.momentum_z = 0.0;
cf.lrate_z    = 1e-4;
cf.maxiter = MAXIT;
cf.nbatch = 200;
cf.beta = 1/0.1;
cf.initz = 'random';
cf.w = ones(size(y,2),Q);
cf.monitor_elbo = 50;
cf.fix_first = false;
%M = 100; assumed defined!

smses = zeros(runs,1);
nlpds = zeros(runs,1);
times = zeros(runs,1);
for r=1:runs % runs assumed defined!
  par.g = cell(Q,1);
  tic;
  [elbo,par] = slfm2_learn(x,y,M,par,cf);
  times(r) = toc;
  [mu,vaar,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,x,size(y,2));
  mu = mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);
  fvar = vaar.*repmat(ystd.^2,size(mu,1),1);
  per_out_smses = zeros(nout, 1);
  per_out_nlpds = zeros(nout, 1);
  for i=1:nout
    per_out_smses(i) = mysmse(ytest{i},mu(xtest{i},i),ymean(i));
    per_out_nlpds(i) = mynlpd(ytest{i},mu(xtest{i},i),fvar(xtest{i},i));
  end
  smses(r) = mean(per_out_smses);
  nlpds(r) = mean(per_out_nlpds);
  if r == 1
    csvwrite([[tempdir 'cogp-fx33k-mu-q'], num2str(Q)], mu);
    csvwrite([[tempdir 'cogp-fx33k-var-q'], num2str(Q)], fvar);
  end
end
disp('mean times')
disp(mean(times))
disp('mean smses')
disp(mean(smses))
disp('mean nlpds')
disp(mean(nlpds))
