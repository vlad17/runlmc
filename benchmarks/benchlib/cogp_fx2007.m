% copied from various parts of COGP paper code
% original repo: https://github.com/trungngv/cogp
% repo with a argument count bug corrected:
% https://github.com/vlad17/cogp

% Make sure the vlad17 one is cloned in /tmp/cogp
% Make sure the csv is in /tmp/
% this is all done for you in benchmarks/financial_exchange.ipynb

% assume M,infile,runs are already defined

addpath(genpath([tempdir 'cogp']));
format long;

y = csvread(infile);
y = 1./y; % usd / currency
y0 = y;
x = (1:size(y,1))';
y(y == -1) = nan; % missing data

xtest = x;
ytest = y(xtest,:);

% imput missing data
% CAD = 4, JPY = 6, AUD = 9
y(50:100,4) = nan;
y(100:150,6) = nan;
y(150:200,9) = nan;

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
cf.maxiter = 500;
cf.nbatch = 200;
cf.beta = 1/0.1;
cf.initz = 'random';
cf.w = ones(size(y,2),2);
cf.monitor_elbo = 50;
cf.fix_first = false;
Q = 2;
xtest = cell(3,1);
xtest{1} = (50:100)';
xtest{2} = (100:150)';
xtest{3} = (150:200)';
outputs = [4,6,9];

smses = zeros(runs,1);
nlpds = zeros(runs,1);
times = zeros(runs,1);
for r=1:runs
  par.g = cell(Q,1);
  tic;
  [elbo,par] = slfm2_learn(x,y,M,par,cf);
  times(r) = toc;
  [mu,vaar,mu_g,var_g] = slfm2_predict(cf.covfunc_g,par,x,size(y,2));
  mu = mu.*repmat(ystd,size(mu,1),1) + repmat(ymean,size(mu,1),1);
  fvar = vaar.*repmat(ystd.^2,size(mu,1),1);
  per_out_smses = zeros(3, 1);
  per_out_nlpds = zeros(3, 1);
  csvwrite([tempdir 'cogp-fx2007-mu'], mu(:, outputs))
  csvwrite([tempdir 'cogp-fx2007-var'], fvar(:, outputs))
  for i=1:3
    t = outputs(i);
    per_out_smses(i) = mysmse(ytest(xtest{i},t),mu(xtest{i},t),ymean(t));
    per_out_nlpds(i) = mynlpd(ytest(xtest{i},t),mu(xtest{i},t),fvar(xtest{i},t));
  end
  smses(r) = mean(per_out_smses);
  nlpds(r) = mean(per_out_nlpds);
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
