% copied from various parts of COGP paper code
% original repo: https://github.com/trungngv/cogp
% repo with a argument count bug corrected:
% https://github.com/vlad17/cogp

% Uses similar assumptions to cogp_fx2007: datadir and runs required
% datadir should go the csv directory
% requires M, runs and datadir to be defined

addpath(genpath([tempdir 'cogp']));
format long;


bray = csvread([datadir 'bray.csv']);
camy = csvread([datadir 'camy.csv']);
chiy = csvread([datadir 'chiy.csv']);
soty = csvread([datadir 'soty.csv']);

x = csvread([datadir 'sotx.csv']);
sel = x >= 10 & x <= 15;

x = x(sel);
y = [bray(:,4), camy(:,4), chiy(:,4), soty(:,4)];
y = y(sel, :);
y(y == -1) = nan; % missing data
ytest = y;
xtest = x;

y(x >= 10.2 & x <= 10.8,2) = nan;
y(x >= 13.5 & x <= 14.2,3) = nan;

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
cf.maxiter = 1500;
cf.nbatch = 1000;
cf.beta = 1/0.1;
cf.initz = 'random';
cf.w = ones(size(y,2),2);
cf.monitor_elbo = 100;
cf.fix_first = false;
Q = 2;
xtest_ix = cell(2,1);
xtest_ix{1} = x >= 10.2 & x <= 10.8;
xtest_ix{2} = x >= 13.5 & x <= 14.2;
outputs = [2,3];

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
  per_out_smses = zeros(2, 1);
  per_out_nlpds = zeros(2, 1);
  csvwrite([tempdir 'cogp-weather-mu' num2str(runs) num2str(M)], mu(:, outputs))
  csvwrite([tempdir 'cogp-weather-var' num2str(runs) num2str(M)], fvar(:, outputs))
  for i=1:2
    t = outputs(i);
    tmp = ~isnan(ytest(:,t)) & xtest_ix{i};
    per_out_smses(i) = mysmse(ytest(tmp,t),mu(tmp,t),ymean(t));
    per_out_nlpds(i) = mynlpd(ytest(tmp,t),mu(tmp,t),fvar(tmp,t));
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


