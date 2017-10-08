from runlmc.lmc.likelihood import ExactLMCLikelihood
from runlmc.lmc.functional_kernel import FunctionalKernel
from standard_tester import gen_random_k
import numpy as np

np.random.seed(1234)

D = 5
NperD = 1000
fk = gen_random_k()
fk.coreg_vecs[0][:] = np.random.rand(5)
fk.coreg_vecs[1][:] = np.random.rand(5)
print('generated Q = 2 SLFM RBF kernel')

Xs = np.random.rand(D, NperD, 2)
print('generated', Xs.shape, 'inputs')

fk.set_input_dim(2)
K = ExactLMCLikelihood.kernel_from_indices(list(Xs), list(Xs), fk)
K += np.diag(np.ones(len(K)) / 10)
print('generated', K.shape, 'kernel')

L = np.linalg.cholesky(K)
Ys = L.dot(np.random.randn(len(K))).reshape(D, -1)
print('generated', Ys.shape, 'sampled outputs')

np.save('xss.npy', Xs)
np.save('yss.npy', Ys)

# cogp has a silly representation that's fixed-length
np.savetxt('xss.csv', Xs.reshape(-1, 2), delimiter=",")
cogp_ys = np.empty((D * NperD, D))
cogp_ys[:] = np.nan
for i in range(D):
    start = i * NperD
    end = (i + 1) * NperD
    cogp_ys[start:end, i] = Ys[i]
np.savetxt('yss.csv'.format(i), cogp_ys, delimiter=",")
