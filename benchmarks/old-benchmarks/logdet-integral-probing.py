    if testtype == 'logdet':
        print('    logdet')
        sgn, tru = np.linalg.slogdet(exact.K)
        assert sgn > 0

        # TODO: generalize logdet, abstract algorithms, use in optimization?
        # TODO: in abstracted algorithms, need more robust 'cutoffs':
        #       e.g., when convergence stalls

        def minres(mvm):
            ctr = 0
            def cb(_):
                nonlocal ctr
                ctr += 1
            m = len(grid_dists)
            n = params.n
            lo = sla.LinearOperator((n, n), matvec=mvm)

            def inv_mvm(x):
                Kinv_x, succ = sla.minres(
                    lo, x, tol=1e-10, maxiter=m, callback=cb)
                error = np.linalg.norm(x - mvm(Kinv_x))
                #print('            '
                #      'minres conv recon {:8.4e} it {:6d} m {:6d} succ {}'
                #      .format(error, ctr, m, succ))
                return Kinv_x

            return inv_mvm

        dd = np.diag(exact.K) # actual constructor harder w/o exact
        D = lambda x: dd * x
        N = lambda x: apprx.K.matvec(x) - D(x)
        delta0 = np.log(dd).sum()

        def mvm(t):
            ndtn = lambda x: D(x) + t * N(x)
            invmvm = minres(ndtn)
            return lambda x: N(invmvm(x))

        def apx_tr(mvm, n):
            rs = np.random.randint(0, 2, (n, params.n)) * 2 - 1
            trace = 0
            var = 0
            for r in rs:
                x = r.dot(mvm(r))
                trace += x
            trace /= len(rs)
            return trace

        def sample(t):
            return apx_tr(mvm(t), 2)

        vs = np.vectorize(sample)

        import scipy
        integ = scipy.integrate.fixed_quad(vs, 0, 1, n=10)[0]
        #integ, _, _, _ = scipy.integrate.quad(
        # sample, 0, 1, limit=1, full_output=1)
        trace = integ + delta0

        print('        true value from exact {:8.4e}'.format(tru))
        chol = np.log(np.diag(exact.L[0])).sum() * 2
        print('        from cholesky diag    {:8.4e}'.format(chol))
        print('        from         trace    {:8.4e}'.format(trace))
        return
