import numpy as np
from minimal_varx import gen_random_stable, varx_minimal_estimator
from minimal_varx import calc_extended_covariance
from utils import VAR_sim, random_orthogonal, list_all_psi, psi_hat_to_psi


def test_random_stable():
    Psi = [(2, 1), (1, 2)]
    k_zero = 2
    k = sum([a[1] for a in Psi]) + k_zero
    
    es = varx_minimal_estimator(Psi, m=k)
    stable, H, F, G, Phi = gen_random_stable(es, k)

    p = Psi[0][0]
    T = 1000
    n = T + p
    D = np.arange(k) * 0.2 + 1
    OO = random_orthogonal(k)
    Y, e = VAR_sim(Phi, n, D, OO)
    Y = Y.T
    e = e.T
    X = Y.copy()
    print(Y[:, -5:])
    print(X[:, -5:])
    
    
def k_2_p_2_sphere():
    Psi = [(2, 1), (1, 1)]
    k_zero = 0
    k = sum([a[1] for a in Psi]) + k_zero
    p = Psi[0][0]
    T = 1000
    n = T + p
    for ii in range(10):
        D = np.arange(k) * 0.2 + 1
        OO = random_orthogonal(k)
        try:
            es = varx_minimal_estimator(Psi, m=k)
            stable, H, F, G, Phi = gen_random_stable(es, k)
            Y, e = VAR_sim(Phi, n, D, OO)
            Y = Y.T
            e = e.T
            X = Y.copy()
            cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, p)
        except Exception:
            continue

        es_G = varx_minimal_estimator(Psi, m=k)
        es_G.set_covs(cov_res, cov_xlag)
        es_G.calc_states(G)

        opt = es.simple_fit(Y, X)
        if not opt['success']:
            print("failed with opt=")
            print(opt)
        else:
            print('orig_llk %s est_llk=%s' % (
                es_G.neg_log_llk, es.neg_log_llk))

            print(Phi.PolynomialMatrix_to_3darray())
            print(es.Phi)


def k_2_p_2_sphere_line_bundle():
    Psi = [(2, 1)]
    k_zero = 1
    k = sum([a[1] for a in Psi]) + k_zero
    p = Psi[0][0]
    T = 1000
    n = T + p
    for ii in range(10):
        D = np.arange(k) * 0.2 + 1
        OO = random_orthogonal(k)
        try:
            es = varx_minimal_estimator(Psi, m=k)
            stable, H, F, G, Phi = gen_random_stable(es, k)
            Y, e = VAR_sim(Phi, n, D, OO)
            Y = Y.T
            e = e.T
            X = Y.copy()
            cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, p)
        except Exception:
            continue

        es_G = varx_minimal_estimator(Psi, m=k)
        es_G.set_covs(cov_res, cov_xlag)
        es_G.calc_states(G)

        opt = es.simple_fit(Y, X)
        if not opt['success']:
            print("failed with opt=")
            print(opt)
        else:
            print('orig_llk %s est_llk=%s' % (
                es_G.neg_log_llk, es.neg_log_llk))

            print(Phi.PolynomialMatrix_to_3darray())
            print(es.Phi)


def k_5_p_2_all_psi():
    k = 5
    m = k
    p = 2
    Psi0 = [(2, 2), (1, 2)]

    T = 1000
    n = T + p
    D = np.arange(k) * 0.2 + 1
    OO = random_orthogonal(k)

    try:
        es = varx_minimal_estimator(Psi0, m=k)
        stable, H, F, G, Phi = gen_random_stable(es, k)
        Y, e = VAR_sim(Phi, n, D, OO)
        Y = Y.T
        e = e.T
        X = Y.copy()
        cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, p)
        es_G = varx_minimal_estimator(Psi0, m=k)
        es_G.set_covs(cov_res, cov_xlag)
        es_G.calc_states(G)
    except Exception:
        pass
    
    all_psi = list_all_psi(m=m, p=p)
    for Psi_ in all_psi:
        Psi = psi_hat_to_psi(Psi_)
        es = varx_minimal_estimator(Psi, m=k)
        opt = es.simple_fit(Y, X)
        print('Psi=%s' % Psi)
        if not opt['success']:
            print("failed with opt=")
            print(opt)
        else:
            print('orig_llk %s est_llk=%s' % (
                es_G.neg_log_llk, es.neg_log_llk))


def k_5_p_3_x_not_autogressive():
    np.random.seed(0)
    k = 5
    m = 7
    p = 2
    Psi = [(2, 2), (1, 2)]
    # first generate X
    
    es = varx_minimal_estimator(Psi, m=m)
    stable, _, _, _, PhiX = gen_random_stable(es, m)
    Dx = np.arange(m) * 0.2 + 1
    OOx = random_orthogonal(m)
    T = 1000
    n = T + p
    X, ex = VAR_sim(PhiX, n, Dx, OOx)
    X = X.T
    # next generate Y
    stable, _, F, G, _ = gen_random_stable(es, m)
    H = np.random.randn(k, es.mm_degree)
    from polynomial_matrix import state_to_Phi
    from utils import random_innovation_series
    Phi = state_to_Phi(H, F, G, Psi)
    Phi_arr = Phi.PolynomialMatrix_to_3darray()
    D = np.arange(k) * 0.2 + 1
    OO = random_orthogonal(k)
    e = random_innovation_series(D, OO, n)
    Y = np.zeros_like(e)

    for j in range(0, n):
        for i in range(min(j, p)):
            Y[j, :] += Phi_arr[:, :, p-i-1] @ X[j-i-1, :]
        Y[j, :] += e[j, :]
    Y = Y.T
    es_G = varx_minimal_estimator(Psi, m=m)
    es_G.set_covs(es._cov_numerator, es._cov_denominator)
    es_G.calc_states(G)

    opt = es.simple_fit(Y, X)
    if not opt['success']:
        print("failed with opt=")
        print(opt)
    else:
        print('orig_llk %s est_llk=%s' % (
            es_G.neg_log_llk, es.neg_log_llk))


def k_8_p3_generic():
    Psi = [(3, 1), (2, 1), (1, 2)]
    k_zero = 4
    k = sum([a[1] for a in Psi]) + k_zero
    p = Psi[0][0]
    T = 1000
    n = T + p
    for ii in range(10):
        D = np.arange(k) * 0.2 + 1
        OO = random_orthogonal(k)
        try:
            es = varx_minimal_estimator(Psi, m=k)
            stable, H, F, G, Phi = gen_random_stable(
                es, k, scale_range=(0.3, .6))
            Y, e = VAR_sim(Phi, n, D, OO)
            Y = Y.T
            e = e.T
            X = Y.copy()
            cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, p)
        except Exception:
            pass

        es_G = varx_minimal_estimator(Psi, m=k)
        es_G.set_covs(cov_res, cov_xlag)
        es_G.calc_states(G)

        # simple fit does not work
        found = False
        cnt = 0
        while (not found) and (cnt < 11):
            opt = es.simple_fit(Y, X)
            if opt['success']:
                print(Phi.PolynomialMatrix_to_3darray())
                print(es.Phi)
                print(es_G.neg_log_llk)
                print(es.neg_log_llk)
                found = True
            else:
                print(opt)
                cnt += 1


def k_8_p3_orthogonal_bundle():
    pass

