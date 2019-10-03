import numpy as np
from minimal_varx.minimal_varx import gen_random_stable, varx_minimal_estimator
from minimal_varx.minimal_varx import calc_extended_covariance
from minimal_varx.utils import VAR_sim, random_orthogonal, list_all_psi_hat, psi_hat_to_psi


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
    
    all_psi = list_all_psi_hat(m=m, p=p)
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


def gradient_fit(self, Y, X):
    """simple fitting
    If success, self.Phi, self.H are set to optimal
    values. Returning the optimizer values
    """
    from scipy.optimize import minimize
    from minimal_varx import make_normalized_G
    from numpy import eye
    cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, self.p)
    self.set_covs(cov_res, cov_xlag)
    # k = Y.shape[0]
    m = X.shape[0]

    def f_ratio(x):
        # x is a vector
        if self.neg_log_llk is None:
            self.calc_states(G=x.reshape(-1, m))
        val = self.neg_log_llk
        self.neg_log_llk = None
        return val

    def f_gradient(x):
        try:
            if self._gradient_tensor is not None:
                grd = self._gradient_tensor.copy()
            else:
                self.calc_states(G=x.reshape(-1, m))
                grd = self._gradient_tensor.copy()
        except Exception:
            self.calc_states(G=x.reshape(-1, m))
            grd = self._gradient_tensor.copy()
        self._gradient_tensor = None
        return grd

    c = np.random.randn(self.mm_degree - self.agg_rnk, m - self.agg_rnk)
    G_init = make_normalized_G(self, m, eye(m), c)
    x0 = G_init.reshape(-1)
    opt = minimize(f_ratio, x0, jac=f_gradient)
    if opt['success']:
        G_opt = opt['x'].reshape(-1, m)
        self.calc_H_F_Phi(G_opt, cov_y_xlag)
    return opt


def test_gradient_fit():
    Psi = [(2, 1), (1, 1)]
    k_zero = 0
    k = sum([a[1] for a in Psi]) + k_zero
    p = Psi[0][0]
    T = 1000
    n = T + p
    for ii in range(1):
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
        
        opt = gradient_fit(es, Y, X)
        if not opt['success']:
            print("failed with opt=")
            print(opt)
        else:
            print('orig_llk %s est_llk=%s' % (
                es_G.neg_log_llk, opt['fun']))

            print(Phi.PolynomialMatrix_to_3darray())
            print(es.Phi)


def hessian_fit(self, Y, X):
    """hessian fitting
    If success, self.Phi, self.H are set to optimal
    values. Returning the optimizer values
    """
    from scipy.optimize import minimize
    from minimal_varx import make_normalized_G
    from numpy import eye
    cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, self.p)
    self.set_covs(cov_res, cov_xlag)
    # k = Y.shape[0]
    m = X.shape[0]

    def f_ratio(x):
        # x is a vector
        if self.neg_log_llk is None:
            self.calc_states(G=x.reshape(-1, m))
        val = self.neg_log_llk
        self.neg_log_llk = None
        return val

    def f_gradient(x):
        try:
            if self._gradient_tensor is not None:
                grd = self._gradient_tensor.copy()
            else:
                self.calc_states(G=x.reshape(-1, m))
                grd = self._gradient_tensor.copy()
        except Exception:
            self.calc_states(G=x.reshape(-1, m))
            grd = self._gradient_tensor.copy()
        self._gradient_tensor = None
        return grd

    def hessian_prod(x, eta):
        return self.hessian_prod(
            eta.reshape(self.mm_degree, self.m))
    
    def hessian_prod_(x, eta):
        hessp = np.zeros(eta.shape[0])

        for i in range(hessp.shape[0]):
            a_i = self.kappa_tensor[:, i].reshape(
                -1, self.p * self.m)
            # numerator_mat
            kappa_num_a_i = self._kappa_cov_numerator @ a_i.T
            a_i_num = a_i @ self._cov_numerator
            kappa_eta_T = self.calc_kappa(
                eta.reshape(self.mm_degree, self.m)).T
            a_i_num_eta = a_i_num @ kappa_eta_T
            s1 = solve(self._numerator_mat, kappa_num_a_i + kappa_num_a_i.T)
            s2 = solve(self._numerator_mat, self._kappa_cov_numerator @ kappa_eta_T)
            first_part_num = - s1 @ s2
            second_part_num = solve(self._numerator_mat, a_i_num_eta)

            hess_num = first_part_num + second_part_num

            # denominator
            kappa_denom_a_i = self._kappa_cov_denominator @ a_i.T
            a_i_denom = a_i @ self._cov_denominator
            a_i_denom_eta = a_i_denom @ kappa_eta_T

            sd1 = solve(self._denominator_mat, kappa_denom_a_i + kappa_denom_a_i.T)
            sd2 = solve(self._denominator_mat, self._kappa_cov_denominator @ kappa_eta_T)
            first_part_denom = - sd1 @ sd2
            second_part_denom = solve(self._denominator_mat, a_i_denom_eta)

            hess_denom = first_part_denom + second_part_denom
            hessp[i] = 2 * np.sum(
                np.diagonal(hess_num - hess_denom))
        return hessp

    c = np.random.randn(self.mm_degree - self.agg_rnk, m - self.agg_rnk)
    OO = random_orthogonal(m)
    G_init = make_normalized_G(self, m, OO, c)
    x0 = G_init.reshape(-1)
    opt = scipy.optimize.minimize(
        f_ratio, x0, method='trust-ncg',
        jac=f_gradient, hessp=hessian_prod)
    if opt['success']:
        G_opt = opt['x'].reshape(-1, m)
        self.calc_H_F_Phi(G_opt, cov_y_xlag)
    return opt


def test_hessian_calc():
    from numpy.linalg import solve
    Psi = [(3, 1), (2, 1), (1, 1)]
    k_zero = 1
    k = sum([a[1] for a in Psi]) + k_zero
    p = Psi[0][0]
    T = 1000
    n = T + p
    for ii in range(1):
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
        """
        es_G = varx_minimal_estimator(Psi, m=k)
        es_G.set_covs(cov_res, cov_xlag)
        es_G.calc_states(G)
        """
        
        es1 = varx_minimal_estimator(Psi, m=k)
        es1.set_covs(cov_res, cov_xlag)
        m = k
        c = np.random.randn(self.mm_degree - self.agg_rnk, m - self.agg_rnk)
        OO = random_orthogonal(m)
        from minimal_varx import make_normalized_G
        G_test = make_normalized_G(es1, m, OO, c)

        es1.calc_states(G_test)
        
        # h = 1e-6
        # self = es1
        
        es2 = varx_minimal_estimator(Psi, m=k)
        es2.set_covs(cov_res, cov_xlag)
        
        h = 1e-6
        self = es1
        # eta = np.random.randn(*G.shape)
        # G_1 = G_test.reshape(-1).copy()
        # G_1 += h * eta.reshape(-1)
        # es2.calc_states(G_1.reshape(self.G.shape))
        # diff = (es2._gradient_tensor - es1._gradient_tensor) / h
        # hs = hessian_prod(self, eta)
        # print(diff)
        g_size = self._gradient_tensor.shape[0]
        appx_Hess_matrix = np.zeros((g_size, g_size))

        for i in range(g_size):
            G_1 = es1.G.reshape(-1).copy()
            G_1[i] += h
            es2.calc_states(G_1.reshape(self.G.shape))
            appx_Hess_matrix[i, :] = (es2._gradient_tensor - es1._gradient_tensor) / h
        print(appx_Hess_matrix)

        def hessian_prod(self, eta):
            hessp = np.zeros(eta.reshape(-1).shape[0])

            for i in range(hessp.shape[0]):
                a_i = self.kappa_tensor[:, i].reshape(
                    -1, self.p * self.m)
                # numerator_mat
                kappa_num_a_i = self._kappa_cov_numerator @ a_i.T
                a_i_num = a_i @ self._cov_numerator
                kappa_eta_T = self.calc_kappa(eta).T
                a_i_num_eta = a_i_num @ kappa_eta_T
                s1 = solve(self._numerator_mat, kappa_num_a_i + kappa_num_a_i.T)
                s2 = solve(self._numerator_mat, self._kappa_cov_numerator @ kappa_eta_T)
                first_part_num = - s1 @ s2
                second_part_num = solve(self._numerator_mat, a_i_num_eta)

                hess_num = first_part_num + second_part_num

                # denominator
                kappa_denom_a_i = self._kappa_cov_denominator @ a_i.T
                a_i_denom = a_i @ self._cov_denominator
                a_i_denom_eta = a_i_denom @ kappa_eta_T

                sd1 = solve(self._denominator_mat, kappa_denom_a_i + kappa_denom_a_i.T)
                sd2 = solve(self._denominator_mat, self._kappa_cov_denominator @ kappa_eta_T)
                first_part_denom = - sd1 @ sd2
                second_part_denom = solve(self._denominator_mat, a_i_denom_eta)

                hess_denom = first_part_denom + second_part_denom
                hessp[i] = 2 * np.sum(
                    np.diagonal(hess_num - hess_denom))
            return hessp
        
        exact_Hess = np.zeros((g_size, g_size))
        for jj in range(g_size):
            eta_ = np.zeros(g_size)
            eta_[jj] += 1
            eta = eta_.reshape(G_test.shape)
            exact_Hess[jj, :] = hessian_prod(self, eta)

        print(exact_Hess)
        

def test_hessian_fit():
    from numpy.linalg import solve
    Psi = [(3, 1), (2, 2), (1, 1)]
    k_zero = 0
    k = sum([a[1] for a in Psi]) + k_zero
    p = Psi[0][0]
    T = 1000
    n = T + p
    for ii in range(1):
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

        es1 = varx_minimal_estimator(Psi, m=k)
        es1.set_covs(cov_res, cov_xlag)

        opt = es.gradient_fit(Y, X)
        opt = es1.hessian_fit(Y, X)
        
        if not opt['success']:
            print("failed with opt=")
            print(opt)
        else:
            print('orig_llk %s est_llk=%s' % (
                es_G.neg_log_llk, opt['fun']))

            print(Phi.PolynomialMatrix_to_3darray())
            print(es1.Phi)


def map_o_c_grad(self, grad, o, c=None):
    """map a gradient to the full G
    to a gradient where G is orthogonalized.
    if c is not empty the first c.shape[1] rows
    of o will be the orthogonal part
    """
    grad_o = np.zeros((self.m, self.m))
    p = self.p
    mm = self.mm_degree
    n_psi = len(self.Psi)
    grad_ = grad.reshape(self.mm_degree, self.m)
    if (c is not None) and (np.prod(c.shape) > 0):
        grad_c = np.zeros(c.shape)
        # do the c blocks:
        in_c_row = 0
        out_c_row = 0
        r_unalloc = c.shape[1]
        for rho in range(n_psi):
            d_rho, r_rho = self.Psi[rho]
            
            in_c_end = in_c_row+(d_rho-1)
            out_c_end = out_c_row+(d_rho-1)

            grad_o[:r_unalloc, :] += c[in_c_row:in_c_end, :].T @ grad_[in_c_row:in_c_end, :] 
            grad_c[out_c_row:out_c_end, :] += grad_[in_c_row:in_c_end, :] @ o[:r_unalloc, :].T
            in_c_row = in_c_end + r_rho
            out_c_row = out_c_end
        in_row = 0
        out_row = r_unalloc
        for rho in range(n_psi):
            d_rho, r_rho = self.Psi[rho]
            in_row += (d_rho - 1) * r_rho
            try:
                grad_o[out_row:out_row+r_rho, :] = grad_[in_row:in_row+r_rho]
            except Exception as e:
                print(e)
                raise(e)
            in_row += r_rho
            out_row += r_rho

    else:
        grad_c = None
        # row for output gradient
        out_row = 0
        # row for input_gradient
        in_row = 0
        for rho in range(n_psi):
            d_rho, r_rho = self.Psi[rho]
            in_row += (d_rho - 1) * r_rho
            try:
                grad_o[out_row:out_row+r_rho, :] = grad_[in_row:in_row+r_rho]
            except Exception as e:
                print(e)
                raise(e)
            in_row += r_rho
            out_row += r_rho

    return grad_o, grad_c
        
            

def manifold_fit(self, Y, X):
    from pymanopt.manifolds import Rotations, Euclidean, Product
    from pymanopt import Problem
    from pymanopt.solvers import TrustRegions
    from minimal_varx import make_normalized_G

    cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, self.p)
    self.set_covs(cov_res, cov_xlag)
    m = X.shape[0]

    with_c = (self.mm_degree > self.agg_rnk) and (self.m - self.agg_rnk)
    C = Euclidean(
        self.mm_degree - self.agg_rnk, self.m - self.agg_rnk)
    RG = Rotations(self.m)
    if with_c:
        manifold = Product([RG, C])
    else:
        manifold = RG
    if not with_c:
        c_null = np.zeros((self.mm_degree - self.agg_rnk, self.m - self.agg_rnk))
    else:
        c_null = None
    
    if with_c:
        def cost(x):
            o, c = x
            G = make_normalized_G(self, self.m, o, c)
            self.calc_states(G)
            return self.neg_log_llk
    else:
        def cost(x):
            G = make_normalized_G(self, self.m, x, c_null)
            self.calc_states(G)
            return self.neg_log_llk
    if with_c:                            
        def egrad(x):
            o, c = x
            grad_o, grad_c  = map_o_c_grad(self, self._gradient_tensor, o, c)
            return [grad_o, grad_c]
    else:
        def egrad(x):
            grad_o, grad_c = map_o_c_grad(self, self._gradient_tensor, x, c_null)

            return grad_o
        

    if with_c:
        def ehess(x, Heta):
            o, c = x
            eta = make_normalized_G(
                self, self.m, Heta[0],
                Heta[1])
            hess_raw = self.hessian_prod(eta)
            hess_o, hess_c = map_o_c_grad(self, hess_raw, o, c)
            return [hess_o, hess_c]

    else:
        def ehess(x, Heta):
            eta = make_normalized_G(
                self, self.m,
                Heta, c_null)
            hess_raw = self.hessian_prod(eta)
            hess_o, hess_c = map_o_c_grad(self, hess_raw, x, c_null)
            return hess_o
        
    if with_c:
        min_mle = Problem(
            manifold, cost, egrad=egrad, ehess=ehess)
    else:
        min_mle = Problem(
            manifold, cost, egrad=egrad, ehess=ehess)
            
    solver = TrustRegions()
    opt = solver.solve(min_mle)
    if with_c:
        G_opt = make_normalized_G(
            self, self.m,
            opt[0], opt[1])
    else:
        G_opt = make_normalized_G(
            self, self.m,
            opt, c_null)

    self.calc_H_F_Phi(G_opt, cov_y_xlag)
    return opt


def test_manifold_fit():
    from numpy.linalg import solve
    Psi = [(2, 2), (1, 1)]
    k_zero = 0
    k = sum([a[1] for a in Psi]) + k_zero
    p = Psi[0][0]
    T = 1000
    n = T + p

    for ii in range(1):
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

        es1 = varx_minimal_estimator(Psi, m=k)
        es1.set_covs(cov_res, cov_xlag)

        es2 = varx_minimal_estimator(Psi, m=k)
        es2.set_covs(cov_res, cov_xlag)
        h_opt = es2.hessian_fit(Y, X)
        print(h_opt)
        # opt = es.gradient_fit(Y, X)
        # self = es1

        opt = es1.manifold_fit( Y, X)
        print es1.neg_log_llk
        """
        if not opt['success']:
            print("failed with opt=")
            print(opt)
        else:
            print('orig_llk %s est_llk=%s' % (
                es_G.neg_log_llk, opt['fun']))

            print(Phi.PolynomialMatrix_to_3darray())
            print(es1.Phi)
        """

