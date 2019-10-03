import numpy as np
from numpy import log, eye, zeros, diagonal
from numpy.linalg import det, solve, cholesky, qr
from scipy.linalg import solve_triangular
from . import polynomial_matrix as pm


class varx_minimal_estimator(object):
    """ class to estimate minimal state space representation of varx model
    """
    
    def __init__(self, Psi, Psi_hat=None, m=None, G=None):
        """One of $Psi or Psi_hat should be None.
        one is computed from the other
        Psi_hat is a vector of size p with entries allowed to be zero.

        Psi has from [(r_g, l_g), ... (r_1, l_1)]
        with l_g > 0 and r_g = p
        Psi is obtained from Psi_hat by drop all zero entries
        and order in reverse.
        if both are not none will use Psi
        """
        assert (Psi is not None) or (Psi_hat is not None)
        assert Psi[0][1] > 0
        assert m is not None
        
        if Psi is not None:
            Psi_hat_ = zeros(Psi[0][0])
            for a in Psi:
                Psi_hat_[a[0]-1] = a[1]
        else:
            Psi_hat_ = Psi_hat.copy()
            Psi = [(i, Psi_hat[i-1])
                   for i in range(Psi_hat.shape[0], 0, -1) if Psi_hat[i] != 0]

        self.Psi = Psi
        self.Psi_hat = Psi_hat_
        self.agg_rnk = np.sum([Psi[a][1] for a in range(len(Psi))])

        if (G is not None) and (self.agg_rnk > G.shape[1]):
            raise(ValueError("total rank in Psi=%d is higher than k=%d" % (
                self.agg_rnk, G.shape[1])))

        self._G = G
        self.m = m
        self.mm_degree = pm.calc_McMillan_degree(Psi)
        self.p = Psi[0][0]
        self.kappa_tensor = self.make_vector_kappa_matrix(m)

    def make_vector_kappa_matrix(K_s, m):
        """ the matrix representing the tensor
        """
        p = K_s.p
        mm = K_s.mm_degree
        mat = zeros((mm * p * m, mm * m))
        n_psi = len(K_s.Psi)
        b_rho_col = 0
        b_row = 0
        for rho in range(n_psi):
            d_rho, r_rho = K_s.Psi[rho]
            for j_max in range(d_rho, 0, -1):
                for rr in range(r_rho):
                    b_row += (p - j_max) * m
                    for j_rho in range(j_max):
                        b_col = b_rho_col + (
                            (d_rho - j_rho - 1) * r_rho + rr) * m
                        try:
                            mat[b_row:b_row+m, b_col:b_col+m] = eye(m)
                        except Exception as e:
                            print(e)
                            raise(e)
                        b_row += m
            b_rho_col += d_rho * r_rho * m
        return mat
        
    def set_covs(self, cov_numerator, cov_denominator):
        self._cov_denominator = cov_denominator
        self._cov_numerator = cov_numerator

    def set_kappa(self):
        self.kappa = self.calc_kappa(self.G)

    def calc_states(self, G):
        self.G = G
        self.set_kappa()
        self._kappa_cov_numerator = self.kappa @ self._cov_numerator
        self._numerator_mat = self._kappa_cov_numerator @ self.kappa.T
        self._kappa_cov_denominator = self.kappa @ self._cov_denominator
        self._denominator_mat = self._kappa_cov_denominator @ self.kappa.T
        
        self._numerator_det = det(self._numerator_mat)
        self._denominator_det = det(self._denominator_mat)
        self.rayleigh_quotient = self._numerator_det / self._denominator_det
        self.neg_log_llk = log(self._numerator_det)-log(self._denominator_det)
        self._gradient_tensor = zeros((self.kappa_tensor.shape[1]))
        self._numerator_gradient = zeros((self.kappa_tensor.shape[1]))
        self._denominator_gradient = zeros((self.kappa_tensor.shape[1]))
        
        for i in range(self._gradient_tensor.shape[0]):
            der_denom = self._kappa_cov_denominator @\
                self.kappa_tensor[:, i].reshape(
                    -1, self.p * self.m).T
            der_num = self._kappa_cov_numerator @\
                self.kappa_tensor[:, i].reshape(-1, self.p * self.m).T

            self._numerator_gradient = 2 * np.sum(diagonal(
                 solve(self._numerator_mat, der_num)))
            self._denominator_gradient = 2 * np.sum(diagonal(
                solve(self._denominator_mat, der_denom)))

            self._gradient_tensor[i] = 2 * np.sum(diagonal(
                solve(self._numerator_mat, der_num) -
                solve(self._denominator_mat, der_denom)))

    def hessian_prod(self, eta):
        hessp = np.zeros(eta.reshape(-1).shape[0])

        for i in range(hessp.shape[0]):
            a_i = self.kappa_tensor[:, i].reshape(
                -1, self.p * self.m)
            # numerator_mat
            kappa_num_a_i = self._kappa_cov_numerator @ a_i.T
            a_i_num = a_i @ self._cov_numerator
            kappa_eta_T = self.calc_kappa(
                eta).T
            a_i_num_eta = a_i_num @ kappa_eta_T
            s1 = solve(self._numerator_mat, kappa_num_a_i + kappa_num_a_i.T)
            s2 = solve(self._numerator_mat,
                       self._kappa_cov_numerator @ kappa_eta_T)
            first_part_num = - s1 @ s2
            second_part_num = solve(self._numerator_mat, a_i_num_eta)

            hess_num = first_part_num + second_part_num

            # denominator
            kappa_denom_a_i = self._kappa_cov_denominator @ a_i.T
            a_i_denom = a_i @ self._cov_denominator
            a_i_denom_eta = a_i_denom @ kappa_eta_T

            sd1 = solve(
                self._denominator_mat, kappa_denom_a_i + kappa_denom_a_i.T)
            sd2 = solve(
                self._denominator_mat,
                self._kappa_cov_denominator @ kappa_eta_T)
            first_part_denom = - sd1 @ sd2
            second_part_denom = solve(self._denominator_mat, a_i_denom_eta)

            hess_denom = first_part_denom + second_part_denom
            hessp[i] = 2 * np.sum(
                np.diagonal(hess_num - hess_denom))
        return hessp
                        
    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, G):
        if self.agg_rnk > G.shape[1]:
            raise(ValueError("total rank in Psi=%d is higher than k=%d" % (
                self.agg_rnk, G.shape[1])))
        self._G = G
        self.mm_degree = pm.calc_McMillan_degree(self.Psi)
        self.m = G.shape[1]
        self.p = self.Psi[0][0]

    def calc_kappa(self, G):
        kappa = zeros((self.mm_degree, self.p*self.m))
        kappa_start = 0
        G_start = 0

        for rho in range(len(self.Psi)):
            d_rho, m_rho = self.Psi[rho]
            G_end = G_start + d_rho * m_rho
            for j in range(d_rho):
                # fill the j-th diagonal block of kappa
                for jj in range(d_rho - j):
                    kappa[G_start+jj*m_rho:G_start+(jj+1)*m_rho,
                          (j+jj+self.p-d_rho) *
                          self.m:(j+jj+1+self.p-d_rho)*self.m] =\
                        G[G_end-(j+1)*m_rho:G_end-j*m_rho, :]
            kappa_start = kappa_start + d_rho * m_rho
            G_start = G_end
        return kappa

    def calc_H_F_Phi(
            self, G, cov_y_xlag):
        self.G = G
        self.set_kappa()
        h1 = self.kappa @ self._cov_denominator
        h1 = h1 @ self.kappa.T
        L = cholesky(h1)
        S1 = solve_triangular(L, self.kappa @ cov_y_xlag.T, lower=True)
        self.H = solve_triangular(L.T, S1).T
        self.F = pm.calc_Jordan_matrix(self.Psi, 0)
        self.Phi = pm.state_to_Phi(
            self.H, self.F, G, self.Psi).PolynomialMatrix_to_3darray()

    def simple_fit(self, Y, X):
        """simple fitting
        If success, self.Phi, self.H are set to optimal
        values. Returning the optimizer values
        """
        from scipy.optimize import minimize
        cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, self.p)
        self.set_covs(cov_res, cov_xlag)
        # k = Y.shape[0]
        m = X.shape[0]
        
        def f_ratio(x):
            # x is a vector
            self.calc_states(G=x.reshape(-1, m))
            return self.neg_log_llk

        def f_gradient(x):
            return self._gradient_tensor
        
        c = np.random.randn(self.mm_degree - self.agg_rnk, m - self.agg_rnk)
        G_init = make_normalized_G(self, m, eye(m), c)
        x0 = G_init.reshape(-1)
        opt = minimize(f_ratio, x0)
        if opt['success']:
            G_opt = opt['x'].reshape(-1, m)
            self.calc_H_F_Phi(G_opt, cov_y_xlag)
        return opt

    def gradient_fit(self, Y, X):
        """gradient_fit fitting
        If success, self.Phi, self.H are set to optimal
        values. Returning the optimizer values
        """
        from scipy.optimize import minimize
        cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, self.p)
        self.set_covs(cov_res, cov_xlag)
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

    def hessian_fit(self, Y, X):
        """hessian fitting
        If success, self.Phi, self.H are set to optimal
        values. Returning the optimizer values
        """
        from scipy.optimize import minimize
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

        c = np.random.randn(self.mm_degree - self.agg_rnk, m - self.agg_rnk)
        # OO = random_orthogonal(m)
        G_init = make_normalized_G(self, m, eye(m), c)
        x0 = G_init.reshape(-1)
        opt = minimize(
            f_ratio, x0, method='trust-ncg',
            jac=f_gradient, hessp=hessian_prod)
        if opt['success']:
            G_opt = opt['x'].reshape(-1, m)
            self.calc_H_F_Phi(G_opt, cov_y_xlag)
        return opt    

    def predict(self, X_in):
        T = X_in.shape[1] - self.p
        Y_out = zeros((self.k, T))
        for i in range(self.Phi.shape[2]):
            Y_out += self.Phi[:, :, i] @ X_in[:, i:T+i]
        return Y_out

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
        # from minimal_varx import make_normalized_G

        cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, self.p)
        self.set_covs(cov_res, cov_xlag)
        # m = X.shape[0]

        with_c = (self.mm_degree > self.agg_rnk) and (self.m - self.agg_rnk)
        C = Euclidean(
            self.mm_degree - self.agg_rnk, self.m - self.agg_rnk)
        RG = Rotations(self.m)
        if with_c:
            raise(ValueError(
                "This option is implemented only for self.agg_rnk == m"))
        if with_c:
            manifold = Product([RG, C])
        else:
            manifold = RG
        if not with_c:
            c_null = np.zeros(
                (self.mm_degree - self.agg_rnk, self.m - self.agg_rnk))
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
                grad_o, grad_c = self.map_o_c_grad(
                    self._gradient_tensor, o, c)
                return [grad_o, grad_c]
        else:
            def egrad(x):
                grad_o, grad_c = self.map_o_c_grad(
                    self._gradient_tensor, x, c_null)

                return grad_o

        if with_c:
            def ehess(x, Heta):
                o, c = x
                eta = make_normalized_G(
                    self, self.m, Heta[0],
                    Heta[1])
                hess_raw = self.hessian_prod(eta)
                hess_o, hess_c = self.map_o_c_grad(hess_raw, o, c)
                return [hess_o, hess_c]

        else:
            def ehess(x, Heta):
                eta = make_normalized_G(
                    self, self.m,
                    Heta, c_null)
                hess_raw = self.hessian_prod(eta)
                hess_o, hess_c = self.map_o_c_grad(hess_raw, x, c_null)
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

    
def get_structure_row_block(K_s, rho, i):
    """get the i-th power block index
    """
    start = np.sum([K_s.Psi[x] for x in range(rho)])
    return start, start + K_s.Psi[rho][1]
                                                             

def make_normalized_G(K_s, m, OO, c):
    """
    Normalized G would have form
    G_0 G_0.T = I_{sum r_i} = I_ll
    G_{r, j} are orthogonal to G_0
    via generalized GramSchmidt procedure
    the first sum mult_i rows
    x is an upper triangular matrix
    of size (mm-ll) times (m-ll)
    """
    """
    desired_size = K_s.mm_degree * m - np.sum(
        [K_s.Psi[rho][1]*K_s.Psi[rho][1] for rho in range
         (len(K_s.Psi))])
    if x.shape[0] < desired_size:
        raise(ValueError("size of x=%d is less than desired size %d" % (
            x.shape[0], desired_size)))
    """
    G = zeros((K_s.mm_degree, m))
    n_psi = len(K_s.Psi)
    blk_end = 0
    O_start = 0
    for rho in range(n_psi):
        d_rho, mult_rho = K_s.Psi[rho]
        blk_end += d_rho * mult_rho
        G[blk_end-mult_rho:blk_end, :] = OO[O_start:O_start+mult_rho, :]
        O_start += mult_rho
    blk_end = 0
    c_start = 0

    for rho in range(n_psi):
        d_rho, mult_rho = K_s.Psi[rho]
        blk_end += d_rho * mult_rho

        for i in range(1, d_rho):
            G[blk_end-(i+1)*mult_rho:blk_end-i*mult_rho, :] =\
                c[c_start:c_start+mult_rho, :] @ OO[O_start:, :]
            c_start += mult_rho
    return G


def get_zero_blocks(es, G):
    n_psi = len(es.Psi)
    G0 = zeros((es.agg_rnk, es.m))
    G0_start = 0
    blk_end = 0

    for rho in range(n_psi):
        d_rho, mult_rho = es.Psi[rho]
        blk_end += d_rho * mult_rho
        G0[G0_start:G0_start+mult_rho, :] = G[blk_end-mult_rho:blk_end, :]
        G0_start += mult_rho
    return G0


def LQ_multi_lags(es, G):
    """
    Do the LQ decomposition for multi lags. Due to naming
    conflict we use W instead of Q

    First step: is W0, using QR on G0.

    Second step: solve for the remaining blocks of S
    S consists of blocks S_{rho, j, rho_2, j_2}
    Equation is
    sum_{rho_2, j_2} S_{rho, j, rho_2, j_2} G_{rho_2, j_2} W_{rho_1, 0}.T = 0
    note the block G0 @ W0.T = RR_.T
    W0 @ G0 = RR_
    RR_ @ S_{rho, j, rho_1, 0}.T = -\sum_{rho_2, j_2>0} W_{rho_1, 0}\
         G_{rho_2, j_2}.T S_{rho, j-1, rho_2, j_2-1}.T

    S_{rho, j, rho_1, 0} = -(\sum_{rho_2, j_2>0}\
         S_{rho, j-1, rho_2, j_2-1} @ G_{rho_2, j_2}  @ W_{rho_1, 0}.T) @ S0
    Third step: propagate on diagonals
    """
    n_psi = len(es.Psi)    
    S = zeros((G.shape[0], G.shape[0]))
    W = zeros(G.shape)
    
    # first populate S with S0
    G0 = get_zero_blocks(es, G)
    QQ_, RR_ = qr(G0.T)
    S0 = solve_triangular(RR_.T, eye(es.agg_rnk), lower=True)
    # L0 = RR_.T
    # QQ = QQ_.T

    S_end_1 = 0  # left most of the d_r_1 block
    S0_start_1 = 0

    S0_start_2 = 0

    # populate L with the zero block solution:
    # this consists of blocks of form
    debug = False
    for r_1 in range(n_psi):
        d_r_1, mult_r_1 = es.Psi[r_1]
        S_end_2 = 0
        S_end_1 += d_r_1 * mult_r_1
        S0_start_2 = 0
        W[S_end_1-mult_r_1:S_end_1, :] =\
            QQ_[:, S0_start_1:S0_start_1+mult_r_1].T
        for r_2 in range(r_1+1):
            d_r_2, mult_r_2 = es.Psi[r_2]
            S_end_2 += d_r_2 * mult_r_2
            for j in range(min(d_r_1, d_r_2)):
                S[S_end_1-(j+1)*mult_r_1:S_end_1-j*mult_r_1,
                  S_end_2-(j+1)*mult_r_2:S_end_2-j*mult_r_2] =\
                    S0[S0_start_1:S0_start_1+mult_r_1,
                       S0_start_2:S0_start_2+mult_r_2]
            S0_start_2 += mult_r_2
        S0_start_1 += mult_r_1
        
    # second step

    for j in range(1, es.p):
        S_r_end = 0
        for r in range(n_psi):
            d_r, mult_r = es.Psi[r]
            if j >= d_r:
                continue
            elif debug:
                print('doing j=%d d_r=%d' % (j, d_r))
                
            n_j_col = sum([rr[1] for rr in es.Psi if rr[0] >= d_r - j])
            S_rhs = zeros((mult_r, n_j_col))

            S_r_end += d_r*mult_r
            S_r_j = S_r_end - j * mult_r
            S_r_j_ = S_r_j - mult_r

            # S_c = 0
            S_c_2_end = 0
            for r_2 in range(n_psi):
                d_r_2, mult_r_2 = es.Psi[r_2]
                S_c_2_end += d_r_2 * mult_r_2

                for j_2 in range(1, min(j+1, d_r_2, d_r_2-d_r+j+1)):
                    S_c = S_c_2_end - j_2 * mult_r_2
                    S_c_ = S_c - mult_r_2
                    W[S_r_j_:S_r_j, :] += S[S_r_j_:S_r_j, S_c_:S_c] @\
                        G[S_c_:S_c, :]
            pos_rho_1 = 0
            S_rhs_c = 0
            for r_1 in range(n_psi):
                d_r_1, mult_r_1 = es.Psi[r_1]
                # S_c += d_r_1 * mult_r_1
                pos_rho_1 += d_r_1 * mult_r_1
                pos_rho_1_ = pos_rho_1 - mult_r_1

                S_rhs_c_ = S_rhs_c
                S_rhs_c += mult_r_1
                if j >= d_r - d_r_1:
                    S_rhs[:, S_rhs_c_:S_rhs_c] -=\
                        W[S_r_j_:S_r_j, :]  @ W[pos_rho_1_:pos_rho_1, :].T
                else:
                    break

            # solve
            # S_lhs_ = np.linalg.solve(W0[:n_j_col, :] @\
            #    G0[:n_j_col, :].T, S_rhs.T).T
            # S_lhs = np.linalg.solve(RR_[:n_j_col, :n_j_col], S_rhs.T).T

            S_lhs = S_rhs @ S0[:n_j_col, :n_j_col]

            # propagate:

            col_r1_base = 0
            lhs_col = 0
            for rs1 in range(n_psi):
                d_rs_1, mult_rs_1 = es.Psi[rs1]
                col_r1_base += d_rs_1 * mult_rs_1
                if j >= d_r - d_rs_1:
                    W[S_r_j_:S_r_j, :] +=\
                        S_lhs[:, lhs_col:lhs_col+mult_rs_1] @\
                        G[col_r1_base-mult_rs_1:col_r1_base, :]
                    for j3 in range(min(d_r - j, d_rs_1)):
                        S[S_r_j-(j3+1)*mult_r:S_r_j-j3*mult_r,
                          col_r1_base-(1+j3)*mult_rs_1:col_r1_base-j3*mult_rs_1] =\
                            S_lhs[:, lhs_col:lhs_col+mult_rs_1]
                lhs_col += mult_rs_1

    return W, S


def gen_extended_matrix(K_s, U, vec_kappa_mat):
    bU = np.dot(K_s.kappa, U)
    bUbT = np.dot(bU, K_s.kappa.T)
    inv_bUbT = np.linalg.inv(bUbT)

    return np.dot(np.dot(
        vec_kappa_mat.T,
        np.kron(inv_bUbT, U)), vec_kappa_mat)


def calc_extended_covariance(
        Y, X, p):
    """ Y of size T+p
    X of size T+p
    return cov_xlag = XLAG @ XLAG.T
    return cov_res = XLAG @ XLAG.T - XLAG @ Y.T @ (Y @ Y.T)^{-1} @ Y @ XLAG.T
    """
    m = X.shape[0]
    k = Y.shape[0]
    cov_res = zeros((p*m, p*m), dtype=float)
    cov_xlag = zeros((p*m, p*m), dtype=float)
    cov_y_xlag = zeros((k, p*m), dtype=float)
    T = Y.shape[1] - p
    YYT = np.dot(Y[:, p:], Y[:, p:].T) / T
    L = cholesky(YYT)
    # inv_L = solve_triangular(L, np.eye(L.shape[0]))
    for i in range(p):
        XYT2 = np.dot(X[:, i:T+i], Y[:, p:].T) / T
        cov_y_xlag[:, i*m:(i+1)*m] = XYT2.T
        for j in range(p):
            XXT = np.dot(X[:, i:T+i], X[:, j:T+j].T) / T
            YXT = np.dot(Y[:, p:], X[:, j:T+j].T) / T
            S1 = solve_triangular(L, YXT, lower=True)
            S2 = solve_triangular(L.T, S1)
            cov_xlag[i*m:(i+1)*m, j*m:(j+1)*m] = XXT
            cov_res[i*m:(i+1)*m, j*m:(j+1)*m] = XXT - XYT2 @ S2
    return cov_res, cov_xlag, cov_y_xlag


def gen_random_stable(
        es: varx_minimal_estimator, k: int,
        max_trials=1000, scale_range=(.8, .99)):
    """ generating a stable matrix of a particlular type
    """
    from .utils import random_orthogonal
    c = np.random.randn(es.mm_degree - es.agg_rnk, k - es.agg_rnk)
    OO = random_orthogonal(es.m)
    G = make_normalized_G(es, k, OO, c)
    return gen_random_stable_with_G(
        es, k, G, max_trials, scale_range)

    
def gen_random_stable_with_G(es, k, G, max_trials=1000,
                             scale_range=(.8, .99)):
    """Generate a random stable matrix
    """
    stable = False
    cnt = 0
    F = pm.calc_Jordan_matrix(es.Psi, 0)

    while (not stable) and (cnt < max_trials):
        H = np.random.uniform(-1, 1, (k, es.mm_degree))
        scale = np.random.uniform(*scale_range)
        rel_start = 0
        for ir in range(len(es.Psi)):
            r, ll = es.Psi[ir]
            rel_end = rel_start + r * ll
            for lx in range(r):
                H[:, rel_start+lx*ll:(lx+1)*ll] *= scale
            rel_start = rel_end
            
        Phi = pm.state_to_Phi(H, F, G, es.Psi)
        stable, roots, dd = pm.check_stable(Phi, es.p)
        if stable:
            return stable, H, F, G, Phi
    return stable, None, None, G, None


if __name__ == '__main__':
    from .utils import gen_stable_model_p_2, random_orthogonal, VAR_sim

    np.random.seed(0)
    Psi = [(2, 1), (1, 2)]
    # k_zero = 1
    # Psi = [(2, 1), (1, 1)]
    k_zero = 0
    
    p = Psi[0][0]
    k = sum([a[1] for a in Psi]) + k_zero
    stable, H, G, F, Phi = gen_stable_model_p_2(Psi, k)
    T = 1000
    n = T + p
    D = np.arange(k) * 0.2 + 1
    OO = random_orthogonal(k)
    Y, e = VAR_sim(Phi, n, D, OO)
    Y = Y.T
    e = e.T
    X = Y.copy()
    # X = np.concatenate([Y, np.ones((1, X.shape[1]))])
    cov_res, cov_xlag, cov_y_xlag = calc_extended_covariance(Y, X, p)

    """ now try to fit. Dont care about symmetries
    scan over the whole space
    """
    from scipy.optimize import minimize
    K_s = varx_minimal_estimator(Psi, m=k)
    K_s.set_covs(cov_res, cov_xlag)
    c = np.zeros((K_s.mm_degree - K_s.agg_rnk, k - K_s.agg_rnk))
    G_init = make_normalized_G(K_s, k, np.eye(k), c)
    x0 = G_init.reshape(-1)
    
    def f_ratio(x):
        # x is a vector
        K_s.calc_states(G=x.reshape(-1, k))
        return K_s.neg_log_llk

    def f_gradient(x):
        return K_s._gradient_tensor

    xlag = np.zeros((p*k, T))
    for i in range(p):
        xlag[i*k:(i+1)*k, :] = X[:, i:T+i]

    K_s.calc_states(G)
    dnom = K_s.kappa @ cov_xlag @ K_s.kappa.T
    res = K_s.kappa @ cov_res @ K_s.kappa.T
    YYT = Y[:, p:] @ Y[:, p:].T / T

    res2 = YYT - cov_y_xlag @ K_s.kappa.T @ solve(
        K_s.kappa @ cov_xlag @ K_s.kappa.T, K_s.kappa @ cov_y_xlag.T) 
    print(det(YYT) * K_s.rayleigh_quotient)
    kp_xlag = K_s.kappa @ xlag

    kp_xlag2 = np.zeros_like(kp_xlag)
    kp_xlag2[0, :] = G[0, :] @ X[:, 1:T+1] + G[1, :] @ X[:, :T]
    kp_xlag2[1, :] = G[1, :] @ X[:, 1:T+1]
    kp_xlag2[2, :] = G[2, :] @ X[:, 1:T+1]
    
    v1 = solve(kp_xlag @ kp_xlag.T, kp_xlag @ Y[:, p:].T) / T
    expln = Y[:, p:] @ kp_xlag.T @ v1
    res2_good = YYT - expln
    print(det(res2))
    print(det(res2_good))

    # double check H:
    H_check = Y[:, p:] @ kp_xlag.T @ np.linalg.inv(kp_xlag @ kp_xlag.T)
    e2 = Y[:, p:] - H_check @ kp_xlag2
    e3 = Y[:, p:] - H @ kp_xlag2
    print(e2 @ e2.T / T)
    print(e3 @ e3.T / T)

    Phi1 = H[:, [2]] @ G[[2], :] + H[:, [0]] @ G[[0], :] +\
        H[:, [1]] @ G[[1], :]
    Phi2 = H[:, [0]] @ G[[1], :]
    e4 = Y[:, p:] - Phi1 @ Y[:, 1:T+1] - Phi2 @ Y[:, :T]

    # ok so far. So we have proved Y = Phi(L) Y + e.
    # However, H_check is not recovered. Why ?
    # it is a regression. With more data will converges
    # finite sample we dont have orthogonality
    # opt = minimize(f_ratio, x0, jac=f_gradient)

    # It works!!!
    
    opt = minimize(f_ratio, x0)
    G_opt = opt['x'].reshape(-1, k)
    K_s_opt = varx_minimal_estimator(Psi, m=k)
    K_s_opt.set_covs(cov_res, cov_xlag)
    K_s_opt.calc_states(G_opt)

    # Psi = [(2, 1), (1, 1)]
    H
    
    H_opt, F_opt, Phi_opt = K_s_opt.calc_H_F_Phi(G_opt, cov_y_xlag)
    Phi_ar = Phi.PolynomialMatrix_to_3darray()
    Phi_ar_opt = Phi_opt.PolynomialMatrix_to_3darray()
    e1 = Y[:, p:] - Phi_ar[:, :, 0] @ Y[:, p-1:-1] -\
        Phi_ar[:, :, 1] @ Y[:, p-2:-2]
    e_opt = Y[:, p:] - Phi_ar_opt[:, :, 0] @ Y[:, p-1:-1] -\
        Phi_ar_opt[:, :, 1] @ Y[:, p-2:-2]



    

    

    
    

