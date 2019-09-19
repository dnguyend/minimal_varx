import numpy as np
from numpy import log, eye, zeros, ones, diagonal, sqrt
from numpy.linalg import det, solve, cholesky
from scipy.linalg import solve_triangular
import polynomial_matrix as pm


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
        kappa_cov_numerator = self.kappa @ self._cov_numerator
        numerator_mat = kappa_cov_numerator @ self.kappa.T
        kappa_cov_denominator = self.kappa @ self._cov_denominator
        denominator_mat = kappa_cov_denominator @ self.kappa.T
        
        self._numerator_det = det(numerator_mat)
        self._denominator_det = det(denominator_mat)
        self.rayleigh_quotient = self._numerator_det / self._denominator_det
        self.neg_log_llk = log(self._numerator_det)-log(self._denominator_det)
        self._gradient_tensor = zeros((self.kappa_tensor.shape[1]))
        for i in range(self._gradient_tensor.shape[0]):
            der_denom = kappa_cov_denominator @\
                self.kappa_tensor[:, i].reshape(
                    -1, self.p * self.m).T
            der_num = kappa_cov_numerator @ self.kappa_tensor[:, i].reshape(
                -1, self.p * self.m).T

            self._gradient_tensor[i] = 2 * np.sum(diagonal(
                solve(numerator_mat, der_num) -
                solve(denominator_mat, der_denom)))
            
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

    def predict(self, X_in):
        T = X_in.shape[1] - self.p
        Y_out = zeros((self.k, T))
        for i in range(self.Phi.shape[2]):
            Y_out += self.Phi[:, :, i] @ X_in[:, i:T+i]
        return Y_out
        

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


def get_known_terms(K_s, GGT, rho, j, S):
    d_rho, mult_rho = K_s.Psi[rho]
    b_knw = sum([K_s.Psi[a][0] * K_s.Psi[a][1] for a in range(rho)])
    return -S[b_knw:b_knw+K_s.Psi[rho][1], :] @ GGT
    

def simultaneous_orthogonalize(A, B):
    L = cholesky(B)
    inv_L = solve_triangular(L, eye(L.shape[0]))
    return inv_L @ (A @ inv_L), L


def __orthogonalize(K_s, G, V0=None, U0=None):
    """
    Function is not done yet
    Normalize. We have the Cholesky decomposition: of the zero blocks
    G_0 V G_0.T = I_r
    We complete this to a base W such that
    W V W.T = I_m
    If V is none we take V to be I_m
    """
    m = G.shape[1]
    if V0 is None:
        V0 = eye(m)
    if U0 is None:
        U0 = eye(m)    
    
    blk_end = 0
    G_0 = zeros((K_s.agg_rnk, K_s.m))
    G_0_start = 0
    n_psi = len(K_s.Psi)
    """ the ouputs """
    # G_norm = np.zeros_like(G)
    S = zeros((G.shape[0], G.shape[0]))
    W = zeros((m, m))
    
    for rho in range(n_psi):
        d_rho, mult_rho = K_s.Psi[rho]
        blk_end += d_rho * mult_rho
        G_0[G_0_start:G_0_start+mult_rho, :] = G[blk_end-mult_rho:blk_end, :]
        G_0_start += mult_rho
    GG_V_0 = (G_0 @ (V0 @ G_0.T))
    """In the first version we dont do the simultaneous_orthogonalize yet
    """
    L = cholesky(GG_V_0)
    invL = solve_triangular(L, eye(L.shape[0]))
    
    S_start = 0
    iL_start = 0
    G_0_norm = invL @ G_0
    G_0_start = 0
    for r_1 in range(n_psi):
        d_r_1, mult_r_1 = K_s.Psi[r_1]
        for i in range(d_r_1):
            S[S_start:S_start+mult_r_1, S_start:S_start+mult_r_1] =\
                invL[iL_start:iL_start+mult_r_1, iL_start:iL_start+mult_r_1]
            S_start += mult_r_1
        """
        G_norm[S_start+(d_r_1-1)*mult_r_1:S_start+d_r_1*mult_r_1] =\
            G_0_norm[G_0_start:G_0_start+mult_r_1, :]
        """
        G_0_start += mult_r_1
        
    r = K_s.agg_rnk
    W[:r, :] = invL @ G_0
    """now we construct the rest of S by flag Cholesky
    loop from d_rho small to big
    """
    GGT = G @ (V0 @ G.T)
    S_r_end = S.shape[0]

    for r_1 in range(n_psi-1, -1, -1):
        d_r_1, mult_r_1 = K_s.Psi[r_1]
        if d_r_1 <= 1:
            S_r_end = S_r_end - mult_r_1
            continue
        known_terms = zeros()
        for j in range(1, d_r_1):
            pass
            # known_terms[] = get_known_terms(K_s, GGT, r_1, j, S)
            S_tmp = invL @ known_terms
            knw_r = 0
            knw_c = 0
            S_c_end = S.shape[1]
            for r_2 in range(n_psi):
                d_r_2, mult_r_2 = K_s.Psi[r_2]
                for j_2 in range(min(d_r_1, d_r_2)):
                    b_r = S_r_end-(j_2+1)*mult_r_1
                    e_r = S_r_end-j_2*mult_r_1
                    b_c = S_c_end
                    e_c = S_c_end - mult_r_2
                    S[b_r:e_r, b_c:e_c] =\
                        S_tmp[knw_r:knw_r+mult_r_1, knw_c:knw_c+mult_r_2]
                knw_c += mult_r_2
                S_c_end += mult_r_2
        knw_r += mult_r_1
        S_r_end += d_r_1 * mult_r_1
    G_norm = S @ G

    rem_index = _get_pos_index(Psi, G_norm)

    used_index = []
    for i in range(m - r):
        rem_index_where = np.where(rem_index)[0]
        if len(used_index) > 0:
            p = np.dot(np.dot(
                G_norm[rem_index, :], V),
                   G_norm[used_index, :].T)
            G_norm_res = G_norm[rem_index, :] -\
                np.dot(p, G_norm[used_index, :])
        else:
            G_norm_res = np.abs(G_norm[rem_index, :])
        min_col = np.argsort(G_norm_res, axis=1)[:, 0]
        min_val = G_norm_res[:, min_col]
        best_row = rem_index_where[np.argsort(min_val)[0]]

        best_norm = sqrt(np.dot(
            np.dot(
                G_norm_res[best_row, :].reshape(1, -1),
                V0), G_norm_res[best_row, :]))
        W[r+i, :] = G_norm_res[best_row, :] / best_norm

    R = np.dot(np.dot(G_norm, V0), W.T)
    return G_norm, S, W, R


def _get_pos_index(K_s, G_norm):
    pos_index = ones((G_norm.shape[0]), dtype=bool)
    begin = 0
    for rho in range(len(K_s).Psi):
        d_rho, r_rho = K_s.Psi[rho]
        pos_index[begin+(d_rho-1)*r_rho:begin+d_rho*r_rho] = False
        begin += d_rho * r_rho
    
    return pos_index


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
    from utils import random_orthogonal
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
    from utils import gen_stable_model_p_2, random_orthogonal, VAR_sim

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



    

    

    
    

