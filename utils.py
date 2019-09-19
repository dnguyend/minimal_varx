import numpy as np
import numba
import polynomial_matrix as pm
from numpy.random import uniform, randn
from numpy import sqrt
from itertools import combinations


def polar_to_rectangle(modulus, argument):
    return modulus * np.exp(1j*argument)


def rectangle_to_polar(z):
    return np.abs(z), np.angle(z)


def make_real_diag(x, cpx_pairs):
    """ an array of roots, including n_pairs
    of complex roots. the first lx-2*npairs are
    real roots. the last are modulus /angle pairs
    of roots.
    Example:
    make_real_diag([1, 2, 2, pi/4, 3, pi/3], 2)
    Out[13]: 
    array([[ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ],
       [ 0.        ,  2.        ,  0.        ,  0.        ,  0.        ,
         0.        ],
       [ 0.        ,  0.        ,  1.41421356,  1.41421356,  0.        ,
         0.        ],
       [ 0.        ,  0.        , -1.41421356,  1.41421356,  0.        ,
         0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  1.5       ,
         2.59807621],
       [ 0.        ,  0.        ,  0.        ,  0.        , -2.59807621,
         1.5       ]])

    """
    lx = len(x)
    ret = np.zeros((lx, lx), dtype=float)
    if (lx > 2*cpx_pairs):
        np.fill_diagonal(
            ret[:(lx-2*cpx_pairs), :(lx-2*cpx_pairs)], x[:(lx-2*cpx_pairs)])
    if (cpx_pairs != 0):
        for i in range(cpx_pairs):
            root = polar_to_rectangle(
                modulus=x[lx-2*cpx_pairs+2*i],
                argument=x[lx-2*cpx_pairs+2*i+1])
            ret[lx-2*cpx_pairs+2*i:lx-2*cpx_pairs+2*i+2,
                lx-2*cpx_pairs+2*i:lx-2*cpx_pairs+2*i+2] =\
                np.array([root.real, root.imag,
                          -root.imag, root.real]).reshape(2, 2)
    return ret


def random_orthogonal(k):
    """Generate a random orthogonal matrix of size (k, k)
    real matrix based on the paper
    How to generate random matrices from the classical compact groups
    example:
    O = random_orthogonal(3)
    print O
    [[ 0.25452591 -0.92275001  0.28939416]
    [ 0.96115429  0.27441539  0.0296417 ]
    [-0.10676609  0.27060785  0.95675096]]

    np.dot(O, O.T)
    array([[ 1.00000000e+00, -1.21314803e-16, -2.59623113e-18],
       [-1.21314803e-16,  1.00000000e+00,  3.20641969e-17],
       [-2.59623113e-18,  3.20641969e-17,  1.00000000e+00]])
    """
    z = randn(k, k) / sqrt(2.)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    q = np.multiply(q, ph, q)
    return q


def random_innovation_series(D, OO, n):
    """
    generate a random innovation series
    with covariance matrix OO . D OO.T.
    D is an array, diagonal
    OO is orthogonal
    
    Return M of size n times k
    with M
    """
    e = randn(n, D.shape[0])
    e = e - np.mean(e, axis=0)[None, :]
    return np.dot(
        e * sqrt(D)[None, :], OO.T)


def gen_stable_model_p_2(Psi, k):
    """Psi = (2, d_2, 1, d_1)
    Condition: d_2 + d_1 <= k
    We do this by a generalization of SVD:
    A random diagonal positive vector of
    size d_2: Sigma_2
    A random diagonal positve matrix of
    size d_1: Sigma_1
    d_1 + d_2 Random orthogonal vector of size k
    break to matrices U_2,0 and U_1,0 of size
    (k, d_2) and (k, d_1)
    Another unrelated set of random orthogonal
    vector of size d_1 + d_2 break to matrices
    of V_2,0 and V_1, 0 of sizes
    (k, d_2) and (k, d_1)
    Another set of random orthogonal matrix of size 2*d_2
    (so some could be zeros) so we can form
    vector U_2,1 and V_2,1 such that
    U_{2, 1} Sigma_2 V_{2, 1).T = 0
    We form H_{i, j} = U_{i, j} Sigma_i^{1/2}
    We adjust values of Sigma to make sure the system is stable
    """
    low_bound = .1
    high_bound = .97
    uv21_size = .2
    U0 = random_orthogonal(k)
    V0 = random_orthogonal(k)
    UV_21 = random_orthogonal(k) * uv21_size
    U1 = random_orthogonal(k)
    V1 = random_orthogonal(k)

    d_2 = Psi[0][1]
    d_1 = Psi[1][1]

    u_tmp_21 = UV_21[:, :d_2]
    v_tmp_21 = np.zeros((d_2, k), dtype=float)

    if 2*d_2 <= k:
        v_tmp_21[:, :] = UV_21[d_2:2*d_2, :]
    else:
        v_tmp_21[(k-d_2):, :] = UV_21[d_2:, :]

    mm_degree = pm.calc_McMillan_degree(Psi)
    H = np.zeros((k, mm_degree))
    G = np.zeros((mm_degree, k))
    F = pm.calc_Jordan_matrix(Psi, 0)
    stable = False
    max_search = 100
    cnt = 0
    while (not stable) and cnt < max_search:
        sqrt_Sigma_2 = uniform(low_bound, high_bound, d_2)
        sqrt_Sigma_1 = uniform(low_bound, high_bound, d_1)
        H[:, :d_2] = U0[:, :d_2] * sqrt_Sigma_2[None, :]
        H[:, d_2:2*d_2] = u_tmp_21 / sqrt_Sigma_2[None, :]
        H[:, 2*d_2:] = U1[:, :d_1] * sqrt_Sigma_1[None, :]

        G[:d_2, :] = V0[:d_2, :] * sqrt_Sigma_2[:, None]
        G[d_2:2*d_2, :] = v_tmp_21 / sqrt_Sigma_2[:, None]
        G[2*d_2:, :] = V1[:d_1, :] * sqrt_Sigma_1[:, None]

        Phi = pm.state_to_Phi(H, F, G, Psi)
        stable, roots, dd = pm.check_stable(Phi, 2)
        cnt = cnt + 1
        
    return stable, H, G, F, Phi


@numba.jit
def _calc_var_sim(Phi_arr, e):
    """Note that Polynomial matrix
    is in high to low order.
    Phi is usually is in low to high order
    """
    n = e.shape[0]
    p = Phi_arr.shape[2]
    ret = np.zeros_like(e)

    for j in range(0, n):
        for i in range(min(j, p)):
            ret[j, :] += np.dot(
                Phi_arr[:, :, p-i-1], ret[j-i-1, :])
        ret[j, :] += e[j, :]
    return ret


@numba.jit
def calc_residual(Y, Phi_arr):
    n = Y.shape[0]
    k = Y.shape[1]
    p = Phi_arr.shape[2]
    e = np.zeros_like(Y)
    Y_ex = np.zeros((n+p, k))
    Y_ex[-n:, :] = Y[:, :]
    e[-n:, :] = Y[:, :]
    for i in range(p):
        e[-n:, :] -= np.dot(
            Y_ex[p-i-1:p-i-1+n, :], Phi_arr[:, :, i].T)
    return e
    

def VAR_sim(Phi, n, D=None, OO=None):
    """Generate a random stable series
    based on Phi, where the innovation series
    have covariance matrix OO D OO.T
    """
    k = Phi.shape[0]
    if D is None:
        D = np.eye(k)
    if OO is None:
        OO = np.eye(k)
    e = random_innovation_series(D, OO, n)
    Phi_arr = Phi.PolynomialMatrix_to_3darray()
    ret = _calc_var_sim(Phi_arr, e)
    return ret, e


def list_all_psi_hat(m, p):
    """ List all possible Psi. The funny formula below is just stars and bars map
    """
    off = (1,)+(p-1)*(0,)
    for c in combinations(range(m+p-1), p):
        yield [b-a-1+o for a, b, o in zip((-1,)+c, c+(m+p-1,), off)]


def psi_hat_to_psi(Psi_hat):
    p = len(Psi_hat)
    return [(p-i, Psi_hat[i]) for i in range(p)]


def psi_counts(m, p):
    import scipy
    return int(scipy.special.comb(p+m-1, p))

        
if __name__ == '__main__':
    pass
