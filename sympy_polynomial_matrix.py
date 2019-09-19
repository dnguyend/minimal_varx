from __future__ import print_function
from __future__ import division
import sympy as sp
import numpy as np

x = sp.symbols('x')


"""Various construction for polynomial matrix using sympy
coefficients will be rational only.
Translation: from PolynomialMatrix to SympyMatrix.
From SympyMatrix to polynomial matrix
what else:
"""


class SymPyPolynomialMatrix(sp.Matrix):
    """Input matrix of dimension 2 of type np.poly1d
    We could realize a polynomial matrix in two ways
    One is a matrix of poly1d which is the main method we use here.
    Another is as a 3d array with the third dimension is the polynomial
    dimension. We provide a few method to navigate between the two ways
    as it may be convenient to use the secondway sometime
    """

    @classmethod
    def __new__(cls, *args, **kwargs):
        return super(SymPyPolynomialMatrix, cls).__new__(
            cls, *args, **kwargs)

    def degree(self):
        return max([expr_degree(a) for a in self.vec()])
             
    def eval(self, val, dtype=float):
        """x is
        """
        shape = self.shape
        ret = sp.zeros(
            *shape,
            dtype=dtype)
        for i in range(shape[0]):
            for j in range(shape[1]):
                ret[i, j] = self[i, j].subs(x, val)
        return ret

    @staticmethod
    def coef_array_to_SPMatrix(arr, factor=None):
        """array is a numpy array
        """
        if len(arr.shape) == 2:
            if factor is None:
                return SymPyPolynomialMatrix(
                    arr.shape[0], arr.shape[1],
                    list(arr.reshape(-1)))
            else:
                return SymPyPolynomialMatrix.zeros(
                    arr.shape[0], arr.shape[1],
                    [sp.Rational(round(a * factor), factor)
                     for a in arr.reshape(-1)])

        m = SymPyPolynomialMatrix.zeros(
            *(arr.shape[:2]))
            
        for i in xrange(m.shape[0]):
            for j in xrange(m.shape[1]):
                m[i, j] = array_to_poly(arr[i, j, :], factor)
        return m

    def smith_mcmillan_form(self):
        """return P, A, Q
        such that self = P * A * Q
        P Q are unimodular and A is diagonal
        if numerator is None it returns the smith normal form
        """

        n_row, n_col = self.shape[:2]
        A = self.copy()
        n = min(n_row, n_col)
        cnt = 0
        det_factor = 1
        P = SymPyPolynomialMatrix.eye(n_row)
        Q = SymPyPolynomialMatrix.eye(n_col)

        while cnt < n:
            cleared = False
            while not cleared:
                position = _find_smallest_degree(
                    A, cnt, n_row, n_col)
                cleared = _check_row_col_cleared(A, cnt, n_row, n_col)

                if False and (position == (cnt, cnt) and cleared):
                    coeffs = sp.Poly(A[position], x).coeffs()
                    if (coeffs[0] != 0 and coeffs[0] != 1):
                        det_factor *= coeffs[0]
                        P[:, cnt] *= coeffs[0]
                        A[position] /= coeffs[0]
                        A[position] = sp.simplify(A[position])
                else:
                    if cnt != position[0]:
                        A.swap_rows(cnt, position[0], P)

                        det_factor *= -1
                    if cnt != position[1]:
                        A.swap_cols(cnt, position[1], Q)
                        det_factor *= -1

                    for i in xrange(cnt + 1, n_row):
                        try:
                            q, r = sp.div(
                                A[i, cnt], A[cnt, cnt], x)
                        except Exception as e:
                            print('Trying to divide by zero %f' % A[cnt, cnt])
                            raise(e)
                        if expr_degree(A[cnt, cnt]) == 0:
                            r = 0
                        A.subtract_rows(i, cnt, q, r, P)

                    for i in xrange(cnt + 1, n_col):
                        q, r = sp.div(A[cnt, i], A[cnt, cnt], x)
                        if expr_degree(A[cnt, cnt]) == 0:
                            r = 0
                        A.subtract_cols(i, cnt, q, r, Q)

            cnt += 1
        P = cleanup(P)
        Q = cleanup(Q)
        A = cleanup(A)
        det_factor = sp.simplify(det_factor)
        return P, A, Q, det_factor

    def swap_rows(self, i, j, P=None):
        sw = self[i, :].copy()
        self[i, :] = self[j, :]
        self[j, :] = sw
        if P is not None:
            sw = P[:, i].copy()
            P[:, i] = P[:, j].copy()
            P[:, j] = sw
    
    def swap_cols(self, i, j, Q=None):
        sw = self[:, i].copy()
        self[:, i] = self[:, j]
        self[:, j] = sw
        if Q is not None:
            sw = Q[i, :].copy()
            Q[i, :] = Q[j, :].copy()
            Q[j, :] = sw

    def subtract_rows(self, r1, r2, q, r=None, P=None):
        """ subtracting row r1 to q times row r2
        the inverse operation is operated on P
        sp that P * self is unchanged
        
        Major use case is when q, r is from the division
        self[r1, r2] = q* self[r2, r2] +r
        in that case we force the result of the operation to be
        r to avoid residual term. So r should be None
        except for this case
        """
        # self[r1, :] -= scalar_mult(q, self[r2, :])
        self[r1, :] -= q * self[r2, :]
        if r is not None:
            self[r1, r2] = r
        for ix in range(self.shape[1]):
            self[r1, ix] = sp.simplify(self[r1, ix])
        if P is not None:
            # P[:, r2] += scalar_mult(q, P[:, r1])
            P[:, r2] += q * P[:, r1]
            
    def subtract_cols(self, r1, r2, q, r=None, Q=None):
        """ subtracting row r1 to q times row r2
        the inverse operation is operated on P
        sp that P * self is unchanged
        
        Major use case is when q, r is from the division
        self[r1, r2] = q* self[r2, r2] +r
        in that case we force the result of the operation to be
        r to avoid residual term. So r should be None
        except for this case
        """

        # self[:, r1] -= scalar_mult(q, self[:, r2])
        self[:, r1] -= q * self[:, r2]
        if r is not None:
            self[r2, r1] = r
        for ix in range(self.shape[0]):
            self[ix, r1] = sp.simplify(self[ix, r1])
        if Q is not None:
            Q[r2, :] += q * Q[r1, :]

            
def cleanup(B):
    B1 = B
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B1[i, j] = sp.Poly(B[i, j], x, domain='QQ').as_expr()
    return B1


def _find_smallest_degree(A, cnt, n_row, n_col):
    position = (cnt, cnt)
    for i in xrange(cnt, n_row):
        for j in xrange(cnt, n_col):
            entry = A[i, j]
            if not is_zero_polynomial(entry):
                if is_zero_polynomial(A[position]):
                    position = (i, j)
                elif expr_degree(entry) < expr_degree(A[position]):
                    position = (i, j)
    return position


def _check_row_col_cleared(A, cnt, n_row, n_col):
    return _check_vector_cleared(A[:, cnt], cnt) and\
        _check_vector_cleared(A[cnt, :], cnt)


def _check_vector_cleared(poly_vec, cnt):
    cleared = True
    for i in range(poly_vec.shape[0]):
        if (i != cnt) and (not is_zero_polynomial(poly_vec[i])):
            return False
    return cleared


def is_zero_polynomial(poly):
    if poly == 0:
        return True
    pl = sp.Poly(poly, x)
    return (pl.degree() == 0) and (pl.coeffs()[-1] == 0)


def normalize_diagonal(P, A, Q):
    """Assuming A is diagonal. We put A in normalized
    Smith form by applying appropriate operations that
    keep PAQ unchanged and P, Q invertible"""
    n = min(A.shape)
    for i in range(n-1):
        for j in range(i+1, n):
            _normalize_one_pair_diagonal_entries(P, A, Q, i, j)


def scalar_mult(pol, vec):
    out = sp.zeros(*vec.shape, domain='QQ')
    for j in range(vec.shape[0]):
        out[j] = vec[j] * pol
    return out


def _normalize_one_pair_diagonal_entries(P, A, Q, i, j):
    """Normalize one pair of diagonal entries
    We note
    if a = ug, b = vg with s and t
    relatively prime:
    su + tv = 1
    then
      [ug 0]
      [0 vg]
    =
    [u -t]  [g   0] [su tv]
    [v  s]  [0 uvg] [-1    1]

    """
    g, s, t, u, v = polynomial_gcd(A[i, i], A[j, j])
    """
    new_p_i_col = scalar_mult(u, P[:, i]) + scalar_mult(v, P[:, j])
    new_p_j_col = scalar_mult(s, P[:, j]) - scalar_mult(t, P[:, i])
    """
    new_p_i_col = u * P[:, i] + v * P[:, j]
    new_p_j_col = s * P[:, j] - t * P[:, i]

    su = s * u
    tv = 1 - su
    # new_q_i_col = scalar_mult(su, Q[i, :]) + scalar_mult(tv, Q[j, :])
    new_q_i_row = su * Q[i, :] + tv * Q[j, :]
    # new_q_j_col = Q[j, :] - Q[i, :]
    new_q_j_row = Q[j, :] - Q[i, :]
    P[:, i] = new_p_i_col
    P[:, j] = new_p_j_col
    Q[i, :] = new_q_i_row
    Q[j, :] = new_q_j_row
    A[i, i] = g
    A[i, j] = 0
    A[j, i] = 0
    A[j, j] = u * A[j, j]
    

def polynomial_gcd(a, b):
    """Function to find gcd of two poly1d polynomials.
    Return gcd, s, t, u, v
    with a s + bt = gcd (Bezout s theorem)

    a = u gcd
    b = v gcd
    Hence
    s u + t v = 1
    These are used in diagimalize procedure
    """

    s = sp.Poly(0, x, domain='QQ').as_expr()
    old_s = sp.Poly(1, x, domain='QQ').as_expr()
    t = sp.Poly(1, x, domain='QQ').as_expr()
    old_t = sp.Poly(0, x, domain='QQ').as_expr()
    r = b
    old_r = a

    while not is_zero_polynomial(r):
        quotient, remainder = sp.div(old_r, r, x)
        (old_r, r) = (r, remainder)
        (old_s, s) = (s, old_s - quotient * s)
        (old_t, t) = (t, old_t - quotient * t)
    # output "Bézout coefficients:", (old_s, old_t)
    # output "greatest common divisor:", old_r
    # output "quotients by the gcd:", (t, s)
    u, _ = sp.div(a, old_r, x, domain='QQ')
    v, _ = sp.div(b, old_r, x, domain='QQ')
    return old_r.as_expr(), old_s.as_expr(),\
        old_t.as_expr(), u.as_expr(), v.as_expr()


def gen_random(row, col, diag_items, n_ops=5):
    """Generate a random matrix of size row col
    root mults add up to
    """
    # N_OPS = 6
    n = min(row, col)
    if n != len(diag_items):
        raise(ValueError(
            "diagonal size is not consistent with row %d and column %d" % (
                row, col)))
    A = SymPyPolynomialMatrix.zeros(row, col)
    for i in range(n):
        A[i, i] = diag_items[i]
    # for each side (right or left)
    # pick one of three operations:
    # subtract, swap, multiply
    choices = ['SUB', 'SWAP', 'MULT']

    action_list = np.random.choice(choices, n_ops)
    # doing rows
    for shp in [0, 1]:
        for a in action_list:
            if a == 'SUB':
                terms = np.random.choice([1, 2, 3], 1)
                p = array_to_poly(
                    np.random.randint(0, 10, size=terms+1))
                r1, r2 = tuple(np.random.choice(
                    np.arange(A.shape[shp]), 2, replace=False))
                if shp == 0:
                    A.subtract_rows(r1, r2, p)
                else:
                    A.subtract_cols(r1, r2, p)
            elif a == 'SWAP':
                r1, r2 = tuple(np.random.choice(
                    np.arange(A.shape[shp]), 2, replace=False))
                if shp == 0:
                    A.swap_rows(r1, r2)
                else:
                    A.swap_cols(r1, r2)
            else:
                r1 = np.random.choice(
                    np.arange(A.shape[shp]), 1, replace=False)[0]
                f = sp.Rational(np.random.randint(1, 5), 5)
                if shp == 0:
                    # A[r1, :] = scalar_mult(f, A[r1, :])
                    A[r1, :] = f * A[r1, :]
                else:
                    # A[:, r1] = scalar_mult(f, A[:, r1])
                    A[:, r1] = f * A[:, r1]
    return A


def gen_stable_polyomial(d, k):
    """Generate $Phi_1,...,Phi_n$
    such that  $s^n -Phi_1 s^{n-1}-...-Phi_n$
    has roots inside the unit circle.
    The VAR equation is
    $Y_t -Phi_1 Y_{t-1}-...-Phi_nY_{t-n} = epsilon_t$
    """
    pass


def round_rational(M, factor):
    M1 = sp.Matrix.zeros(*M.shape)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            M1[i, j] = sp.Rational((M[i, j] * factor).round(0), factor)
    return M1


def gen_a_random_matrix(k):
    """Generate univariate random scalar matrix
    with integer coefficients. This will reduce the
    calculation load later
    """
    """
    V0 = np.random.normal(size=k*k).reshape(k, k)
    a_det = np.linalg.det(V0)
    if a_det < 0:
        sw = V0[:, 0].copy()
        V0[:, 0] = V0[:, 1]
        V0[:, 1] = sw
        a_det = - a_det
    V0 /= np.exp(np.log(a_det) / k)
    D0 = round_rational(V0, factor)
    """
    max_b = 5
    min_b = -5
    V = np.random.randint(min_b, max_b, size=k*k-k)
    L = SymPyPolynomialMatrix.eye(k, domain='ZZ')
    U = SymPyPolynomialMatrix.eye(k, domain='ZZ')
    for i in range(k):
        for j in range(i):
            L[i, j] = V[i*(i-1) // 2 + j]
            U[j, i] = V[V.shape[0] // 2 + i*(i-1) // 2 + j]
    return L * U


def gen_unimodular(k, r, factor):
    """Generate a unimodular matrix of size k and
    and jordan size r of order 1
    s + J. This gives rise to 1 + JL which is unimodular
    """
    b = sp.Matrix.zeros(k, k)
    # b[:, :, 0] = np.eye(k)
    for i in range(r):
        b[i, i+1] = 1
    V0 = gen_a_random_matrix(k)
    V0a = V0.inv()
    h = cleanup(V0 * b * V0a)
    
    return SymPyPolynomialMatrix.diag(k * [x]) + h


def gen_simple_rr(k, mult, factor):
    """of form sI - H
    H has eigenvalue of multiplicity mult
    """
    dg = np.zeros((k))
    c = 0
    
    for m in mult:
        dg[c:c+m] = np.random.uniform(-1, 1, 1)
        c = c + m
    V0 = gen_a_random_matrix(k)
    V0a = V0.inv()

    b = SymPyPolynomialMatrix.diag(k*[x])
    dgr = [sp.Rational(round(a * factor), factor) for a in dg]
    h = cleanup(V0 * sp.Matrix.diag(dgr) * V0a)

    return b + h
    

def _test_one_stable():
    # degree invertible with nilpotenpart degree 1 rank 2
    # rank 3, degree 1
    # rank 3, degree 1
    # total k = 12
    factor = 1024
    np.random.seed(0)
    k = 10
    rj0 = 2
    rj1 = 3
    rj2 = 1

    U0 = gen_unimodular(k, rj0, factor)
    U1 = gen_unimodular(k, rj1, factor)
    rr0 = [2, 1]
    rr1 = [3]
    R0 = gen_simple_rr(k, rr0, factor)
    R1 = gen_simple_rr(k, rr1, factor)
    U2 = gen_unimodular(k, rj2, factor)
    
    ret = cleanup(U0 * R0)
    ret = cleanup(ret * U1)
    # ret = cleanup(ret * R1)
    # ret = cleanup(ret * U2)
    return ret
    

def array_to_poly(arr, factor=None):
    if factor is None:
        return sp.Poly(arr, x, domain='QQ').as_expr()
    return sp.Poly([sp.Rational(round(a * factor), factor)
                    for a in arr], x, domain='QQ').as_expr()


def expr_degree(expr):
    return sp.Poly(expr, x).degree()


def convert_T_to_Phi(T):
    # Phi = SymPyPolynomialMatrix.zeros(*T.shape)
    dtop = T.degree()
    top_term = sp.Poly([1] + (dtop)*[0], x, domain='QQ').as_expr()
    return SymPyPolynomialMatrix.diag(T.shape[0]*[top_term]) - T
    """
    for i in xrange(Phi.shape[0]):
        for j in xrange(Phi.shape[1]):
            if i == j:
                Phi[i, j], rmd = sp.div(
                    top_term - T[i, j], x, x)
            else:
                Phi[i, j], rmd = sp.div(
                    - T[i, j], x, x)
            if not is_zero_polynomial(rmd):
                print("Non zero remainder i=%d j=%d" % (i, j))
    return Phi
    """            


def _test_one_stable_1():
    T = _test_one_stable_2()
    # T is stable polynomial of order 5 with highest order 1
    # the rational function rT(s) = s^{-5} T(s) is proper
    # prT(s) = rT(s) - 1 is strictly proper
    # the polynomial matrix Ti(L) = rT(1/L) has Ti(0) = I_k

    poly_Phi = convert_T_to_Phi(T)
    P, A, Q, det_factor = poly_Phi.smith_mcmillan_form()
        

def gen_unimodular_pol(k, d):
    max_b = 5
    min_b = -5
    V0 = np.random.randint(min_b, max_b, size=(k*k-k)*(d+1)).reshape(
        (k*k-k), d+1)
    V = [sp.Poly(V0[i, :], x, domain='ZZ').as_expr() for i in range(k*k-k)]
    L = SymPyPolynomialMatrix.eye(k, domain='ZZ')
    U = SymPyPolynomialMatrix.eye(k, domain='ZZ')
    for i in range(k):
        for j in range(i):
            L[i, j] = V[i*(i-1) // 2 + j]
            U[j, i] = V[len(V) // 2 + i*(i-1) // 2 + j]
    return cleanup(L * U)
    

def gen_simple_rr_pol(k, mult):
    """of form sI - H
    H has eigenvalue of multiplicity mult
    """
    
    dg = []
    c = 0
    
    for m in mult:
        dg += m * [np.random.choice([-2, -3, 2, 3], 1, replace=True)]
        c = c + m
    dg += (k-len(dg))*[0]
    # V0 = gen_a_random_matrix(k)
    # V0a = V0.inv()

    b = SymPyPolynomialMatrix.diag(k*[x])
    # dgr = [sp.Rational(round(a * factor), factor) for a in dg]
    # h = cleanup(V0 * sp.Matrix.diag(dg) * V0a)
    h = SymPyPolynomialMatrix.diag(dg)
    return b + h


def _test_one_stable_2():
    k = 10
    # d = 1
    np.random.seed(0)
    d0 = 1
    d1 = 1
    d2 = 1

    U0 = gen_unimodular_pol(k, d0)
    U1 = gen_unimodular_pol(k, d1)
    rr0 = [2, 1]
    rr1 = [3]
    R0 = gen_simple_rr_pol(k, rr0)
    # R1 = gen_simple_rr_pol(k, rr1)
    # U2 = gen_unimodular_pol(k, d2)
    
    ret = cleanup(U0 * R0)
    ret = cleanup(ret * U1)
    # ret = cleanup(ret * R1)
    # ret = cleanup(ret * U2)
    return ret


if __name__ == '__main__':
    def _test_diag():
        np.random.seed(1)
        B = np.random.choice(
            np.arange(10), 3*3*5, replace=True).reshape(3, 3, 5)
        B_mat = SymPyPolynomialMatrix.coef_array_to_SPMatrix(B)
        P, A, Q, det_factor = B_mat.smith_mcmillan_form()
        B_mat1 = cleanup(P * A * Q)
        sp.pprint(B_mat1 - B_mat)
        normalize_diagonal(P, A, Q)
        B_mat2 = cleanup(P * A * Q)
        sp.pprint(B_mat2 - B_mat)

    def _test1():
        A_mat = SymPyPolynomialMatrix.diag(
            [sp.Poly([1, 0], x, domain='QQ').as_expr(), 1])
        P = SymPyPolynomialMatrix.eye(2, domain='QQ')
        Q = SymPyPolynomialMatrix.eye(2, domain='QQ')
        normalize_diagonal(P, A_mat, Q)
        sp.pprint(P * A_mat * Q)

    def _test_random():
        np.random.seed(3)
        diag_list = [array_to_poly([2, 1]),
                     array_to_poly([1, 3, 2]),
                     array_to_poly([1, 3])]
        T = gen_random(
            3, 3, diag_list, n_ops=6)
        sp.pprint(T)
        P, A, Q, det_factor = T.smith_mcmillan_form()
        sp.pprint(P * A * Q - T)
        sp.pprint(sp.Matrix.det(T))
        det = det_factor
        for i in range(A.shape[0]):
            det *= A[i, i]
        sp.pprint(det)

    def to_poly_matrix(X):
        ret = sp.Matrix.zeros(X.shape[0], X.shape[1])
        for i in xrange(ret.shape[0]):
            for j in xrange(ret.shape[1]):
                ret[i, j] = sp.Poly(X[i, j], x, domain='QQ')
        return ret
        
    def quick_mult(X, Y):
        ret = SymPyPolynomialMatrix.zeros(X.shape[0], Y.shape[1])
        for i in xrange(ret.shape[0]):
            for j in xrange(ret.shape[1]):
                ret[i, j] = sp.Poly(sp.simplify(
                    sp.Matrix.dot(X[i, :], Y[:, j])), x, domain='QQ')
        return ret
            
    np.random.seed(0)
    k = 10
    k2 = 10
    B = np.random.choice(
        np.arange(10), k*k2*2, replace=True).reshape(k, k2, 2)
    # B = choice(np.arange(8), 2*2*2, replace=True).reshape(2, 2, 2)
    B_mat = SymPyPolynomialMatrix.coef_array_to_SPMatrix(B)
    sp.pprint(B_mat)

    # ret = [B_mat.subs(x1) for x1 in xrange(5)]
    # sp.pprint(ret)

    P, A, Q, det_factor = B_mat.smith_mcmillan_form()
    """
    P_poly = to_poly_matrix(P)
    A_poly = to_poly_matrix(A)
    Q_poly = to_poly_matrix(Q)

    Bm1 = quick_mult(P_poly, A_poly)
    Bm2 = quick_mult(Bm1, Q)
    """
    B_mat1a = cleanup(A * Q)
    B_mat1 = cleanup(P * B_mat1a)
    sp.pprint(B_mat1 - B_mat)

