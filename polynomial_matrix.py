from __future__ import print_function
import numpy as np
from numpy.random import normal, randint, choice, seed, uniform
from numpy import exp, pi, log
from numpy.fft import ifft
from numpy import poly1d, polydiv
from numpy.linalg import eig, solve, inv


"""Various construction for polynomial matrix
"""


class PolynomialMatrix(np.ndarray):
    """Input matrix of dimension 2 of type np.poly1d
    We could realize a polynomial matrix in two ways
    One is a matrix of poly1d which is the main method we use here.
    Another is as a 3d array with the third dimension is the polynomial
    dimension. We provide a few method to navigate between the two ways
    as it may be convenient to use the secondway sometime
    """

    def __new__(cls, *args, **kwargs):
        kwargs['dtype'] = poly1d
        return super(PolynomialMatrix, cls).__new__(
            cls, *args, **kwargs)
    
    def __init__(self, shape):
        """Input matrix of dimension 2 of type np.poly1d
        """
        if len(shape) != 2:
            raise(ValueError("matrix is not of dimension 2"))
        self.order = 0

    def calc_order(self):
        self.order = np.max([a.order for a in self.reshape(-1)])

    """
    def scalar_div(self, b):
        q = zeros(self.shape)
        r = zeros(self.shape)
        for i in range(self.shape[0]):
            for j in self.shape[1]:
                qx, rx = polydiv(self[i, j], b)
                q[i, j] = qx
                r[i, j] = rx
        return q, r
    """

    def pprint(self):
        for i in range(0, self.shape[0]):
            for j in range(0, self.shape[1]):
                print_poly(self[i, j])
                if j < self.shape[1] - 1:
                    print(", ", end='')
            print("")

    @classmethod
    def coef_array_to_PolynomialMatrix(cls, array, tolerance=None):
        m = PolynomialMatrix(array.shape[:2])
        if len(array.shape) == 2:
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    if tolerance is None or\
                       np.abs(array[i, j]) > tolerance:
                        m[i, j] = poly1d(
                            array[i, j])
                    else:
                        m[i, j] = poly1d(0)
            m.calc_order()
            return m
            
        if tolerance is None:
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    m[i, j] = poly1d(
                        array[i, j, :])
            m.calc_order()
            return m

        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                m[i, j] = poly1d(
                    cut_off_polynomial(
                        array[i, j, :], tolerance))
            m.calc_order()
        return m
    
    def PolynomialMatrix_to_3darray(self):
        """Create a 3d array of coefficients of the matrix
        The third dimension is order
        """
        ret = np.zeros(
            (self.shape[0], self.shape[1], self.order+1),
            dtype=float)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                ret[i, j, -self[i, j].coeffs.shape[0]:] =\
                    self[i, j].coeffs
        return ret

    def eval(self, x, dtype=float):
        """x is a vector
        """
        shape = self.shape
        ret = np.zeros(
            (shape[0], shape[1], x.shape[0]),
            dtype=dtype)
        for i in range(shape[0]):
            for j in range(shape[1]):
                ret[i, j, :] = np.polyval(self[i, j], x)
        return ret

    def determinant(self, real=True):
        """Evaluate the determinant of a polynomial matrix
        the return is a scalar polynomial.
        We return the full polynomial of degree
        k * p. In case of low McMillan degree the higher degree
        terms could be zeros. The caller should use cut off to
        reduce terms.
        The algorithm is to evaluate the determinant
        at kp+1 points (using fft) then use lagrange interpolation
        """
        if self.shape[0] != self.shape[1]:
            raise(
                ValueError("not a square matrix sizes are {}{}".format(
                    self.shape[:2])))
        k = self.shape[0]
        p = self.order
        d = k * p
        x = np.vectorize(lambda i: exp(-2j*pi*i/(d+1)))(
            np.arange(d+1))
        # evals = np.full((k, k, d+1), np.nan)
        evals = self.eval(x, dtype=np.complex)
        det_evals = [np.linalg.det(evals[:, :, j])
                     for j in range(evals.shape[2])]
        det_pol = ifft(det_evals)
        if real:
            det_pol = det_pol.real
        return poly1d(np.flip(det_pol, 0))

    def cut_off(self, tolerance):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                cc = self[i, j].coeffs
                cut = None
                top_cut = False
                for k in range(cc.shape[0]):
                    if np.abs(cc[k]) < tolerance:
                        cut = k
                        cc[k] = 0
                    if cut == 0 and cc.shape[0] > 1:
                        self[i, j] = poly1d(cc[1:])
                    elif cut is not None:
                        self[i, j] = poly1d(cc)
                        
    def smith_mcmillan_form(self, tolerance=None):
        """return P, A, Q
        such that self = P * A * Q
        P Q are unimodular and A is diagonal
        if numerator is None it returns the smith normal form
        """

        n_row, n_col = self.shape[:2]
        A = self.copy()
        n = min(n_row, n_col)
        cnt = 0
        det_factor = 1.
        P = PolynomialMatrix.coef_array_to_PolynomialMatrix(
            np.eye(n_row)[:, :, None])

        Q = PolynomialMatrix.coef_array_to_PolynomialMatrix(
            np.eye(n_col)[:, :, None])

        while cnt < n:
            cleared = False
            while not cleared:
                position = _find_smallest_degree(
                    A, cnt, n_row, n_col)
                cleared = _check_row_col_cleared(A, cnt, n_row, n_col)

                if position == (cnt, cnt) and cleared:
                    # entry = A[cnt, cnt]
                    # if (entry.order == 1) and (entry.coef[-1] == 0):
                    # cnt = n
                    if A[position].coef[0] != 0 and A[position].coef[0] != 1:
                        if False:
                            print("Divide row %d with %f" % (
                                cnt, A[position].coef[0]))
                            det_factor *= A[position].coef[0]
                            P[:, cnt] *= A[position].coef[0]
                            A[position] /= A[position].coef[0]

                else:
                    if cnt != position[0]:
                        A.swap_rows(cnt, position[0], P)
                        det_factor *= -1
                        """
                        aux = A[cnt, :].copy()
                        A[cnt, :] = A[position[0], :].copy()
                        A[position[0], :] = aux
                        aux = P[:, cnt].copy()
                        P[:, cnt] = P[:, position[0]].copy()
                        P[:, position[0]] = aux
                        """

                    if cnt != position[1]:
                        A.swap_cols(cnt, position[1], Q)
                        det_factor *= -1
                        """
                        aux = A[:, cnt].copy()
                        A[:, cnt] = A[:, position[1]].copy()
                        A[:, position[1]] = aux

                        aux = Q[cnt, :].copy()
                        Q[cnt, :] = Q[position[1], :].copy()
                        Q[position[1], :] = aux
                        """

                    for i in range(cnt + 1, n_row):
                        q, r = polydiv(
                            A[i, cnt], A[cnt, cnt])
                        if A[cnt, cnt].order == 0:
                            r = np.poly1d(0)
                        A.subtract_rows(i, cnt, q, r, P)
                        if tolerance is not None:
                            A.cut_off(tolerance)

                        """
                        for j in range(n_col):
                            if (j == cnt):
                                A[i, j] = r
                            else:
                                A[i, j] -= q * A[cnt, j]
                        for j in range(n_row):
                            P[j, cnt] += q * P[j, i]
                        """
                        if not is_zero_polynomial(q):
                            print(
                                "Subtract row %d, to row %d (" % (
                                    i+1, cnt+1), end='')
                            print_poly(q)
                            print(")")
                    for i in range(cnt + 1, n_col):
                        q, r = polydiv(
                            A[cnt, i],
                            A[cnt, cnt])
                        if A[cnt, cnt].order == 0:
                            r = np.poly1d(0)
                        A.subtract_cols(i, cnt, q, r, Q)
                        if tolerance is not None:
                            A.cut_off(tolerance)

                        """
                        for j in range(n_row):
                            if j == cnt:
                                A[j, i] = r
                            else:
                                A[j, i] -= q * A[j, cnt]
                        for j in range(n_col):
                            Q[cnt, j] += q * Q[i, j]
                        """
                        if not is_zero_polynomial(q):
                            print(
                                "subtract column %d, to column %d (" % (
                                    i+1, cnt+1), end='')
                            print_poly(q)
                            print(")")

            cnt += 1
            P.calc_order()
            Q.calc_order()
            A.calc_order()
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
        self[r1, :] -= scalar_mult(q, self[r2, :])
        if r is not None:
            self[r1, r2] = r
        if P is not None:
            P[:, r2] += scalar_mult(q, P[:, r1])
            
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

        self[:, r1] -= scalar_mult(q, self[:, r2])
        if r is not None:
            self[r2, r1] = r
        if Q is not None:
            Q[r2, :] += scalar_mult(q, Q[r1, :])

            
def pprint(mat3d):
    for l in range(mat3d.shape[2]):
        print(mat3d[:, :, l])
        print('______')


def mat_mult(A, B):
    ret = np.dot(A, B)
    ret.calc_order()
    return ret

    
def eval_array(array3d, x):
    """x is a vector
    Evaluate array3d as a matrix polynomial
    """
    ret = np.zeros(
        (array3d.shape[0], array3d.shape[1], x.shape[0]),
        dtype=float)
    ret = ret + array3d[:, :, 0][:, :, None]
    for i in range(1, array3d.shape[2]):
        ret = ret * x[None, None, :] + array3d[:, :, i][:, :, None]
    return ret


def pprint_array(array3d):
    for i in range(array3d.shape[2]):
        print(array3d[:, :, i])
        print("______")


def cut_off_array(array3d, cut_off_level):
    d = array3d.shape[2]
    for i in range(d):
        ls = np.where(np.abs(array3d[:, :, i]))[0].shape[0]
        if ls > 0:
            return array3d[:, :, i:].copy()
    return array3d.copy()


def cut_off_polynomial(pol, cut_off_level):
    big_array = np.where(np.abs(pol.coeffs) > cut_off_level)[0]
    if big_array.shape[0] == 0:
        return poly1d([0])
    return poly1d(pol.coeffs[big_array[0]:])


def determinant_array(array3d, real=True):
    """Evaluate the determinant of a polynomial matrix
    the return is a scalar polynomial.
    We return the full polynomial of degree
    k * p. In case of low McMillan degree the higher degree
    terms could be zeros. The caller should use cut off to
    reduce terms.
    The algorithm is to evaluate the determinant
    at kp+1 points (using fft) then use lagrange interpolation
    """
    if array3d.shape[0] != array3d.shape[1]:
        raise(
            ValueError("not a square matrix sizes are {}{}".format(
                array3d.shape[:2])))
    k = array3d.shape[0]
    p = array3d.shape[2]
    d = k * p
    x = np.vectorize(lambda i: exp(-2j*pi*i/(d+1)))(
        np.arange(d))
    evals = np.full((k, k, d+1), np.nan)
    evals = eval_array(array3d, x)
    det_pol = fft(evals)
    if real:
        det_pol = det_pol.real
    return det_pol


def print_poly(poly):
    strpoly = ""
    # counter = poly.order
    coefs = poly.coeffs

    first = True

    for i in range(0, poly.order + 1):
        if coefs[i] != 0:
            if not first:
                strpoly += " "
                if coefs[i] > 0:
                    strpoly += "+"

            if not (abs(coefs[i]) == 1 and i < poly.order):
                strpoly += str(coefs[i])
            elif coefs[i] == -1:
                strpoly += "-"

            if i < poly.order:
                strpoly += "x"
                if i < poly.order - 1:
                    strpoly += str(poly.order - i)

            first = False

    if strpoly == "":
        strpoly = "0"

    print(strpoly, end='')


def _find_smallest_degree(A, cnt, n_row, n_col):
    position = (cnt, cnt)
    for i in range(cnt, n_row):
        for j in range(cnt, n_col):
            entry = A[i, j]
            if not is_zero_polynomial(entry):
                if is_zero_polynomial(A[position]):
                    position = (i, j)
                elif entry.order < A[position].order:
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
    return (poly.order == 0) and (poly.coeffs[-1] == 0)


def normalize_diagonal(P, A, Q):
    """Assuming A is diagonal. We put A in normalized
    Smith form by applying appropriate operations that
    keep PAQ unchanged and P, Q invertible"""
    n = min(A.shape)
    for i in range(n-1):
        for j in range(i+1, n):
            _normalize_one_pair_diagonal_entries(P, A, Q, i, j)


def scalar_mult(pol, vec):
    out = np.empty_like(vec)
    for j in range(vec.shape[0]):
        out[j] = vec[j] * pol
    return out


def scalar_div(M, b):
    q = zeros(M.shape)
    r = zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            qx, rx = polydiv(M[i, j], b)
            q[i, j] = qx
            r[i, j] = rx
    return q, r


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
    new_p_i_col = scalar_mult(u, P[:, i]) + scalar_mult(v, P[:, j])
    new_p_j_col = scalar_mult(s, P[:, j]) - scalar_mult(t, P[:, i])
    su = s * u
    tv = 1 - su
    new_q_i_col = scalar_mult(su, Q[i, :]) + scalar_mult(tv, Q[j, :])
    new_q_j_col = Q[j, :] - Q[i, :]
    P[:, i] = new_p_i_col
    P[:, j] = new_p_j_col
    Q[i, :] = new_q_i_col
    Q[j, :] = new_q_j_col
    A[i, i] = g
    A[i, j] = poly1d([0])
    A[j, i] = poly1d([0])
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

    s = poly1d(0)
    old_s = poly1d(1)
    t = poly1d(1)
    old_t = poly1d(0)
    r = b
    old_r = a

    while not is_zero_polynomial(r):
        quotient, remainder = polydiv(old_r, r)
        (old_r, r) = (r, remainder)
        (old_s, s) = (s, old_s - quotient * s)
        (old_t, t) = (t, old_t - quotient * t)
    u, _ = polydiv(a, old_r)
    v, _ = polydiv(b, old_r)
    return old_r, old_s, old_t, u, v


def diag(poly_array):
    n = len(poly_array)
    ret = PolynomialMatrix.coef_array_to_PolynomialMatrix(
        np.zeros((n, n)))
    for i in range(n):
        ret[i, i] = poly_array[i]
    ret.calc_order()
    return ret


def eye(n):
    return PolynomialMatrix.coef_array_to_PolynomialMatrix(
        np.eye(n))


def zeros(shape):
    return PolynomialMatrix.coef_array_to_PolynomialMatrix(
        np.zeros(shape))


def ones(shape):
    return PolynomialMatrix.coef_array_to_PolynomialMatrix(
        np.ones(shape))


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
    A = zeros((row, col))
    np.fill_diagonal(A, diag_items)
    # for each side (right or left)
    # pick one of three operations:
    # subtract, swap, multiply
    choices = ['SUB', 'SWAP', 'MULT']

    action_list = choice(choices, n_ops)
    # doing rows
    for shp in [0, 1]:
        for a in action_list:
            if a == 'SUB':
                terms = choice([1, 2, 3], 1)
                p = poly1d(randint(0, 10, size=terms+1))
                r1, r2 = tuple(choice(
                    np.arange(A.shape[shp]), 2, replace=False))
                if shp == 0:
                    A.subtract_rows(r1, r2, p)
                else:
                    A.subtract_cols(r1, r2, p)
            elif a == 'SWAP':
                r1, r2 = tuple(choice(
                    np.arange(A.shape[shp]), 2, replace=False))
                if shp == 0:
                    A.swap_rows(r1, r2)
                else:
                    A.swap_cols(r1, r2)
            else:
                r1 = tuple(choice(
                    np.arange(A.shape[shp]), 1, replace=False))
                f = randint(1, 5) / 5.
                if shp == 0:
                    A[r1, :] = scalar_mult(f, A[r1, :])
                else:
                    A[:, r1] = scalar_mult(f, A[:, r1])
    A.calc_order()
    return A


def gen_a_random_matrix(k):
    V0 = normal(size=k*k).reshape(k, k)
    a_det = np.linalg.det(V0)
    if a_det < 0:
        sw = V0[:, 0].copy()
        V0[:, 0] = V0[:, 1]
        V0[:, 1] = sw
        a_det = - a_det
    return V0 / exp(log(a_det) / k)


def gen_unimodular(k, r):
    """Generate a unimodular matrix of size k and
    and jordan size r of order 1
    s + J. This gives rise to 1 + JL which is unimodular
    """
    b = np.zeros((k, k, 2), dtype=float)
    b[:, :, 0] = np.eye(k)
    np.fill_diagonal(b[:r, 1:r+1, 1], 1)
    V0 = gen_a_random_matrix(k)
    V0a = inv(V0)
    # b[:, :, 0] = np.dot(np.dot(V0, b[:, :, 0]) V0a))
    b[:, :, 1] = np.dot(np.dot(V0, b[:, :, 1]), V0a)
    
    return PolynomialMatrix.coef_array_to_PolynomialMatrix(b)


def gen_simple_rr(k, mult):
    """of form sI - H
    H has eigenvalue of multiplicity mult
    """
    dg = np.zeros((k), dtype=float)
    c = 0
    
    for m in mult:
        dg[c:c+m] = uniform(-1, 1, 1)
        c = c + m
    V0 = gen_a_random_matrix(k)
    V0a = inv(V0)
    
    b = np.zeros((k, k, 2), dtype=float)
    b[:, :, 1] = np.dot(np.dot(V0, np.diag(dg)), V0a)
    b[:, :, 0] = np.eye(k)
    return PolynomialMatrix.coef_array_to_PolynomialMatrix(b)
    

def _test_one_stable():
    # degree invertible with nilpotenpart degree 1 rank 2
    # rank 3, degree 1
    # rank 3, degree 1
    # total k = 12
    seed(0)
    k = 12
    rj0 = 2
    rj1 = 3
    rj2 = 1

    U0 = gen_unimodular(k, rj0)
    U1 = gen_unimodular(k, rj1)
    rr0 = [2, 1]
    rr1 = [3]
    R0 = gen_simple_rr(k, rr0)
    R1 = gen_simple_rr(k, rr1)
    U2 = gen_unimodular(k, rj2)
    
    ret = np.dot(np.dot(np.dot(
        np.dot(U0, R0), U1), R1), U2)
    ret.calc_order()
    return ret
    

def _test_one_stable_1():
    T = _test_one_stable()
    # T is stable polynomial of order 5 with highest order 1
    # the rational function rT(s) = s^{-5} T(s) is proper
    # prT(s) = rT(s) - 1 is strictly proper
    # the polynomial matrix Ti(L) = rT(1/L) has Ti(0) = I_k
    
    arr_T = T.PolynomialMatrix_to_3darray()
    poly_Phi = PolynomialMatrix.coef_array_to_PolynomialMatrix(
        arr_T[:, :, 1:])
    P, A, Q, det_factor = poly_Phi.smith_mcmillan_form(1e-8)
        

def split_E_Psi(A, root, mult):
    """ A is diagonal and normalized
    under the form diag(a_1, ..., a_k)
    with $a_i | a_{i+1}$.
    We consider (s-root)^{-mult} A. We reduc
    the fractions $a_i / (s-root)^{mult}$
    to irreducibel form
    $e_i / (s-root)^l$
    We collect terms the non zeros $e_i$ to form $E$

    Psi is returned as a list
    [(l1, mult_1), ...(l_f, mult_f)]
    l_1, ... are powers of (s - root)
    whitch are mult - zero_order after simplifcation
    mult_i denotes the number of terms with
    denominator (s - root)^{l_i}
    l_1 * mult_1 + ... l_f * mult_f = McMillan degree
    mult_1 + ... + mult_f = E.shape[0]
    """
    Psi = []
    E = []
    k = min(A.shape[0], A.shape[1])
    b = poly1d([1, -root])
    current_zero_order = 0
    current_multiple = 0
    for i in range(k):
        if is_zero_polynomial(A[i, i]):
            Psi.append((
                mult-current_zero_order, current_multiple))
            return E, Psi
        else:
            ratio, zero_order = find_zero_order(A[i, i], b)
            E.append(ratio)
            if zero_order == current_zero_order:
                current_multiple = current_multiple + 1
            else:
                if current_multiple > 0:
                    Psi.append((mult-current_zero_order,
                                current_multiple))
                current_zero_order = zero_order
                current_multiple = 1
    if current_multiple > 0:
        Psi.append((
            mult-current_zero_order,
            current_multiple))
    return E, Psi


def find_zero_order(p, b):
    ratio = None
    zero_order = 0
    r = 0
    current_p = p
    Found = False
    old_ratio = p
    while not Found:
        ratio, r = polydiv(current_p, b)
        current_p = ratio
        if r.coeffs[0] != 0:
            Found = True
        else:
            zero_order = zero_order + 1
            old_ratio = ratio
    
    return old_ratio, zero_order


def poly_taylor_expansion(p, root):
    ret = []
    current_p = p
    b = poly1d([1, -root])
    while current_p.order != 0:
        q, r = polydiv(current_p, b)
        ret.append(r)
        current_p = q
    ret.append(current_p)
    return ret.reverse()


def Taylor_expansion(M, root, mult):
    """Expanding the polynomial matrix M
    to taylor series around root. Cut off after
    mult
    """
    ret = []
    current_M = M
    b = poly1d([1, -root])
    M.calc_order()
    m_order = M.order
    order = m_order
    while order >= max(0, m_order - mult + 1):
        q, r = scalar_div(current_M, b)
        ret.append(r.PolynomialMatrix_to_3darray()[:, :, 0])
        current_M = q
        order -= 1
    # ret.append(current_M)
    # ret.reverse()
    return ret


def calc_McMillan_degree(Psi):
    return sum([Psi[i][0] * Psi[i][1]
                for i in range(len(Psi))])


def minimal_realizeation(P, A, Q, T, root=0, mult=None):
    """ Deriving the Kalman's minimal state
    realization for the rational
    matrix P*A*Q / (s-root)^{mult}.
    P, A, Q come from a Smith normal form decomposition.
    P and Q are therefore unimodular
    A is of degree p-1.
    We also assume s^p - PAQ to be stable.
    mult >= p.
    If mult is None we set it to p.

    Steps:
    *** Assuming A is normalized
    *** Reduce by factors of (s-roots)
    *** Apply the formula by Kalman
    Result is H, F, G and that we can verify numerically that
    PAQ / (s-root)^{mult} = H(sI-F)^{-1}G

    """
    if mult is None:
        mult = T.order + 1
    k = P.shape[0]
    E, Psi = split_E_Psi(A, root, mult)

    w = len(E)  # width
    mm_degree = calc_McMillan_degree(Psi)
    f = len(Psi)
    PE = mat_mult(P[:, :w], diag(E))
    QE = Q[:w, :]
    H = np.zeros((k, mm_degree), dtype=float)
    G = np.zeros((mm_degree, k), dtype=float)
    F = np.zeros((mm_degree, mm_degree), dtype=float)
    pol_start = 0  # start for the PE / QE blocks (polynomial)
    rel_start = 0  # start for the H and G blocks (realization)

    for rho in range(f):
        l_rho, m_rho = Psi[rho]
        """
        The block corresponding to rho
        is [start, start + l_rho * m_rho)
        """
        rel_end = rel_start + l_rho * m_rho
        tH = Taylor_expansion(
            PE[:, pol_start:pol_start+m_rho],
            root, l_rho)
        for lx in range(min(len(tH), l_rho)):
            H[:, rel_start+m_rho*lx:rel_start+m_rho*(lx+1)] = tH[lx]
        tG = Taylor_expansion(
            QE[pol_start:pol_start+m_rho, :],
            root, l_rho)
        n_G = len(tG)
        for lx in range(min(l_rho, n_G)):
            G[rel_end-m_rho*(lx+1):rel_end-m_rho*lx, :] = tG[lx]
        eye_m = np.eye(m_rho)
        for lx in range(l_rho-1):
            F[rel_start+lx*m_rho:rel_start+(lx+1)*m_rho,
              rel_start+(lx+1)*m_rho:rel_start+(lx+2)*m_rho] = eye_m
        np.fill_diagonal(F[rel_start:rel_end], root)
        rel_start = rel_end
        pol_start += m_rho

    return H, G, F, Psi


def state_to_Phi(H, F, G, Psi):
    """ Combining state form to the regular Phi form
    return Phi, and root of the corresponding
    system. Returning s^pH(sI-F)^{-1}G
    """
    k = H.shape[0]
    m = G.shape[1]
    mm = H.shape[1]
    p = Psi[0][0]
    Phi_arr = np.zeros((k, m, p))
    F_i = np.eye(mm)
    for i in range(p):
        Phi_arr[:, :, p-i-1] = np.dot(H, np.dot(F_i, G))
        F_i = np.dot(F_i, F)
    return PolynomialMatrix.coef_array_to_PolynomialMatrix(Phi_arr)


def calc_full_transfer_function(Phi, p):
    """Calc s^p - Phi
    """
    s_p = poly1d([1] + p * [0])
    T = diag(Phi.shape[0] * [s_p]) - Phi
    T.calc_order()
    return T


def check_stable(Phi, p):
    """Check that s^p - Phi is stable
    If Psi is not None cut off the determinant
    at mcmillant
    """
    T = calc_full_transfer_function(Phi, p)
    dd = T.determinant()
    """
    if Psi is not None:
        mm_degree = calc_McMillan_degree(Psi)
        dd = poly1d[dd.coeffs()[:mm_degree+1]]
    """
    roots = np.roots(dd)
    is_stable = np.where((np.absolute(roots) < 1))[0].shape[0] == dd.order
    return is_stable, roots, dd
    

def Gilbert_realization(P, A, Q, denominator):
    """ Deriving the Gilbert realization for the polynomial matrix m.
    P, A, Q come from a smith normal form decomposition.
    P and Q are therefore unimodular
    A is of degree p-1.
    We also assume s^p - PAQ to be stable.
    Otherwise it will have the form of
    [(root_1, mult_1), ..., (root_l, mult_l)]
    the d enominator will have form (s-root_1)^{mult_1) ... (s-root_l)^{mult_l)
    we assume sum mult_i >= p (strictly properness)
    """
    pass


def gen_unimodular_pol(k, d):
    max_b = 5
    min_b = -5
    V0 = np.random.randint(min_b, max_b, size=(k*k-k)*(d+1)).reshape(
        (k*k-k), d+1)
    V = [poly1d(V0[i, :]) for i in range(k*k-k)]
    L = eye(k)
    U = eye(k)
    for i in range(k):
        for j in range(i):
            L[i, j] = V[i*(i-1) // 2 + j]
            U[j, i] = V[len(V) // 2 + i*(i-1) // 2 + j]
    return mat_mult(L, U), L, U


def inverse_triangular(M, trig_type):
    """Inverting a triangular polynomial matrix.
    Assuming the diagonal entries are all scalar and invertible
    """
    if M.shape[0] != M.shape[1]:
        raise(ValueError('not a square matrix %d %d ' % M.shape))
    k = M.shape[0]
    inv_diag = np.zeros(k, dtype=float)
    if trig_type.lower() == 'l':
        ret = zeros(M.shape)
        for i in range(M.shape[0]):
            if M[i, i].order > 0 or M[i, i].coeffs[0] == 0:
                raise(ValueError(
                    ("Matrix is not invertible."
                     "Diagonal element %d is zero or non scalar" % i)))
            inv_diag[i] = 1 / M[i, i].coeffs[0]
            ret[i, i] = poly1d(inv_diag[i])
            for j in range(i):
                prd = sum(M[i, i-j-1:i] *
                          ret[i-j-1:i, i-j-1])
                ret[i, i-j-1] = prd / (-inv_diag[i])
        return ret
    elif trig_type.lower() == 'u':
        return inverse_triangular(M.T, 'l').T
    else:
        raise(ValueError('trig_type %s must be l or u' % trig_type))


def test_minimal_realization():
    # Total rank 8
    # 2 wi th s^{-3}: (terms are (s -r1), (s-r1) (s-r2)
    # 1 with s^{-2}:  s(s-r1) (s-r2) (s-r3)
    # 3 with s^{-1} : s^2(s-r1) (s-r2) (s-r3)
    # 2 terms with zero- > max degree 5
    # Start with A. P generated by L, U, Q is just L,
    # U+some deviation. Small enough so not a problem.
    # Total have degree 8 (?)
    # Then run the reduction over
    seed(0)
    roots = randint(-15, 15, size=3) / 16.
    k = 8
    p = [poly1d([1, r]) for r in roots]
    p_s = poly1d([1, 0])
    p_s_2 = poly1d([1, 0, 0])
    if True:
        A = diag(
            [p[0],
             p[0] * p[1],
             p_s * p[0] * p[1] * p[2],
             p_s_2 * p[0] * p[1] * p[2],
             p_s_2 * p[0] * p[1] * p[2],
             p_s_2 * p[0] * p[1] * p[2],
             poly1d([0]),
             poly1d([0])])
        """
        A = diag(
            [p[0] * p[1],
             p[0] * p[1],
             p[0] * p[1]] +
            [p_s * p[0] * p[1] * p[2]] +
            (k-4)*[poly1d([0])])
        """
    else:
        # p[0],
        A = diag(
            [p[0] * p[1],
             p[0] * p[1],
             p[0] * p[1]] + (k-3)*[poly1d([0])])

    P, L, U = gen_unimodular_pol(k, 1)
    U1 = inverse_triangular(U, 'U')
    L1 = inverse_triangular(L, 'L')
    noise = zeros((k, k))
    eps = 1e-1
    for i in range(k):
        for j in range(i-1):
            noise[i, j] = poly1d(normal(0, eps, size=1))

    Q = mat_mult(U1, (L1 + noise))
    # Q, Lq, Uq = gen_unimodular_pol(k, 2)
    T = mat_mult(mat_mult(P, A), Q)
    H, G, F, Psi = minimal_realizeation(
        P, A, Q, T, root=0, mult=None)

    # we now test the result that
    # (s-root)^{-mult} (P A Q ) = H((sI -F)^{-1}G
    # we will verify this numerically for a range of s

    T_order = T.order

    for s in range(1, k * k * 2):
        root = s * 1. / (k * k+1)
        inv_root_power = (1/root) ** (T_order+1)
        V1 = T.eval(np.array([root]))[:, :, 0] * inv_root_power
        middle_term = root * np.eye(F.shape[0]) - F
        V2 = np.dot(H, np.linalg.solve(middle_term, G))
        print('root=%f diff=%f' % (root, np.sum(np.abs(V1 - V2))))


def calc_Jordan_matrix(Psi, root):
    start = 0
    mm_degree = calc_McMillan_degree(Psi)
    F = np.zeros((mm_degree, mm_degree), dtype=float)
    
    for rho in range(len(Psi)):
        l_rho, m_rho = Psi[rho]
        end = start + l_rho * m_rho
        eye_m = np.eye(m_rho)
        for lx in range(l_rho-1):
            F[start+lx*m_rho:start+(lx+1)*m_rho,
              start+(lx+1)*m_rho:start+(lx+2)*m_rho] = eye_m
        np.fill_diagonal(F[start:end], root)
        start += l_rho * m_rho
    return F

        
if __name__ == '__main__':
    def _test_diag(p):
        seed(1)
        B = choice(np.arange(10), 3*3*5, replace=True).reshape(3, 3, 5)
        B_mat = PolynomialMatrix.coef_array_to_PolynomialMatrix(B)
        P, A, Q, det_factor = B_mat.smith_mcmillan_form()
        (np.dot(np.dot(P, A), Q) - B_mat).pprint()
        normalize_diagonal(P, A, Q)

    def _test1():
        A_mat = diag([poly1d([1, 0]), poly1d([1])])
        P = eye(2)
        Q = eye(2)
        normalize_diagonal(P, A_mat, Q)
        np.dot(np.dot(P, A_mat), Q).pprint()

    def _test_random():
        seed(3)
        diag_list = np.array(
            [poly1d([2, 1]), poly1d([1, 3, 2]), poly1d([1, 3])])
        T = gen_random(
            3, 3, diag_list, n_ops=6)
        T.pprint()
        P, A, Q, det_factor = T.smith_mcmillan_form(1e-10)
        (np.dot(np.dot(P, A), Q) - T).pprint()
        print_poly(T.determinant())
        print_poly(cut_off_polynomial(T.determinant(), 1e-7))
        det = poly1d(det_factor)
        for i in range(A.shape[0]):
            det *= A[i, i]
        print_poly(det)
    seed(0)
    B = choice(np.arange(10), 6*3*2, replace=True).reshape(6, 3, 2)
    # B = choice(np.arange(8), 2*2*2, replace=True).reshape(2, 2, 2)
    x = normal(size=10)
    x = np.arange(5)
    B_mat = PolynomialMatrix.coef_array_to_PolynomialMatrix(B)
    B_mat.pprint()
    print(x)
    ret = B_mat.eval(x)
    pprint(ret)

    ret2 = eval_array(B, x)

    P, A, Q, det_factor = B_mat.smith_mcmillan_form()
    (np.dot(np.dot(P, A), Q) - B_mat).pprint()

