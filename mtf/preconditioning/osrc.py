import cmath
import itertools
import time

import bempp.api
import numpy as np
import scipy as sp
from bempp.api import as_matrix
from bempp.api.assembly.boundary_operator import BoundaryOperator
from bempp.api.assembly.discrete_boundary_operator import (
    InverseSparseDiscreteBoundaryOperator,
)


def timerfunc(func):
    def function_timer(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        return value, runtime

    return function_timer


def buildPadeCoefficients(Np, angle):
    a = [
        (2.0 / (2.0 * Np + 1.0)) * np.sin(np.pi * (i + 1.0) / (2.0 * Np + 1.0)) ** 2
        for i in range(Np)
    ]
    b = [np.cos(np.pi * (i + 1.0) / (2.0 * Np + 1.0)) ** 2 for i in range(Np)]
    A = [
        (np.exp(-1.0j * angle / 2.0) * a[i])
        / (1.0 + b[i] * (np.exp(-1.0j * angle) - 1)) ** 2
        for i in range(Np)
    ]
    B = [
        (np.exp(-1.0j * angle) * b[i]) / (1.0 + b[i] * (np.exp(-1.0j * angle) - 1.0))
        for i in range(Np)
    ]
    C_0 = np.exp(1.0j * angle / 2.0) * Sum(a, b, np.exp(-1.0j * angle) - 1.0, 1.0, 1.0)
    R_0 = Sum(A, B, 1.0, 0.0, C_0)
    return A, B, R_0, C_0


def Sum(A, B, inpt, inner_coeff, outer_coeff):
    s = 0.0
    index = 0
    for coefficient in A:
        s += coefficient * inpt / (inner_coeff + B[index] * inpt)
        index += 1
    return outer_coeff + s


def Sum2(A, B, inpt, R_0):
    s = 0.0
    index = 0
    for coefficient in A:
        s += coefficient / (B[index] * (1.0 + B[index] * inpt))
        index += 1
    return R_0 - s


def BranchCutPade(Np, x, angle):
    a = [
        2.0 / (2.0 * Np + 1) * np.sin(np.pi * (i + 1) / (2.0 * Np + 1)) ** 2
        for i in range(Np)
    ]
    b = [np.cos(np.pi * (i + 1) / (2.0 * Np + 1)) ** 2 for i in range(Np)]
    A = [
        np.exp(-1.0j * angle / 2) * a[i] / (1 + b[i] * (np.exp(-1.0j * angle) - 1)) ** 2
        for i in range(Np)
    ]
    B = [
        np.exp(-1.0j * angle) * b[i] / (1 + b[i] * (np.exp(-1.0j * angle) - 1))
        for i in range(Np)
    ]
    C_0 = np.exp(1.0j * angle / 2) * Sum(a, b, np.exp(-1.0j * angle) - 1, 1, 1)
    R_0 = Sum(A, B, 1, 0, C_0)
    return Sum2(A, B, x, R_0)


def InversePade(Np, x, angle):
    a = [
        2.0 / (2.0 * Np + 1) * np.sin(np.pi * (i + 1) / (2.0 * Np + 1)) ** 2
        for i in range(Np)
    ]
    b = [np.cos(np.pi * (i + 1) / (2.0 * Np + 1)) ** 2 for i in range(Np)]
    A = [
        np.exp(-1.0j * angle / 2) * a[i] / (1 + b[i] * (np.exp(-1.0j * angle) - 1)) ** 2
        for i in range(Np)
    ]
    B = [
        np.exp(-1.0j * angle) * b[i] / (1 + b[i] * (np.exp(-1.0j * angle) - 1))
        for i in range(Np)
    ]
    C_0 = np.exp(1.0j * angle / 2) * Sum(a, b, np.exp(-1.0j * angle) - 1, 1, 1)

    gamma = C_0 * B[0] * B[1] + A[0] * B[1] + A[1] * B[0]
    r0 = B[0] * B[1] / gamma
    alpha = C_0 / gamma
    beta = -(C_0 * (B[0] + B[1]) + (A[0] + A[1])) / gamma
    delta = (B[0] + B[1]) / gamma
    omega = (r0 * C_0 - 1) / gamma

    q2 = (beta - np.sqrt(beta**2 - 4 * alpha)) / 2
    q1 = alpha / q2
    r2 = (omega - (delta + beta * r0) * q2) / (q1 - q2)
    r1 = delta + beta * r0 - r2
    Q = [q1, q2]
    R = [r1, r2]
    s = 0
    for i in range(Np):
        s += R[i] / (x - Q[i])
    return r0 + s, (
        x**2 * r0 + x * (r1 + r2 - (q1 + q2) * r0) + r0 * q1 * q2 - r1 * q2 - r2 * q1
    ) / (x**2 - x * (q1 + q2) + q1 * q2)


def scalar_prod(domain, range_, dual_to_range, label="SCALAR_PRODUCT", parameters=None):
    from bempp.api.assembly.functors import (
        scalar_function_value_functor,
        simple_test_trial_integrand_functor,
    )
    from bempp.api.operators.boundary.sparse import operator_from_functors

    return operator_from_functors(
        domain,
        range_,
        dual_to_range,
        scalar_function_value_functor(),
        scalar_function_value_functor(),
        simple_test_trial_integrand_functor(),
        label=label,
        parameters=parameters,
    )


def surface_gradient(
    domain, range_, dual_to_range, label="SURFACE_GRADIENT", parameters=None
):
    from bempp.api.assembly.functors import (
        hcurl_function_value_functor,
        simple_test_trial_integrand_functor,
        surface_gradient_functor,
    )
    from bempp.api.operators.boundary.sparse import operator_from_functors

    return operator_from_functors(
        domain,
        range_,
        dual_to_range,
        surface_gradient_functor(),
        hcurl_function_value_functor(),
        simple_test_trial_integrand_functor(),
        label=label,
        parameters=parameters,
    )


def simple_vector_prod_curl(
    domain, range_, dual_to_range, label="CURL_DP", parameters=None
):
    from bempp.api.assembly.functors import (
        hcurl_function_value_functor,
        simple_test_trial_integrand_functor,
    )
    from bempp.api.operators.boundary.sparse import operator_from_functors

    return operator_from_functors(
        domain,
        range_,
        dual_to_range,
        hcurl_function_value_functor(),
        hcurl_function_value_functor(),
        simple_test_trial_integrand_functor(),
        label=label,
        parameters=parameters,
    )


def surface_curl(domain, range_, dual_to_range, label="SURFACE_CURL", parameters=None):
    from bempp.api.assembly.functors import (
        scalar_surface_curl_functor,
        simple_test_trial_integrand_functor,
    )
    from bempp.api.operators.boundary.sparse import operator_from_functors

    return operator_from_functors(
        domain,
        range_,
        dual_to_range,
        scalar_surface_curl_functor(),
        scalar_surface_curl_functor(),
        simple_test_trial_integrand_functor(),
        label=label,
        parameters=parameters,
    )


def appCoefficients(Ric, DRic, kappa, rho, n):
    coef1 = (cmath.sqrt(1 - (n * (n + 1) / ((kappa * rho) ** 2)))) * Ric
    coef2 = (cmath.sqrt(1 - (n * (n + 1) / ((kappa * rho) ** 2))) ** (-1)) * DRic
    return [coef1, coef2]


def appPadeCoefficients(Np, Ric, DRic, kappa, rho, n):
    angle = np.pi / 2
    coeff = -(n * (n + 1.0)) / (kappa**2)
    s1 = BranchCutPade(Np, coeff, angle)
    s2 = InversePade(Np, coeff, angle)[0]
    coef1 = s1 * Ric
    coef2 = s2 * DRic
    return [coef1, coef2]


class MtEOps:
    def __init__(self, grid, kappa):
        self.grid = grid
        self.kappa = kappa
        self.kappa_eps = kappa + 1.0j * 0.39 * kappa ** (1.0 / 3) * np.sqrt(2) ** (
            2.0 / 3
        )

    def setMtEOperators(self, Np, div_space, curl_space, MtEType, primal):
        if primal:
            self.p1_space = bempp.api.function_space(self.grid, "P", 1)
        else:
            self.p1_space = bempp.api.function_space(self.grid, "B-P", 1)
        self.div_space = div_space
        self.curl_space = curl_space
        self.Np = Np
        vacuum_permittivity = 8.854187817e-12
        vacuum_permeability = 4 * np.pi * 1e-7
        self.Z0 = np.sqrt(vacuum_permeability / vacuum_permittivity)
        A, B, R_0, C_0 = buildPadeCoefficients(self.Np, np.pi / 2)
        self.B = B
        self.R_0 = R_0
        self.alpha = -np.array(A) / (np.array(B))
        self.SA = simple_vector_prod_curl(
            self.curl_space[0], self.curl_space[0], self.curl_space[0]
        )
        self.N = (1.0 / self.kappa_eps) ** 2 * surface_curl(
            self.curl_space[0], self.curl_space[0], self.curl_space[0]
        )
        if MtEType == "1":
            self.K = (self.kappa_eps**2) * scalar_prod(
                self.p1_space, self.p1_space, self.p1_space
            )
            self.L = surface_gradient(self.curl_space[0], self.p1_space, self.p1_space)
        self.Lambda2 = self.SA - self.N
        self.Lambda2luf = InverseSparseDiscreteBoundaryOperator(
            self.Lambda2.weak_form()
        )
        self.shape = self.Lambda2luf._solver.shape
        self.G1 = self.SA.weak_form()

    def applyLambda2(self, v):
        res = -self.Lambda2luf.matvec(v)
        return res

    def invertLambda1(self, rhs):
        rhs = list(
            itertools.chain(
                *[
                    rhs.tolist(),
                    np.zeros(self.P[0]._solver.shape[0] - rhs.shape[0]).tolist(),
                ]
            )
        )
        res = 0
        for i in range(self.Np):
            res += (
                self.alpha[i]
                * self.P[i].matvec(rhs)[0 : self.curl_space[0].global_dof_count]
            )
        return res

    def MtE1Prec(self, v):
        v = np.ravel(v)  # ravel
        res = self.R_0 * v + self.G1 * self.invertLambda1(v)
        return -self.applyLambda2(res)

    def MtE2Prec(self, v):
        return self.applyLambda2(v)

    def MtE2Prec3(self, v):
        return self.applyLambda2(v)

    @timerfunc
    def getMtEPrec(self, MtEType, Np, div_space, curl_space, primal=True):
        self.setMtEOperators(Np, div_space, curl_space, MtEType, primal)
        if MtEType == "1":
            self.P = []
            for i in range(self.Np):
                self.P.append(
                    InverseSparseDiscreteBoundaryOperator(
                        sp.sparse.bmat(
                            [
                                [
                                    as_matrix(
                                        (self.SA - self.B[i] * self.N).weak_form()
                                    ),
                                    as_matrix(
                                        (
                                            self.B[i]
                                            * self.L.transpose(self.curl_space[0])
                                        ).weak_form()
                                    ),
                                ],
                                [
                                    as_matrix(self.L.weak_form()),
                                    as_matrix(self.K.weak_form()),
                                ],
                            ],
                            "csc",
                        )
                    )
                )
            MtEPrec = self.MtE1Prec
        elif MtEType == "2":
            MtEPrec = self.MtE2Prec
        else:
            self.coef = 0
            for j in range(int(np.floor(4.0 * self.Np / 5)), self.Np):
                self.coef += self.alpha[j]
            MtEPrec = self.MtE2Prec
        return sp.sparse.linalg.LinearOperator(self.shape, matvec=MtEPrec)


class OSRC(BoundaryOperator):
    def __init__(self, Prec, mteOp):
        self._domain = Prec.div_space[0]
        self._range = Prec.div_space[0]
        self._dual_to_range = Prec.curl_space[0]
        self._mteOp = mteOp
        self._Prec = Prec
        self._dtype = mteOp.dtype
        self._label = ""

    def weak_form(self):
        return self._mteOp


class _osrcMtE(BoundaryOperator):
    def __init__(self, Prec, mteOp):
        # Prec, mteOp):
        self._domain = Prec.div_space[0]
        self._range = Prec.div_space[0]
        self._dual_to_range = Prec.curl_space[0]
        self._mteOp = mteOp
        self._Prec = Prec
        self._dtype = mteOp.dtype
        self._label = ""

    def weak_form(self):
        return self._mteOp


def osrc_MtE(grid, wave_number, MtEType="1", Np=1):
    Prec = MtEOps(grid, wave_number)
    RWG = bempp.api.function_space(grid, "RWG", 0)
    SNC = bempp.api.function_space(grid, "SNC", 0)
    div_space = [RWG, RWG]
    curl_space = [SNC, SNC]
    mteOp, t = Prec.getMtEPrec(
        MtEType, Np, div_space, curl_space
    )  # es para MtE del tipo 1, con Np=1 en la expansion de Pade %%
    my_osrc = _osrcMtE(Prec, mteOp)
    return my_osrc
