import time

import bempp.api
import numpy as np
import scipy.sparse.linalg
from bempp.api.assembly.boundary_operator import BoundaryOperator
from bempp.api.assembly.grid_function import GridFunction


def tensorize(x1, x2=None):
    if x2 is None:
        x2 = x1

    x1 = np.array([x1])
    x2 = np.array([x2])
    return np.dot(x1.T, x2.conj())


def rescale(A, d1, d2):
    # Rescale the 2x2 block operator matsrix A
    B = bempp.api.BlockedOperator(2, 2)
    for i in range(2):
        for j in range(2):
            B[i, j] = A[i, j]
    B[0, 1] = B[0, 1] * (d2 / d1)
    B[1, 0] = B[1, 0] * (d1 / d2)

    return B


def get_h(grid):
    # routine that gives the exact value of h
    # defined as the maximum length of the triangles
    # of a given mesh
    elements1 = list(grid.leaf_view.entity_iterator(1))
    vol = []
    for el1 in elements1:
        vol.append(el1.geometry.volume)
    vol = np.array(vol)
    return [vol.min(), vol.max(), vol.mean()]


def relative_error(self, fun, element=None):
    """Compute the relative L^2 error compared to a given analytic function."""
    import numpy as np
    from bempp.api.integration import gauss_triangle_points_and_weights

    p_normal = np.array([[0]])
    global_diff = 0
    fun_l2_norm = 0
    accuracy_order = self.parameters.quadrature.far.single_order
    points, weights = gauss_triangle_points_and_weights(accuracy_order)
    npoints = points.shape[1]
    element_list = (
        [element]
        if element is not None
        else list(self.grid.leaf_view.entity_iterator(0))
    )
    for element in element_list:
        integration_elements = element.geometry.integration_elements(points)
        normal = element.geometry.normals(p_normal).reshape(3)
        global_dofs = element.geometry.local2global(points)
        fun_vals = np.zeros((self.component_count, npoints), dtype=self.dtype)
        for j in range(npoints):
            fun_vals[:, j] = fun(global_dofs[:, j], normal)
        diff = np.sum(np.abs(self.evaluate(element, points) - fun_vals) ** 2, axis=0)
        global_diff += np.sum(diff * integration_elements * weights)
        abs_fun_squared = np.sum(np.abs(fun_vals) ** 2, axis=0)
        fun_l2_norm += np.sum(abs_fun_squared * integration_elements * weights)
    return np.sqrt(global_diff / fun_l2_norm)


class _it_counter(object):
    def __init__(self, store_residuals):
        self._count = 0
        self._store_residuals = store_residuals
        self._residuals = []
        self._times = []
        self._ta = time.time()

    def __call__(self, x):
        self._count += 1
        if self._store_residuals:
            self._residuals.append(np.linalg.norm(x))
            current_ta = time.time()
            self._times.append(current_ta - self._ta)
            self._ta = current_ta
            print(
                "iteration -",
                self._count,
                "|| residual -",
                np.linalg.norm(x),
                self._times[-1],
            )

    @property
    def count(self):
        return self._count

    @property
    def residuals(self):
        return self._residuals

    @property
    def times(self):
        return self._times


def gmres(
    A,
    b,
    tol=1e-5,
    restart=None,
    maxiter=None,
    use_strong_form=False,
    return_residuals=False,
):
    """Interface to the scipy.sparse.linalg.gmres function.

    This function behaves like the scipy.sparse.linalg.gmres function. But
    instead of a linear operator and a vector b it can take a boundary operator
    and a grid function. In this case, the result is returned as a grid function in the
    correct space.

    """
    if not isinstance(A, BoundaryOperator) and use_strong_form:
        raise ValueError("Strong form only with BoundaryOperator")
    if isinstance(A, BoundaryOperator) and not isinstance(b, GridFunction):
        raise ValueError("Instance Error")

    # Assemble weak form before the logging messages

    if isinstance(A, BoundaryOperator) and isinstance(b, GridFunction):
        if use_strong_form:
            A_op = A.strong_form()
            b_vec = b.coefficients
        else:
            A_op = A.weak_form()
            b_vec = b.projections(A.dual_to_range)
    else:
        A_op = A
        b_vec = b

    callback = _it_counter(return_residuals)

    x, info = scipy.sparse.linalg.gmres(
        A_op, b_vec, tol=tol, restart=restart, maxiter=maxiter, callback=callback
    )

    if isinstance(A, BoundaryOperator) and isinstance(b, GridFunction):
        res_fun = GridFunction(A.domain, coefficients=x.ravel())

    else:
        res_fun = x
    if return_residuals:
        return res_fun, info, callback.residuals, callback.times
    else:
        return res_fun, info
