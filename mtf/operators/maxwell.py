import bempp.api
import numpy as np
from bempp.api.assembly.blocked_operator import (
    BlockedOperator,
    coefficients_of_grid_function_list,
    grid_function_list_from_coefficients,
    projections_of_grid_function_list,
)
from bempp.api.operators.boundary.maxwell import electric_field, magnetic_field
from bempp.api.operators.boundary.sparse import identity

from bemppUQ.preconditioning.osrc import osrc_MtE

from ..foa.operators import (
    function_product,
    surface_divergence,
    surface_gradient,
    trace_transformation,
)
from ..functions import neumann_trace, tangential_trace
from ..utils.login import gmres, rescale


def multitrace_operator(
    grid, wave_number, parameters=None, target=None, spaces="maxwell_primal"
):
    """Assemble the multitrace operator for Maxwell."""
    if spaces == "maxwell_primal":
        blocked_operator = BlockedOperator(2, 2)

        rwg_space = bempp.api.function_space(grid, "RWG", 0)
        snc_space = bempp.api.function_space(grid, "SNC", 0)

        efie = electric_field(
            rwg_space, rwg_space, snc_space, wave_number, parameters=parameters
        )
        mfie = magnetic_field(
            rwg_space, rwg_space, snc_space, wave_number, parameters=parameters
        )

        blocked_operator[0, 0] = mfie
        blocked_operator[0, 1] = efie
        blocked_operator[1, 0] = -1 * efie
        blocked_operator[1, 1] = mfie
    else:
        blocked_operator = bempp.api.operators.boundary.maxwell.multitrace_operator(
            grid, wave_number, parameters=parameters, target=target
        )
    return blocked_operator


def multitrace_identity(grid, parameters=None, spaces="maxwell_primal"):
    """Return the multitrace identity operator.
    Parameters
    ----------
    grid : bempp.api.grid.Grid
        The underlying grid for the multitrace operator
    parameters : bempp.api.common.ParameterList
        Parameters for the operator. If none given
        the default global parameter object
        `bempp.api.global_parameters` is used.
    spaces: string
        For the proper dual spaces in
        Maxwell problems choose 'maxwell_primal'.
    """
    if spaces == "maxwell_primal":
        blocked_operator = BlockedOperator(2, 2)
        rwg_space = bempp.api.function_space(grid, "RWG", 0)
        snc_space = bempp.api.function_space(grid, "SNC", 0)

        blocked_operator[0, 0] = identity(
            rwg_space, rwg_space, snc_space, parameters=parameters
        )
        blocked_operator[1, 1] = identity(
            rwg_space, rwg_space, snc_space, parameters=parameters
        )
    else:
        blocked_operator = bempp.api.operators.boundary.sparse.multitrace_identity(
            grid, parameters=parameters, spaces=spaces
        )
    return blocked_operator


def multitrace_osrc(grid, wave_number, parameters=None, target=None, MtEType="1", Np=1):
    """Assemble the multitrace osrc operator for Maxwell."""

    blocked_operator = BlockedOperator(2, 2)
    osrc = osrc_MtE(grid, wave_number, MtEType=MtEType, Np=Np)

    blocked_operator[0, 1] = osrc
    blocked_operator[1, 0] = -1 * osrc
    return blocked_operator


def assemble_operators(grid, config):
    # Assemble operators for transmission problem.
    k_ext, k_int = config["k_ext"], config["k_int"]
    osrc = config["osrc"]
    spaces = config["spaces"]
    far_field_points = config["far_field_points"]
    if osrc and spaces != "maxwell_primal":
        raise NotImplementedError("Osrc only for primal formulations")

    multitrace_int = multitrace_operator(grid, k_int, spaces=spaces)
    multitrace_ext = multitrace_operator(grid, k_ext, spaces=spaces)
    identity = multitrace_identity(grid, spaces=spaces)
    electric_far = bempp.api.operators.far_field.maxwell.electric_field(
        multitrace_ext.domain_spaces[1], far_field_points, k_ext
    )
    magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(
        multitrace_ext.domain_spaces[0], far_field_points, k_ext
    )

    if osrc and spaces == "maxwell_primal":
        osrc_int = multitrace_osrc(grid, k_int)
        osrc_ext = multitrace_osrc(grid, k_ext)
        transmission_operators = (
            multitrace_int,
            multitrace_ext,
            osrc_int,
            osrc_ext,
            identity,
            electric_far,
            magnetic_far,
        )
    else:
        transmission_operators = (
            multitrace_int,
            multitrace_ext,
            identity,
            electric_far,
            magnetic_far,
        )

    return transmission_operators


def evaluate_far_field(transmission_operators, config):
    # Solve the penetrable scattering problem and evaluate far-field.
    eps_rel, mu_rel = config["eps_rel"], config["mu_rel"]
    k_ext = config["k_ext"]
    polarization, direction = config["polarization"], config["direction"]
    osrc = config["osrc"]
    spaces = config["spaces"]
    if osrc:
        (
            multitrace_int,
            multitrace_ext,
            osrc_int,
            osrc_ext,
            identity,
            electric_far,
            magnetic_far,
        ) = transmission_operators
        rescaled_int_osrc = rescale(osrc_int, np.sqrt(eps_rel), np.sqrt(mu_rel))
        op_osrc = rescaled_int_osrc + osrc_ext

    else:
        (
            multitrace_int,
            multitrace_ext,
            identity,
            electric_far,
            magnetic_far,
        ) = transmission_operators

    rescaled_int_op = rescale(multitrace_int, np.sqrt(eps_rel), np.sqrt(mu_rel))
    lhs_op = rescaled_int_op + multitrace_ext
    rhs_op = 0.5 * identity - rescaled_int_op

    if spaces == "maxwell_primal":
        dual = lhs_op.domain_spaces
    else:
        dual = lhs_op.dual_to_range_spaces
    electric_incident = bempp.api.GridFunction(
        lhs_op.domain_spaces[0],
        fun=tangential_trace(k_ext, polarization, direction),
        dual_space=dual[0],
    )
    magnetic_incident = bempp.api.GridFunction(
        lhs_op.domain_spaces[1],
        fun=neumann_trace(k_ext, polarization, direction),
        dual_space=dual[1],
    )

    rhs = rhs_op * [electric_incident, magnetic_incident]
    if spaces != "maxwell_primal":
        op_wf = (lhs_op * lhs_op).strong_form()
        b = coefficients_of_grid_function_list(lhs_op * rhs)
        x, info, res, times = gmres(op_wf, b, return_residuals=True)
        solution = grid_function_list_from_coefficients(x, lhs_op.domain_spaces)
    else:
        b = projections_of_grid_function_list(rhs, dual)
        op_osrc_wf = op_osrc.weak_form()
        op_wf = lhs_op.weak_form()
        x, info, res, times = gmres(
            op_osrc_wf * op_wf, op_osrc_wf * b, return_residuals=True
        )
        solution = grid_function_list_from_coefficients(x, lhs_op.domain_spaces)

    far_field = -electric_far * solution[1] - magnetic_far * solution[0]
    return far_field, solution


def evaluate_far_field_sd(
    base_grid,
    transmission_operators,
    config,
    solution,
    grid_funs,
    solve=True,
    density=False,
):
    result = []
    eps_rel, mu_rel = config["eps_rel"], config["mu_rel"]
    k_int, k_ext = config["k_int"], config["k_ext"]
    polarization, direction = config["polarization"], config["direction"]

    (
        multitrace_int,
        multitrace_ext,
        identity,
        electric_far,
        magnetic_far,
    ) = transmission_operators
    rescaled_int_op = rescale(multitrace_int, np.sqrt(eps_rel), np.sqrt(mu_rel))

    lhs_op = rescaled_int_op + multitrace_ext
    rhs_op = 0.5 * identity - rescaled_int_op

    electric_space, magnetic_space = transmission_operators[0].domain_spaces
    electric_dual, magnetic_dual = transmission_operators[0].dual_to_range_spaces

    scalar_space = bempp.api.function_space(base_grid, "B-P", 1)
    electric_incident = bempp.api.GridFunction(
        electric_space,
        fun=tangential_trace(k_ext, polarization, direction),
        dual_space=electric_dual,
    )
    magnetic_incident = bempp.api.GridFunction(
        magnetic_space,
        fun=neumann_trace(k_ext, polarization, direction),
        dual_space=magnetic_dual,
    )

    surface_divergence_electric = surface_divergence(
        electric_space, scalar_space, scalar_space
    )
    surface_divergence_magnetic = surface_divergence(
        magnetic_space, scalar_space, scalar_space
    )

    trace_transformation_electric = trace_transformation(
        electric_space, electric_space, electric_dual
    )
    trace_transformation_magnetic = trace_transformation(
        magnetic_space, magnetic_space, magnetic_dual
    )

    total_trace_electric_ext = solution[0] + electric_incident
    total_trace_magnetic_ext = solution[1] + magnetic_incident

    total_trace_electric_int = np.sqrt(eps_rel) * total_trace_electric_ext
    total_trace_magnetic_int = np.sqrt(mu_rel) * total_trace_magnetic_ext

    e_nu_ext = (
        -1.0 / (1j * k_ext) * (surface_divergence_magnetic * total_trace_magnetic_ext)
    )
    e_nu_int = (
        -1.0 / (1j * k_int) * (surface_divergence_magnetic * total_trace_magnetic_int)
    )
    h_nu_ext = (
        1.0 / (1j * k_ext) * (surface_divergence_electric * total_trace_electric_ext)
    )
    h_nu_int = (
        1.0 / (1j * k_int) * (surface_divergence_electric * total_trace_electric_int)
    )

    tangential_trace_electric_ext = (
        -trace_transformation_electric * total_trace_electric_ext
    )
    tangential_trace_electric_int = (
        -trace_transformation_electric * total_trace_electric_int
    )

    tangential_trace_magnetic_ext = (
        -trace_transformation_magnetic * total_trace_magnetic_ext
    )
    tangential_trace_magnetic_int = (
        -trace_transformation_magnetic * total_trace_magnetic_int
    )

    gradient_operator_electric = surface_gradient(
        scalar_space, electric_space, electric_space
    )
    gradient_operator_magnetic = surface_gradient(
        scalar_space, magnetic_space, magnetic_space
    )
    if not isinstance(grid_funs, list):
        grid_funs = [grid_funs]
    i = 0
    for grid_fun in grid_funs:
        print(i)
        perturb_tangential_electric_ext = function_product(
            grid_fun, tangential_trace_electric_ext, magnetic_space, magnetic_dual
        )
        perturb_tangential_electric_int = function_product(
            grid_fun, tangential_trace_electric_int, magnetic_space, magnetic_dual
        )
        perturb_tangential_magnetic_ext = function_product(
            grid_fun, tangential_trace_magnetic_ext, electric_space, electric_dual
        )
        perturb_tangential_magnetic_int = function_product(
            grid_fun, tangential_trace_magnetic_int, electric_space, electric_dual
        )

        perturb_enu_ext = function_product(
            grid_fun, e_nu_ext, scalar_space, scalar_space
        )
        perturb_hnu_ext = function_product(
            grid_fun, h_nu_ext, scalar_space, scalar_space
        )
        perturb_enu_int = function_product(
            grid_fun, e_nu_int, scalar_space, scalar_space
        )
        perturb_hnu_int = function_product(
            grid_fun, h_nu_int, scalar_space, scalar_space
        )

        trace_grad_perturb_enu_ext = (
            trace_transformation_electric * gradient_operator_electric * perturb_enu_ext
        )
        trace_grad_perturb_enu_int = (
            trace_transformation_electric * gradient_operator_electric * perturb_enu_int
        )
        trace_grad_perturb_hnu_ext = (
            trace_transformation_magnetic * gradient_operator_magnetic * perturb_hnu_ext
        )
        trace_grad_perturb_hnu_int = (
            trace_transformation_magnetic * gradient_operator_magnetic * perturb_hnu_int
        )

        f_one_plus = (
            trace_grad_perturb_enu_ext - 1j * k_ext * perturb_tangential_magnetic_ext
        )
        f_two_plus = (
            trace_grad_perturb_hnu_ext + 1j * k_ext * perturb_tangential_electric_ext
        )

        f_one_minus = (
            trace_grad_perturb_enu_int - 1j * k_int * perturb_tangential_magnetic_int
        )
        f_two_minus = (
            trace_grad_perturb_hnu_int + 1j * k_int * perturb_tangential_electric_int
        )

        rhs_plus = rhs_op * [f_one_plus, f_two_plus]
        rhs_minus = (multitrace_int - 0.5 * identity) * [f_one_minus, f_two_minus]
        rhs_minus[0] *= 1.0 / np.sqrt(eps_rel)
        rhs_minus[1] *= 1.0 / np.sqrt(mu_rel)
        rhs = [rhs_plus[0] + rhs_minus[0], rhs_plus[1] + rhs_minus[1]]
        if solve:
            op_wf = (lhs_op * lhs_op).strong_form()
            b = coefficients_of_grid_function_list(lhs_op * rhs)
            x, info, res, times = gmres(op_wf, b, return_residuals=True)
            sol_p = grid_function_list_from_coefficients(x, lhs_op.domain_spaces)
            far_field_p = -electric_far * sol_p[1] - magnetic_far * sol_p[0]
            if density:
                result.append([sol_p, far_field_p])
            else:
                result.append(far_field_p)
        else:
            result.append(rhs)
        i += 1
    if solve:
        return result
    else:
        return result, lhs_op
