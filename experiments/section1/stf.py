import bempp.api
import numpy as np
import mtf
from mtf.utils import bhmie
from mtf.operators.maxwell import assemble_stf
from mtf.utils.iterative import gmres
import argparse

parser = argparse.ArgumentParser(description="Set parameters")
parser.add_argument("--M", default=3, type=int)
parser.add_argument("--solver", default='gmres', type=str)
parser.add_argument("--precision", default=10, type=int)
parser.add_argument("--case", default="B", type=str)
args = parser.parse_args()

bempp.api.enable_console_logging()

M = args.M
case = args.case
precision = args.precision
solver = args.solver

print('Solver is: ', solver)
config = mtf.config.set_case(case, True)
k0 = config["k_ext"]
k1 = config["k_int"]

numAngles = 901
s1, s2, qext, qsca, qback, gsca = bhmie(
    config["k_ext"], config["k_int"] / config["k_ext"], numAngles
)
# to do: understand numAngles

# Outside: k_ext
# Inside: k_int

k_list = [k0]
eta_rel_list = [1]
mu_list = [config["mu_ext"]]

for index in range(M - 1):
    k_list.append(k1)
    mu_list.append(config["mu_int"])
    eta_rel_list.append(config["eta_rel"])

print("k_ext: {:.2f}".format(config["k_ext"]))
print("k_int: {:.2f}".format(config["k_int"]))

print("----")
print("lambda: {:.2f}".format(config["lambda"]))
print("f: {:.2E}".format(config["frequency"]))

if M == 2:
    # multiple shape, M=2
    segments = [[10, 20], [10, 20]]
    swapped_normals = [[10, 20], []]

elif M == 3:
    # multiple shape, M=3
    segments = [[10, 20], [10, 12], [12, 20]]
    swapped_normals = [[10, 20], [], [12]]

h = 2 * np.pi / (precision * k0)
grid = bempp.api.shapes.multitrace_sphere(h=h)

params = {}
params["M"] = M
params["k_list"] = k_list
params["mu_list"] = mu_list
params["eta_rel_list"] = eta_rel_list
params["segments"] = segments
params["swapped_normals"] = swapped_normals

print(h, ": h")
print(precision, ": precision")
print(grid.number_of_edges * 2, ": N")

if solver == 'gmres':
    lhs_prec, lhs_op, rhs = assemble_stf(grid, params, config, solver=solver)
    P_wf = lhs_prec.weak_form()
else:
    lhs_op, rhs = assemble_stf(grid, params, config, solver=solver)

# Solver
op_wf = lhs_op.weak_form()

b = bempp.api.assembly.blocked_operator.projections_from_grid_functions_list(
    rhs, lhs_op.dual_to_range_spaces
)
N = b.shape[0]

if solver == 'gmres':
    print('starting gmres')
    x, conv_gmres, res_gmres = gmres(P_wf * op_wf, P_wf * b, return_residuals=True, restart = 100, maxiter=100)
else:
    A = bempp.api.as_matrix(op_wf)
    x = np.linalg.solve(A, b)

sol = bempp.api.assembly.blocked_operator.grid_function_list_from_coefficients(
    x.ravel(), lhs_op.domain_spaces
)

far_field_points = config["far_field_points"]
electric_far = bempp.api.operators.far_field.maxwell.electric_field(
    sol[1].space, far_field_points, k0
)
magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(
    sol[0].space, far_field_points, k0
)

far_field = -electric_far * sol[1] - magnetic_far * sol[0]

A22 = far_field[2, :]
uh = 10 * np.log10(4 * np.pi * np.abs(A22[:1801]))
u = 10 * np.log10(4 * np.pi * np.abs(s1 / (-1j * k0)))
rel_error = np.linalg.norm(uh - u) / np.linalg.norm(u)

if M == 2:
    dirichlet_jump = -1
    neumann_jump = -1
print(rel_error)

results = np.array([M, precision, h, N, dirichlet_jump, neumann_jump, rel_error])

np.save("results/" + str(M) + case + str(precision) + "stf" + solver, results)