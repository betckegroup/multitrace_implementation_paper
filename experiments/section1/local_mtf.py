import bempp.api 
import numpy as np
import mtf

from mtf.utils import bhmie
from mtf.config import config
from bempp.api.operators.boundary.sparse import identity
from mtf.operators.maxwell import assemble_mtf
from mtf.utils.iterative import gmres

from mtf.functions import define_bempp_functions
from bempp.api.assembly.blocked_operator import BlockedOperator
from mtf.assembly.operators import GeneralizedBlockedOperator

from matplotlib import pyplot as plt
bempp.api.enable_console_logging()
import argparse

parser = argparse.ArgumentParser(description="Set parameters")

parser.add_argument("--M", default=3, type=int)
parser.add_argument("--solver", default='gmres', type=str)
parser.add_argument("--precision", default=10, type=int)
parser.add_argument("--case", default="B", type=str)
args = parser.parse_args()

M = args.M
solver = args.solver
case = args.case
precision = args.precision

print('Solver is: ', solver)
config = mtf.config.set_case(case, True)
k0 = config['k_ext']
k1 = config['k_int']

numAngles = 901
s1, s2, qext, qsca, qback, gsca = bhmie(config['k_ext'], config['k_int'] / config['k_ext'], numAngles)
# to do: understand numAngles

if M == 2:
    #multiple shape, M=2
    segments = [[10, 20], [10, 20]]
    swapped_normals = [[10, 20], []]
elif M == 3:
    #multiple shape, M=3
    segments = [[10, 20], [10, 12], [12, 20]]
    swapped_normals = [[10, 20], [], [12]]


# Outside: k_ext
# Inside: k_int

k_list = [k0]
eta_rel_list = [1]
mu_list = [config['mu_ext']]


for index in range(M-1):
  k_list.append(k1)
  mu_list.append(config['mu_int'])
  eta_rel_list.append(config['eta_rel'])

print("k_ext: {:.2f}".format(config['k_ext']))
print("k_int: {:.2f}".format(config['k_int']))

print("----")
print("lambda: {:.2f}".format(config['lambda']))
print("f: {:.2E}".format(config['frequency']))

h = 2 * np.pi/(precision*k0)
grid = bempp.api.shapes.multitrace_sphere(h=h)

params = {}
params["M"] = M
params["k_list"] = k_list
params["mu_list"] = mu_list
params["eta_rel_list"] = eta_rel_list
params["segments"] = segments
params["swapped_normals"] = swapped_normals


if solver == 'gmres':
    lhs_prec, lhs_op, rhs = assemble_mtf(grid, params, config, solver=solver)
    P_wf = lhs_prec.weak_form()
else:
    lhs_op, rhs = assemble_mtf(grid, params, config, solver=solver)

# Solver
op_wf = lhs_op.weak_form()

b = bempp.api.assembly.blocked_operator.projections_from_grid_functions_list(
    rhs, lhs_op.dual_to_range_spaces
)
N = b.shape[0]
print('Ndofs: ', N)
if solver == 'gmres':
    print('go solver')
    x, conv_gmres, res_gmres = gmres(P_wf * op_wf, P_wf * b, return_residuals=True, restart = 2000, maxiter=2000)
else:
    A = bempp.api.as_matrix(op_wf)
    x = np.linalg.solve(A, b)

sol = bempp.api.assembly.blocked_operator.grid_function_list_from_coefficients(
    x.ravel(), lhs_op.domain_spaces
)


far_field_points = config['far_field_points']
electric_far = bempp.api.operators.far_field.maxwell.electric_field(sol[1].space, far_field_points, k0)
magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(sol[0].space, far_field_points, k0)    

far_field =  electric_far * sol[1] + magnetic_far * sol[0]

A22 = far_field[2,:]
uh = 10 * np.log10(4 * np.pi * np.abs(A22[:1801]))
u =  10 * np.log10(4 * np.pi * np.abs(s1 / (-1j * k0) ))
rel_error = np.linalg.norm(uh - u) / np.linalg.norm(u)

if M == 3:
    interface = [12]
    interface_space = bempp.api.function_space(grid, "RWG", 0, segments=interface, include_boundary_dofs = True)

    map_dom0_to_interface = identity(sol[2].space, interface_space, interface_space)
    map_dom1_to_interface = identity(sol[4].space, interface_space, interface_space)
    
    trace0i = map_dom0_to_interface @ sol[2]
    trace1i = map_dom1_to_interface @ sol[4]
    
    normal_trace0i = map_dom0_to_interface @ sol[3]
    normal_trace1i = map_dom1_to_interface @ sol[5]
    
    dirichlet_jump = (trace0i + trace1i).l2_norm() / (trace0i.l2_norm())
    neumann_jump = (normal_trace0i + normal_trace1i).l2_norm() / (normal_trace0i.l2_norm())
    
    print(dirichlet_jump,': L^2 relative error for tangential traces')
    print(neumann_jump,': L^2 relative error for magnetic traces')

    interface_space_p1 = bempp.api.function_space(grid, "P", 1)

    c_p1 = np.linalg.norm(trace0i.evaluate_on_vertices(),axis=0)
    
    u_p1 = bempp.api.GridFunction(interface_space_p1, coefficients = c_p1)
elif M ==2: 
    dirichlet_jump = -1
    neumann_jump = -1
print(rel_error)

results = np.array([M, precision, h, N, dirichlet_jump, neumann_jump, rel_error])
name = 'results/' + str(M) + case + str(precision) + 'mtf' + solver

print(name)
np.save(name, results)