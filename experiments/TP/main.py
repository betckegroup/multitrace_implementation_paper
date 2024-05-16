import bempp.api 
import numpy as np
import mtf
from mtf.utils import bhmie
from mtf.config import config
from mtf.functions import define_bempp_functions
from mtf.utils.iterative import gmres
from mtf.preconditioning.osrc import osrc_MtE

from matplotlib import pyplot as plt
from decimal import Decimal
import argparse
import time

bempp.api.enable_console_logging()

parser = argparse.ArgumentParser(
        description="Set the problem parameters"
    )

parser.add_argument(
    "-M", type=int, default=2
)
parser.add_argument(
    "-p", "--precision", type=int, default=10
)
parser.add_argument(
    "--case", type=str, default="B"
)
parser.add_argument(
    "--solver", type=str, default="iterative"
)

parser.add_argument(
    "--noprec",
    action="store_true",
    default=False,
)

args = parser.parse_args()

M = args.M
precision = args.precision
case = args.case
solver = args.solver
noprec = args.noprec

prec = True
assembler = 'default_nonlocal'

if solver == 'direct':
    assembler = 'fmm'
    
if noprec:
    prec = False

mtf.config.set_case(case)

tangential_trace, neumann_trace = define_bempp_functions(config)

k0, k1 = config["k_ext"], config["k_int"]
lambda_par, freq = config["lambda"], config["frequency"]

polarization = config["polarization"]
direction = config["direction"]

eps_rel = config["eps_rel"]
mu_rel = config["mu_rel"]
mu0 = config["mu_ext"]
mu1 = mu_rel * mu0

eta_rel = np.sqrt(mu_rel / eps_rel)

print("The exterior wavenumber is: {0}".format(k0))
print("The interior wavenumber is: {0}".format(k1))
print("----")
print("The exterior wavelenght is: {0}".format(lambda_par))
print("The exterior frequency is: {:.2E}".format(Decimal(freq)))

segments = [[10], [10]]
swapped_normals = [[10], []]

k_int, k_ext = config["k_int"], config["k_ext"]

n = k_int / k_ext
refIndex = n
numAngles = 901
s1, s2, qext, qsca, qback, gsca = bhmie(k_ext, k_int / k_ext, numAngles)
angles = config['angles']

k_list = [k0]
eta_rel_list = [1]
mu_list = [mu0]

for index in range(M-1):
  k_list.append(k1)
  mu_list.append(mu1)
  eta_rel_list.append(eta_rel)

h = 2 * np.pi/(precision*k0)
grid = bempp.api.shapes.sphere(h=h)

print(h, ': h')
print(precision, ': precision')
print(grid.number_of_edges * 2, ': N')

    
dA = [bempp.api.function_space(grid, "RWG", 0, segments=seg, swapped_normals=normals,
                                      include_boundary_dofs=True)
              for seg, normals in zip(segments, swapped_normals)]

p1dA = [bempp.api.function_space(grid, "DP", 1, segments=seg, swapped_normals=normals,
                                      include_boundary_dofs=True)
              for seg, normals in zip(segments, swapped_normals)]

rA = [bempp.api.function_space(grid, "RWG", 0, segments=seg, swapped_normals=normals,
                                      include_boundary_dofs=True)
              for seg, normals in zip(segments, swapped_normals)]
tA = [bempp.api.function_space(grid, "SNC", 0, segments=seg, swapped_normals=normals,
                                      include_boundary_dofs=True)
              for seg, normals in zip(segments, swapped_normals)]

multitrace_ops = []
osrc_ops = []

# > Assemble all diagonal operators
for index in range(M):
  k = k_list[index]
  mu = mu_list[index]
  eta = eta_rel_list[index]
  efie = bempp.api.operators.boundary.maxwell.electric_field(dA[1], rA[1], tA[1], k, assembler=assembler)
  osrc = osrc_MtE(dA[1], rA[1], tA[1], p1dA[1], k)
  mfie = bempp.api.operators.boundary.maxwell.magnetic_field(dA[1], rA[1], tA[1], k, assembler=assembler)
  block_osrc = bempp.api.BlockedOperator(2,2)
  block_osrc[0,1] = eta * osrc
  block_osrc[1,0] = -1/eta * osrc
  osrc_ops.append(block_osrc)
  multitrace_ops.append(bempp.api.GeneralizedBlockedOperator([[mfie, eta * efie],[- 1/eta * efie, mfie]]))
  #osrc_ops.append(bempp.api.GeneralizedBlockedOperator([[zero, eta * osrc],[- 1/eta * osrc, zero]]))

lhs_op = multitrace_ops[0] + multitrace_ops[1]    

prec_op = osrc_ops[0] + osrc_ops[1]    

rhs = [bempp.api.GridFunction(rA[1], dual_space = tA[1], fun=tangential_trace),
      bempp.api.GridFunction(rA[1], dual_space = tA[1], fun=neumann_trace)]

b = bempp.api.assembly.blocked_operator.projections_from_grid_functions_list(rhs, lhs_op.dual_to_range_spaces)
N = b.shape[0]

ta = time.time()
if prec:
    P = prec_op.weak_form()
op_wf = lhs_op.weak_form()
ta = time.time() - ta

ts = time.time()
if solver == 'iterative':
    if prec:
        x, conv, res = gmres(P * op_wf, P * b, return_residuals=True, restart = 1000)
    else:
        x, conv, res = gmres(op_wf, b, return_residuals=True, restart = 1000)

elif solver == 'direct':
    x = np.linalg.solve(A, b)
    res = None

sol = bempp.api.assembly.blocked_operator.grid_function_list_from_coefficients(x.ravel(), lhs_op.domain_spaces)
ts = time.time() - ts

far_field_points = config['far_field_points']
electric_far = bempp.api.operators.far_field.maxwell.electric_field(sol[1].space, far_field_points, k0)
magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(sol[0].space, far_field_points, k0)    
far_field = - electric_far * sol[1] - magnetic_far * sol[0]

A22 = far_field[2,:]
uh = 10 * np.log10(4 * np.pi * np.abs(A22[:1801]))
u =  10 * np.log10(4 * np.pi * np.abs(s1 / (-1j * k_ext) ))
rel_error = np.linalg.norm(uh - u) / np.linalg.norm(u)

import os, os.path
index = len(os.listdir("results/"))
name = "results/" + str(index)

my_dict = {
    "ta" : ta,
    "ts" : ts,
    "precision": precision,
    "case" : case,
    "M": M,
    "N": N,
    "res" : res,
    "solver": solver,
    "noprec" : noprec,
    "rel_error" : rel_error
}
np.save(name + ".npy", my_dict)

print(rel_error)

