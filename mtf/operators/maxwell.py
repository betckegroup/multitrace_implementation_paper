import bempp.api
from mtf.assembly.operators import GeneralizedBlockedOperator
from bempp.api.assembly.blocked_operator import BlockedOperator
from mtf.functions import define_bempp_functions
from mtf.preconditioning.osrc import osrc_MtE
import numpy as np
from bempp.api.operators.boundary.sparse import identity
from bempp.api import ZeroBoundaryOperator

def assemble_stf(grid, params, config, solver='gmres'):

    tangential_trace, neumann_trace = define_bempp_functions(config)
    
    M = params['M']
    k_list = params['k_list']
    mu_list = params['mu_list']
    eta_rel_list = params['eta_rel_list']
    segments = params['segments']
    swapped_normals = params['swapped_normals']
    
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
    if solver == 'gmres':
        osrc_ops = []
    # > Assemble all diagonal operators
    for index in range(M):
      k = k_list[index]
      mu = mu_list[index]
      eta = eta_rel_list[index]
      efie = bempp.api.operators.boundary.maxwell.electric_field(dA[1], rA[1], tA[1], k)
      mfie = bempp.api.operators.boundary.maxwell.magnetic_field(dA[1], rA[1], tA[1], k)
      multitrace_ops.append(GeneralizedBlockedOperator([[mfie, eta * efie],[- 1/eta * efie, mfie]]))
      if solver == 'gmres':
          osrc = osrc_MtE(dA[1], dA[1], tA[1], p1dA[1], k)
          zero = (1+1j) * bempp.api.ZeroBoundaryOperator(dA[1], dA[1], tA[1])
          osrc_ops.append(GeneralizedBlockedOperator([[zero, eta * osrc],[- 1/eta * osrc, zero]]))

    
    # Define the final operator
    
    lhs_op = multitrace_ops[0] + multitrace_ops[1]  
    if solver == 'gmres':
        lhs_prec = osrc_ops[0] + osrc_ops[1]  
    
    
    
    rhs = [bempp.api.GridFunction(rA[1], dual_space = tA[1], fun=tangential_trace),
          bempp.api.GridFunction(rA[1], dual_space = tA[1], fun=neumann_trace)]
    if solver == 'gmres':
        return lhs_prec, lhs_op, rhs
    else:
        return lhs_op, rhs

def assemble_mtf(grid, params, config, solver='gmres'):

    tangential_trace, neumann_trace = define_bempp_functions(config)
    
    M = params['M']
    k_list = params['k_list']
    mu_list = params['mu_list']
    eta_rel_list = params['eta_rel_list']
    segments = params['segments']
    swapped_normals = params['swapped_normals']

    
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

    if solver == 'gmres':
        osrc_ops = []
    multitrace_ops = []
    # > Assemble all diagonal operators
    for index in range(M):
        k = k_list[index]
        mu = mu_list[index]
        eta = eta_rel_list[index]
        efie = bempp.api.operators.boundary.maxwell.electric_field(dA[index], rA[index], tA[index], k)
        mfie = bempp.api.operators.boundary.maxwell.magnetic_field(dA[index], rA[index], tA[index], k)
        multitrace_ops.append(GeneralizedBlockedOperator([[mfie, eta * efie],[- 1/eta * efie, mfie]]))
        if solver == 'gmres':
            osrc = osrc_MtE(dA[index], dA[index], tA[index], p1dA[index], k)
            zero = (1+1j) * bempp.api.ZeroBoundaryOperator(dA[index], dA[index], tA[index])
            osrc_ops.append(GeneralizedBlockedOperator([[zero, eta * osrc],[- 1/eta * osrc, zero]]))

    
        # Define the final operator
        
    block_system = [M * [None] for _ in range(M)]
    
    for i in range(M):
      for j in range(M):
        if i == j:
          block_system[i][j] = 2 * multitrace_ops[i]
        else:
          all = segments[i] + segments[j]
          non_disjoint = np.unique(all).shape[0] != len(all)
          
          if non_disjoint:
            ident = identity(dA[j], rA[i], tA[i])
            op = BlockedOperator(2, 2)
            #op[0, 0] = -ident
            op[0, 0] = ident
            op[1, 1] = ident
            op.weak_form()
            #op[1, 1] = ident
            block_system[i][j] = op
          else:
            op = BlockedOperator(2, 2)
            zero = ZeroBoundaryOperator(dA[j], rA[i], tA[i])
            op[0, 0] = zero
            op[1, 1] = zero
            block_system[i][j] = op

    if solver == 'gmres':
        # Define the final operator
        block_osrc = [M * [None] for _ in range(M)]
        for i in range(M):
          for j in range(M):
            if i == j:
              block_osrc[i][j] = 2 * osrc_ops[i]
            else:
              all = segments[i] + segments[j]
              non_disjoint = np.unique(all).shape[0] != len(all)
              
              if non_disjoint:
                ident = identity(dA[j], rA[i], tA[i])
                op = BlockedOperator(2, 2)
                #op[0, 0] = -ident
                op[0, 0] = ident
                op[1, 1] = ident
                op.weak_form()
                #op[1, 1] = ident
                block_osrc[i][j] = op
              else:
                op = BlockedOperator(2, 2)
                zero = ZeroBoundaryOperator(dA[j], rA[i], tA[i])
                op[0, 0] = zero
                op[1, 1] = zero
                block_osrc[i][j] = op

        P_op = GeneralizedBlockedOperator(block_osrc)
        

    lhs_op = GeneralizedBlockedOperator(block_system)
    
    rhs = [2 * bempp.api.GridFunction(rA[0], dual_space = tA[0], fun=tangential_trace),
          2 * bempp.api.GridFunction(rA[0], dual_space = tA[0], fun=neumann_trace)]
    
    for i in range(1, M):
        zero_func = [bempp.api.GridFunction.from_zeros(dA[i]),bempp.api.GridFunction.from_zeros(dA[i])]
        rhs = rhs + zero_func
    
    if solver == 'gmres':
        return P_op, lhs_op, rhs
    else:
        return lhs_op, rhs

        