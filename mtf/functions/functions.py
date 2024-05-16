import numpy as np
import bempp.api

def define_bempp_functions(config):
    direction = config['direction']
    polarization = config['polarization']
    k0 = config['k_ext']
    
    def plane_wave(point):
        return polarization * np.exp(1j * k0 * np.dot(point, direction))

    
    @bempp.api.complex_callable
    def tangential_trace(point, n, domain_index, result):
        value = polarization * np.exp(1j * k0 * np.dot(point, direction))
        result[:] = np.cross(value, n)


    @bempp.api.complex_callable
    def neumann_trace(point, n, domain_index, result):
        value = (
            np.cross(direction, polarization)
            * 1j
            * k0
            * np.exp(1j * k0 * np.dot(point, direction))
        )
        result[:] = 1.0 / (1j * k0) * np.cross(value, n)
    return tangential_trace, neumann_trace
