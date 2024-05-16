import bempp.api
from bempp.api.assembly.boundary_operator import BoundaryOperator
from bempp.api.operators.boundary.maxwell import osrc_mte

class osrcMtE(BoundaryOperator):
    def __init__(self, wf, domain, range_, dual_to_range, parameters=None):
        self.wf = wf
        self._domain = domain
        self._range = range_
        self._dual_to_range = dual_to_range
        self._parameters = parameters
        
    def weak_form(self):
        return self.wf

def osrc_MtE(domain, range_, dual_to_range, p1d, wave_number, parameters=None):
    mte = bempp.api.operators.boundary.maxwell.osrc_mte( [dual_to_range, p1d],  [dual_to_range, p1d],  [dual_to_range, p1d], wave_number)
    wf = mte._assemble()
    return osrcMtE(wf, domain, domain, dual_to_range, parameters)