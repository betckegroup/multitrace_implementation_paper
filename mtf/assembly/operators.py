from bempp.api.utils.data_types import combined_type
import numpy as _np
import numpy as _np
from bempp.api.assembly.discrete_boundary_operator import _DiscreteOperatorBase
from bempp.api.assembly.blocked_operator import BlockedOperatorBase, BlockedOperator

class GeneralizedBlockedOperator(BlockedOperatorBase):
    """
    Construct a generalized blocked operator.

    A generalized blocked operator has as components either

    - Simple operators
    - Blocked operators
    - Generalized blocked operators
    - Arrays of simple/blocked/generalized blocked operators

    """

    def __init__(self, array):
        """
        Initialize the operator.

        The input array must be a two-dimensional iterable that
        specifies the components. As long as the components make sense
        in terms of compatibility of spaces, the input will be
        accepted.

        """
        from bempp.api.assembly.boundary_operator import BoundaryOperator
        from collections.abc import Iterable

        def make_blocked(operator):
            """Turn a BoundaryOperator into a 1x1 blocked operator."""
            blocked_operator = BlockedOperator(1, 1)
            blocked_operator[0, 0] = operator
            return blocked_operator

        self._ops = []
        self._components_per_row = None
        self._components_per_column = None

        # First iterate through the array and transform each component into a
        # generalized blocked operator.

        for row in array:
            current_row = []
            for elem in row:
                if isinstance(elem, Iterable):
                    current_row.append(GeneralizedBlockedOperator(elem))
                elif isinstance(elem, BoundaryOperator):
                    current_row.append(make_blocked(elem))
                elif isinstance(elem, BlockedOperatorBase):
                    current_row.append(elem)
                else:
                    raise ValueError(
                        "Cannot process element of type: {0}".format(type(elem))
                    )
            self._ops.append(current_row)

            all_domain_spaces = []
            all_range_spaces = []
            all_dual_to_range_spaces = []

            for row in self._ops:
                range_spaces = row[0].range_spaces
                dual_to_range_spaces = row[0].dual_to_range_spaces
                domain_spaces = []
                for elem in row:
                    if elem.range_spaces != range_spaces:
                        raise ValueError("Incompatible range spaces detected.")
                    if elem.dual_to_range_spaces != dual_to_range_spaces:
                        raise ValueError("Incompatible dual to range spaces detected.")
                    domain_spaces.extend(elem.domain_spaces)
                all_range_spaces.extend(range_spaces)
                all_dual_to_range_spaces.extend(dual_to_range_spaces)
                if all_domain_spaces:
                    # We have already processed one row
                    # and compare domain spaces to it.
                    if domain_spaces != all_domain_spaces:
                        raise ValueError("Incompatible domain spaces detected.")
                else:
                    # We are at the first row.
                    all_domain_spaces = domain_spaces

            self._domain_spaces = tuple(all_domain_spaces)
            self._dual_to_range_spaces = tuple(all_dual_to_range_spaces)
            self._range_spaces = tuple(all_range_spaces)

            super().__init__()

    def _assemble(self):
        """Implement the weak form."""
        assembled_list = []
        for row in self._ops:
            assembled_row = []
            for elem in row:
                assembled_row.append(elem.weak_form())
            assembled_list.append(assembled_row)
        return GeneralizedDiscreteBlockedOperator(assembled_list)

    @property
    def range_spaces(self):
        """Return the list of range spaces."""
        return tuple(self._range_spaces)

    @property
    def dual_to_range_spaces(self):
        """Return the list of dual_to_range spaces."""
        return tuple(self._dual_to_range_spaces)

    @property
    def domain_spaces(self):
        """Return the list of domain spaces."""
        return tuple(self._domain_spaces)

class GeneralizedDiscreteBlockedOperator(_DiscreteOperatorBase):
    """A discrete generalized blocked operator."""

    def __init__(self, operators):
        """Initialize a generalized blocked operator."""
        from bempp.api.utils.data_types import combined_type

        self._operators = operators

        shape = [0, 0]
        # Get column dimension
        for elem in operators[0]:
            shape[1] += elem.shape[1]
        # Get row dimension
        for row in operators:
            shape[0] += row[0].shape[0]

        shape = tuple(shape)

        # Get dtype

        dtype = operators[0][0].dtype
        for row in operators:
            for elem in row:
                dtype = combined_type(dtype, elem.dtype)

        # Sanity check of dimensions

        for row in operators:
            row_dim = row[0].shape[0]
            column_dim = 0
            for elem in row:
                if elem.shape[0] != row_dim:
                    raise ValueError("Incompatible dimensions detected.")
                column_dim += elem.shape[1]
            if column_dim != shape[1]:
                raise ValueError("Incompatible dimensions detected.")

        super().__init__(dtype, shape)

    def to_dense(self):
        """Return dense matrix."""
        rows = []
        for row in self._operators:
            rows.append([op.to_dense() for op in row])
        return _np.block(rows)

    def matvec(self, other):
        """Implement the matrix/vector product."""
        from bempp.api.utils.data_types import combined_type

        row_count = 0
        output = _np.zeros(
            (self.shape[0]),
            dtype=combined_type(self.dtype, other.dtype),
        )
        
        for row in self._operators:
            row_dim = row[0].shape[0]
            column_count = 0
            for elem in row:
                matvec = elem.matvec(other[column_count : column_count + elem.shape[1]])
                output[row_count : row_count + row_dim] += (
                    matvec
                )
                column_count += elem.shape[1]
            row_count += row_dim

        return output

    def _matmat(self, other):
        """Implement the matrix/vector product."""
        from bempp.api.utils.data_types import combined_type

        row_count = 0
        output = _np.zeros(
            (self.shape[0], other.shape[1]),
            dtype=combined_type(self.dtype, other.dtype),
        )

        for row in self._operators:
            row_dim = row[0].shape[0]
            column_count = 0
            for elem in row:
                output[row_count : row_count + row_dim, :] += (
                    elem @ other[column_count : column_count + elem.shape[1], :]
                )
                column_count += elem.shape[1]
            row_count += row_dim

        return output


