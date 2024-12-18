"""Provides classes and operations for performing geometric algebra
with PyTorch.

The `GeometricAlgebra` class is used to construct the algebra given a metric.
It exposes methods for operating on `torch.Tensor` instances where their last
axis is interpreted as blades of the algebra.
"""
import numbers
from typing import List, Union

import torch

from torchga.blades import (
    BladeKind,
    get_blade_indices_from_names,
    get_blade_of_kind_indices,
    get_blade_repr,
    invert_blade_indices,
)
from torchga.cayley import blades_from_bases, get_cayley_tensor
from torchga.mv import MultiVector
from torchga.mv_ops import mv_conv1d, mv_grade_automorphism, mv_multiply, mv_reversion


class GeometricAlgebra:
    """Class used for performing geometric algebra operations on `torch.Tensor` instances.
    Exposes methods for operating on `torch.Tensor` instances where their last
    axis is interpreted as blades of the algebra.
    Holds the metric and other quantities derived from it.
    """

    def __init__(self, metric: List[float]):
        """Creates a GeometricAlgebra object given a metric.
        The algebra will have as many basis vectors as there are
        elements in the metric.

        Args:
            metric: Metric as a list. Specifies what basis vectors square to
        """
        self._metric = torch.tensor(metric, dtype=torch.float32)

        self._num_bases = len(metric)
        self._bases = list(map(str, range(self._num_bases)))

        self._blades, self._blade_degrees = blades_from_bases(self._bases)
        self._blade_degrees = torch.tensor(self._blade_degrees, dtype=torch.float32)
        self._num_blades = len(self._blades)
        self._max_degree = torch.max(self._blade_degrees)

        # [Blades, Blades, Blades]

        #print(self.metric, torch.Tensor(self._bases), torch.Tensor(self._blades))
        
        #self._cayley, self._cayley_inner, self._cayley_outer = torch.tensor(
        #    get_cayley_tensor(self.metric, self._bases, self._blades), dtype=torch.float32
        #)

        self._cayley, self._cayley_inner, self._cayley_outer = get_cayley_tensor(self.metric, self._bases, self._blades)

        #self._cayley = torch.tensor(self._cayley, dtype=torch.float32)
        #self._cayley_inner = torch.tensor(self._cayley_inner, dtype=torch.float32)
        #self._cayley_outer = torch.tensor(self._cayley_outer, dtype=torch.float32)

        self._blade_mvs = torch.eye(self._num_blades)
        self._basis_mvs = self._blade_mvs[1 : 1 + self._num_bases]

        # Find the dual by looking at the anti-diagonal in the Cayley tensor.
        self._dual_blade_indices = []
        self._dual_blade_signs = []

        for blade_index in range(self._num_blades):
            dual_index = self.num_blades - blade_index - 1
            anti_diag = self._cayley[blade_index, dual_index]
            dual_sign = torch.gather(anti_diag, 0, torch.nonzero(anti_diag != 0.0)[:, 0])
            self._dual_blade_indices.append(dual_index)
            self._dual_blade_signs.append(dual_sign)


        self._dual_blade_indices = torch.tensor(
            self._dual_blade_indices, dtype=torch.int64
        )
        self._dual_blade_signs = torch.tensor(
            self._dual_blade_signs, dtype=torch.float32
        )

    def print(self, *args, **kwargs):
        """Same as the default `print` function but formats `torch.Tensor`
        instances that have as many elements on their last axis
        as the algebra has blades using `mv_repr()`.
        """

        def _is_mv(arg):
            return (
                isinstance(arg, torch.Tensor)
                and arg.ndimension() > 0
                and arg.shape[-1] == self.num_blades
            )

        new_args = [self.mv_repr(arg) if _is_mv(arg) else arg for arg in args]

        print(*new_args, **kwargs)

    @property
    def metric(self) -> torch.Tensor:
        """Metric list which contains the number that each
        basis vector in the algebra squares to
        (ie. the diagonal of the metric tensor).
        """
        return self._metric

    @property
    def cayley(self) -> torch.Tensor:
        """`MxMxM` tensor where `M` is the number of basis
        blades in the algebra. Used for calculating the
        geometric product:

        `a_i, b_j, cayley_ijk -> c_k`
        """
        return self._cayley

    @property
    def cayley_inner(self) -> torch.Tensor:
        """Analagous to cayley but for inner product."""
        return self._cayley_inner

    @property
    def cayley_outer(self) -> torch.Tensor:
        """Analagous to cayley but for outer product."""
        return self._cayley_outer

    @property
    def blades(self) -> List[str]:
        """List of all blade names.

        Blades are all possible independent combinations of
        basis vectors. Basis vectors are named starting
        from `"0"` and counting up. The scalar blade is the
        empty string `""`.

        Example
        - Bases: `["0", "1", "2"]`
        - Blades: `["", "0", "1", "2", "01", "02", "12", "012"]`
        """
        return self._blades

    @property
    def blade_mvs(self) -> torch.Tensor:
        """List of all blade tensors in the algebra."""
        return self._blade_mvs

    def dual_blade_indices(self) -> torch.Tensor:
        return self._dual_blade_indices

    @property
    def dual_blade_signs(self) -> torch.Tensor:
        return self._dual_blade_signs

    @property
    def num_blades(self) -> int:
        return self._num_blades

    @property
    def blade_degrees(self) -> torch.Tensor:
        return self._blade_degrees

    @property
    def max_degree(self) -> int:
        return self._max_degree

    @property
    def basis_mvs(self) -> torch.Tensor:
        return self._basis_mvs

    def get_kind_blade_indices(self, kind: BladeKind, invert: bool = False) -> torch.Tensor:
        """Find all indices of blades of a given kind in the algebra.

        Args:
            kind: kind of blade to give indices for
            invert: whether to return all blades not of the kind

        Returns:
            indices of blades of a given kind in the algebra
        """
        return get_blade_of_kind_indices(
            self.blade_degrees, kind, self.max_degree, invert=invert
        )

    def get_blade_indices_of_degree(self, degree: int) -> torch.Tensor:
        return torch.gather(
            torch.arange(self.num_blades), torch.nonzero(self.blade_degrees == degree).squeeze(-1)
        )

    def is_pure(self, tensor: torch.Tensor, blade_indices: torch.Tensor) -> bool:
        inverted_blade_indices = invert_blade_indices(self.num_blades, blade_indices)  # Implement the `invert_blade_indices` function
        return torch.all(torch.gather(tensor, -1, inverted_blade_indices) == 0)


    def is_pure_kind(self, tensor: torch.Tensor, kind: BladeKind) -> bool:
        """Returns whether the given tensor is purely of a given kind
        and has no non-zero values for blades not of the kind.

        Args:
            tensor: tensor to check purity for
            kind: kind of blade to check purity for

        Returns:
            Whether the tensor is purely of a given kind
            and has no non-zero values for blades not of the kind
        """
        # Ensure the tensor is a PyTorch tensor of dtype float32
        tensor = torch.as_tensor(tensor, dtype=torch.float32)

        #print(kind)
        
        # Get the indices of blades that are *not* of the specified kind
        inverted_kind_indices = self.get_kind_blade_indices(kind, invert=True)
        
        
        # If inverted_kind_indices is a list or array, convert it to a tensor
        inverted_kind_indices = torch.as_tensor(inverted_kind_indices, dtype=torch.long)
        
        # Use torch.index_select to gather elements along the last dimension
        inverted_blades = torch.index_select(tensor, dim=-1, index=inverted_kind_indices)
        
        # Check if all values in the gathered indices are zero
        return torch.all(inverted_blades == 0)




    def from_tensor(self, tensor: torch.Tensor, blade_indices: torch.Tensor) -> torch.Tensor:
        """
        Creates a geometric algebra torch.Tensor from a torch.Tensor and blade
        indices. The blade indices have to align with the last axis of the tensor.

        Args:
            tensor: torch.Tensor to take as values for the geometric algebra tensor
            blade_indices: Blade indices corresponding to the tensor.

        Returns:
            Geometric algebra torch.Tensor from tensor and blade indices
        """
        # Ensure blade_indices and tensor are in the correct dtype
        blade_indices = blade_indices.to(dtype=torch.long)
        #blade_indices = blade_indices.to(tensor.device)
        tensor = tensor.to(dtype=torch.float32)

        # Put last axis on the first axis, making scatter easier
        t = torch.cat([torch.tensor([tensor.dim() - 1]), torch.arange(0, tensor.dim() - 1)])
        t_inv = torch.cat([torch.arange(1, tensor.dim()), torch.tensor([0])])

        tensor = tensor.permute(*t)  # Transpose tensor

        # Define the output shape, with `self.num_blades` in the first dimension
        shape = tuple(int(dim) for dim in torch.cat(
        [torch.tensor([self.num_blades]), torch.tensor(tensor.shape[1:])]
        )) 

        # Initialize the output tensor with zeros
        output = torch.zeros(*shape, dtype=tensor.dtype)
        output = output.to(tensor.device)

        #print(output)
        #print(tensor)

        # Scatter values from tensor into the output tensor at blade_indices

        #print(blade_indices)
        #print(tensor)

        output.index_add_(0, blade_indices, tensor)

        return output.permute(*t_inv)  # Undo the transposition


    def from_tensor_with_kind(self, tensor: torch.Tensor, kind: BladeKind) -> torch.Tensor:
        """Creates a geometric algebra torch.Tensor from a torch.Tensor and a kind.
        The kind's blade indices have to align with the last axis of the tensor.

        Args:
            tensor: torch.Tensor to take as values for the geometric algebra tensor
            kind: Kind corresponding to the tensor

        Returns:
            Geometric algebra torch.Tensor from tensor and kind
        """
        # Convert tensor to float32 (if it's not already)
        tensor = tensor.to(dtype=torch.float32)
        
        # Get the kind's blade indices (this function needs to be adapted for PyTorch if needed)
        kind_indices = self.get_kind_blade_indices(kind)
        
        # Call from_tensor (assuming this function works similarly in PyTorch)
        return self.from_tensor(tensor, kind_indices)

    def from_scalar(self, scalar: numbers.Number) -> torch.Tensor:
        return self.from_tensor(torch.tensor([scalar], dtype=torch.float32), torch.tensor([0], dtype=torch.long))



    def e(self, *blades: List[str]) -> torch.Tensor:
        """Returns a geometric algebra torch.Tensor with the given blades set
        to 1.

        Args:
            blades: list of blade names, can be unnormalized

        Returns:
            torch.Tensor with blades set to 1
        """
        blade_signs, blade_indices = get_blade_indices_from_names(blades, self.blades)

        blade_indices = blade_indices.to(torch.long)

        # Don't allow duplicate indices
        assert blade_indices.shape[0] == len(torch.unique(blade_indices)), f"Duplicate blade indices: {blades}"

        # Prepare the tensor for the blades
        x = blade_signs.unsqueeze(-1) * self.blade_mvs[blade_indices]

        # Sum across the specified axis (axis=-2 in TF becomes axis=1 in PyTorch)
        return torch.sum(x, dim=-2)


    def __getattr__(self, name: str) -> torch.Tensor:
        if name.startswith("e") and (name[1:] == "" or int(name[1:]) >= 0):
            return self.e(name[1:])
        raise AttributeError

    def dual(self, tensor: torch.Tensor) -> torch.Tensor:
        #print(self._dual_blade_indices)
        return self.dual_blade_signs * torch.gather(tensor, -1, self._dual_blade_indices)

    def grade_automorphism(self, tensor: torch.Tensor) -> torch.Tensor:
        return mv_grade_automorphism(tensor, self.blade_degrees)  # Implement `mv_grade_automorphism`

    def reversion(self, tensor: torch.Tensor) -> torch.Tensor:
        return mv_reversion(tensor, self.blade_degrees)  # Implement `mv_reversion`

    def conjugation(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.grade_automorphism(self.reversion(tensor))

    def simple_inverse(self, a: torch.Tensor) -> torch.Tensor:
        rev_a = self.reversion(a)
        divisor = self.geom_prod(a, rev_a)  # Implement `geom_prod`

        if not self.is_pure_kind(divisor, "scalar"):
            raise Exception("Can't invert multi-vector (inversion divisor V ~V not scalar).")

        return rev_a / divisor[..., 0:1]

    def reg_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.dual(self.ext_prod(self.dual(a), self.dual(b)))

    def ext_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._cayley_outer = self._cayley_outer.to(a.device)
        return mv_multiply(a, b, self._cayley_outer)  # Implement `mv_multiply`

    def geom_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        #print(a)
        #print(b)
        self._cayley = self._cayley.to(a.device)
        return mv_multiply(a, b, self._cayley)  # Implement `mv_multiply`

    def inner_prod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._cayley_inner = self._cayley_inner.to(a.device)
        return mv_multiply(a, b, self._cayley_inner)  # Implement `mv_multiply`

    def geom_conv1d(self, a: torch.Tensor, k: torch.Tensor, stride: int, padding: str, dilations: Union[int, None] = None) -> torch.Tensor:
        return mv_conv1d(a, k, self._cayley.to(a.device), stride=stride, padding=padding)  # Implement `mv_conv1d`

    def mv_repr(self, a: torch.Tensor) -> str:
        if len(a.shape) == 1:
            return "MultiVector[%s]" % " + ".join(
                "%.2f*%s" % (value, get_blade_repr(blade_name))  # Implement `get_blade_repr`
                for value, blade_name in zip(a, self.blades)
                if value != 0
            )
        else:
            return f"MultiVector[batch_shape={str(a.shape[:-1])}]"

    def approx_exp(self, a: torch.Tensor, order: int = 50) -> torch.Tensor:
        v = self.from_scalar(1.0)
        result = self.from_scalar(1.0)
        for i in range(1, order + 1):
            v = self.geom_prod(a, v)
            i_factorial = torch.exp(torch.lgamma(torch.tensor(i + 1.0)))
            result += v / i_factorial
        return result

    def exp(self, a: torch.Tensor, square_scalar_tolerance: Union[float, None] = 1e-4) -> torch.Tensor:
        self_sq = self.geom_prod(a, a)

        if square_scalar_tolerance is not None:
            assert torch.all(torch.abs(self_sq[..., 1:]) < square_scalar_tolerance)

        scalar_self_sq = self_sq[..., :1]

        s_sqrt = torch.sign(scalar_self_sq) * torch.sqrt(torch.abs(scalar_self_sq))

        return torch.where(
            scalar_self_sq < 0,
            (self.from_tensor(torch.cos(s_sqrt), [0]) + a / s_sqrt * torch.sin(s_sqrt)),
            (self.from_tensor(torch.cosh(s_sqrt), [0]) + a / s_sqrt * torch.sinh(s_sqrt)),
        )

    def approx_log(self, a: torch.Tensor, order: int = 50) -> torch.Tensor:
        result = self.from_scalar(0.0)

        a_minus_one = a - self.from_scalar(1.0)
        v = None

        for i in range(1, order + 1):
            v = a_minus_one if v is None else v * a_minus_one
            result += (((-1.0) ** i) / i) * v

        return result

    def int_pow(self, a: torch.Tensor, n: int) -> torch.Tensor:
        """Returns the geometric algebra tensor to the power of an integer
        using repeated multiplication.

        Args:
            a: Geometric algebra tensor to raise
            n: integer power to raise the multivector to

        Returns:
            `a` to the power of `n`
        """
        #a = torch.tensor(a, dtype=torch.float32)
        a = a.to(torch.float32)

        if not isinstance(n, int):
            raise Exception("n must be an integer.")
        if n < 0:
            raise Exception("Can't raise to negative powers.")

        if n == 0:
            return torch.ones_like(a) * self.e("")

        result = a
        for i in range(n - 1):
            result = self.geom_prod(result, a)
        return result

    def keep_blades(self, a: torch.Tensor, blade_indices: List[int]) -> torch.Tensor:
        """Takes a geometric algebra tensor and returns it with only the given
        blade_indices as non-zeros.

        Args:
            a: Geometric algebra tensor to copy
            blade_indices: Indices for blades to keep

        Returns:
            `a` with only `blade_indices` components as non-zeros
        """
        #a = torch.tensor(a, dtype=torch.float32)
        a = a.to(torch.float32)
        #blade_indices = torch.tensor(blade_indices, dtype=torch.int64)
        blade_indices = blade_indices.to(torch.int64)

        blade_values = torch.gather(a, dim=-1, index=blade_indices)

        return self.from_tensor(blade_values, blade_indices)

    def keep_blades_with_name(
        self, a: torch.Tensor, blade_names: Union[List[str], str]
    ) -> torch.Tensor:
        """Takes a geometric algebra tensor and returns it with only the given
        blades as non-zeros.

        Args:
            a: Geometric algebra tensor to copy
            blade_names: Blades to keep

        Returns:
            `a` with only `blade_names` components as non-zeros
        """
        if isinstance(blade_names, str):
            blade_names = [blade_names]

        _, blade_indices = get_blade_indices_from_names(blade_names, self.blades)

        return self.keep_blades(a, blade_indices)

    def select_blades(self, a: torch.Tensor, blade_indices: List[int]) -> torch.Tensor:
        """Takes a geometric algebra tensor and returns a `torch.Tensor` with the
        blades in blade_indices on the last axis.

        Args:
            a: Geometric algebra tensor to copy
            blade_indices: Indices for blades to select

        Returns:
            `torch.Tensor` based on `a` with `blade_indices` on last axis.
        """
        #a = torch.tensor(a, dtype=torch.float32)
        a = a.to(torch.float32)
        #blade_indices = torch.tensor(blade_indices, dtype=torch.int64)
        blade_indices = torch.tensor(blade_indices)
        blade_indices = blade_indices.to(torch.int64)
        #blade_indices  = self.from_tensor(blade_indices)

        #print(blade_indices.shape)

        shape_of_a = a.shape  # Get shape (B, N, M)
        blade_indices_expanded = blade_indices.unsqueeze(-1)
        blade_indices_expanded = blade_indices_expanded.expand(shape_of_a) 


        #print(blade_indices_expanded.shape)
        #print(a.shape)  
        
        result = torch.gather(a, dim=-1, index=blade_indices_expanded)

        return result

    def select_blades_with_name(
        self, a: torch.Tensor, blade_names: Union[List[str], str]
    ) -> torch.Tensor:
        """Takes a geometric algebra tensor and returns a `torch.Tensor` with the
        blades in blade_names on the last axis.

        Args:
            a: Geometric algebra tensor to copy
            blade_names: Blades to keep

        Returns:
            `torch.Tensor` based on `a` with `blade_names` on last axis.
        """
        #a = torch.tensor(a, dtype=torch.float32)
        a = a.to(torch.float32)

        is_single_blade = isinstance(blade_names, str)
        if is_single_blade:
            blade_names = [blade_names]

        blade_signs, blade_indices = get_blade_indices_from_names(blade_names, self.blades)

        result = blade_signs * self.select_blades(a, blade_indices)

        if is_single_blade:
            return result[..., 0]

        return result

    def inverse(self, a: torch.Tensor) -> torch.Tensor:
        """Returns the inverted geometric algebra tensor
        `X^-1` such that `X * X^-1 = 1`.

        Using Shirokov's inverse algorithm that works in arbitrary dimensions,
        see https://arxiv.org/abs/2005.04015 Theorem 4.

        Args:
            a: Geometric algebra tensor to return inverse for

        Returns:
            inverted geometric algebra tensor
        """
        #a = torch.tensor(a, dtype=torch.float32)
        a = a.to(torch.float32)

        n = 2 ** ((len(self.metric) + 1) // 2)

        u = a
        for k in range(1, n):
            c = n / k * self.keep_blades_with_name(u, "")
            u_minus_c = u - c
            u = self.geom_prod(a, u_minus_c)

        if not self.is_pure_kind(u, BladeKind.SCALAR):
            raise Exception("Can't invert multi-vector (det U not scalar: %s)." % u)

        # adj / det
        return u_minus_c / u[..., :1]

    def __call__(self, a: torch.Tensor) -> "MultiVector":
        """Creates a `MultiVector` from a geometric algebra tensor.
        Mainly used as a wrapper for the algebra's functions for convenience.

        Args:
            a: Geometric algebra tensor to return `MultiVector` for

        Returns:
            `MultiVector` for `a`
        """
        return MultiVector(a, self)

    

