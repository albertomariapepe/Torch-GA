"""Operations on geometric algebra tensors used internally."""
from typing import Union

import torch


def mv_multiply(
    a_blade_values: torch.Tensor, b_blade_values: torch.Tensor, cayley: torch.Tensor
) -> torch.Tensor:
    # ...i, ijk -> ...jk

    #print(a_blade_values)
    #print(b_blade_values)
    #print(cayley)

    x = torch.tensordot(a_blade_values, cayley, dims=([-1], [0]))

    # ...1j, ...jk -> ...1k
    x = torch.matmul(b_blade_values.unsqueeze(-1).transpose(-1, -2), x)

    # ...1k -> ...k
    x = x.squeeze(-2)

    return x


def mv_conv1d(
    a_blade_values: torch.Tensor,
    k_blade_values: torch.Tensor,
    cayley: torch.Tensor,
    stride: int,
    padding: str,
    dilations: Union[int, None] = None,
) -> torch.Tensor:
    # Kernel size (first dimension of k_blade_values)
    kernel_size = k_blade_values.shape[0]
    
    # Batch shape of the input tensor
    a_batch_shape = a_blade_values.shape[:-3]
    a_blade_S, a_blade_CI, a_blade_BI = a_blade_values.shape[-3:]

    # Reshape a_blade_values to fit 4D `[batch, channels, height, width]` for unfold
    # New shape: [*, S, 1, CI*BI]
    a_image = a_blade_values.reshape(*a_batch_shape, a_blade_S, 1, a_blade_CI * a_blade_BI)

    # Define unfold for extracting patches
    unfold = torch.nn.Unfold(
        kernel_size=(kernel_size, 1),
        dilation=(1, 1),
        padding=(kernel_size // 2, 0) if padding == "SAME" else (0, 0),
        stride=(stride, 1)
    )

    # Apply unfold and reshape patches to have the correct output shape
    # Output shape before reshape: [batch, K*CI*BI, P], where P is the number of patches
    a_patches = unfold(a_image.reshape(-1, 1, a_blade_S, a_blade_CI * a_blade_BI))
    a_slices = a_patches.transpose(1, 2).reshape(*a_batch_shape, -1, kernel_size, a_blade_CI, a_blade_BI)

    #print(a_slices.shape)
    #print(k_blade_values.shape)

    # Perform einsum for final computation
    x = torch.einsum("...abcd,bcfg,dgh->...afh", a_slices, k_blade_values, cayley)

    return x.flip(dims=[1])


def mv_reversion(a_blade_values, algebra_blade_degrees):
    algebra_blade_degrees = algebra_blade_degrees.to(torch.float32)

    # for each blade, 0 if even number of swaps required, else 1
    odd_swaps = (torch.floor(algebra_blade_degrees * (algebra_blade_degrees - 0.5)) % 2).to(torch.float32)

    # [0, 1] -> [-1, 1]
    reversion_signs = 1.0 - 2.0 * odd_swaps
    reversion_signs = reversion_signs.to(a_blade_values.device)

    return reversion_signs * a_blade_values


def mv_grade_automorphism(a_blade_values, algebra_blade_degrees):
    algebra_blade_degrees = algebra_blade_degrees.to(torch.float32)
    signs = 1.0 - 2.0 * (algebra_blade_degrees % 2.0)
    return signs * a_blade_values
