# Torch-GA: building Geometric Algebra Networks in PyTorch
<p align="center">
  <img src="https://github.com/albertomariapepe/Torch-GA/blob/main/logo.tiff?raw=true" width="250" height="250">
</p>

Python package for Geometric / Clifford Algebra in PyTorch, adapted from [Tensorflow GA](https://github.com/RobinKa/tfga).

This project allows to build basic layers in Geometric / Clifford Algebra of **any dimensionality**. 

You are more than welcome to expand and contribute with new types of layers.

## Requirements

  ```python 3```
  ```torch```
  ```numpy```
  
## Getting Started

```git clone https://github.com/albertomariapepe/Torch-GA.git```

## Example 1: GA Fundamentals

```python

#importing modules
import torch
from torchga.torchga import GeometricAlgebra

#defining an algebra
sta = GeometricAlgebra([1, -1, -1, -1])

#geometric product between bases e0 and e1
sta.print(sta.geom_prod(sta.e0, sta.e1))

#basic operations
a = sta.geom_prod(sta.e0, sta.from_scalar(4.0))
b = sta.geom_prod(sta.from_scalar(9.0), sta.e1)

sta.print("a:", a)
sta.print("~a:", sta.reversion(a))
sta.print("b:", b)
sta.print("~b:", sta.reversion(b))

c = sta.geom_prod(sta.from_scalar(9.0), sta.e12)
sta.print("~c:", sta.reversion(c))

```

## Example 2: GA Neural Networks

```python

#importing
import torch.nn as nn
from torchga.torchga import GeometricAlgebra
from torchga.layers import GeometricProductDense, TensorToGeometric, GeometricToTensor

#defining the algebra and useful indices
ga = GeometricAlgebra([1, 1])
s_indices = [0]
v_indices = [1, 2]
mv_indices = torch.arange(0, ga.num_blades)

#defining the architecture: from triplets 2D points (i.e. vectors) defining a triangle --> single scalar area
model = nn.Sequential(
    TensorToGeometric(ga, blade_indices=v_indices),
    GeometricProductDense(
        ga, num_input_units=3, num_output_units=64, activation="relu",
        blade_indices_kernel=mv_indices,
        blade_indices_bias=mv_indices
    ),
    GeometricProductDense(
        ga, num_input_units=64, num_output_units=64, activation="relu",
        blade_indices_kernel=mv_indices,
        blade_indices_bias=mv_indices
    ),
    GeometricProductElementwise(
        ga, num_input_units=64, num_output_units=64, activation="relu",
        blade_indices_kernel=mv_indices,
        blade_indices_bias=mv_indices
    ),
    GeometricProductDense(
        ga, num_input_units=64, num_output_units=1,
        blade_indices_kernel=mv_indices,
        blade_indices_bias=s_indices
    ),
    GeometricToTensor(ga, blade_indices=s_indices)
)


#check dimensionality
sample_points = torch.randn([1000, 3, 2])
print("Samples:", sample_points[0])
print("Model(Samples):", model(sample_points).shape)
```


## Available Layers


| Class                              | Description                                                                                                                                  |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| GeometricProductDense              | Analagous to Keras' [Dense] with multivector-valued weights and biases. Each term in the matrix multiplication does the geometric product x * w. |
| GeometricSandwichProductDense      | Analagous to Keras' [Dense] with multivector-valued weights and biases. Each term in the matrix multiplication does the geometric product w *x * ~w. |
| GeometricProductElementwise        | Performs multivector-valued elementwise geometric product of the input units with a different weight for each unit.                           |
| GeometricSandwichProductElementwise| Performs multivector-valued elementwise geometric sandwich product of the input units with a different weight for each unit.               |
| GeometricProductConv1D             | Analagous to Keras' [Conv1D] with multivector-valued kernels and biases. Each term in the kernel multiplication does the geometric product x * k. |
| TensorToGeometric                  | Converts from a [torch.Tensor] to the geometric algebra [torch.Tensor] with as many blades on the last axis as basis blades in the algebra where blade indices determine which basis blades the input's values belong to. |
| GeometricToTensor                  | Converts from a geometric algebra [torch.Tensor] with as many blades on the last axis as basis blades in the algebra to a torch.Tensor where blade indices determine which basis blades we extract for the output. |
| TensorWithKindToGeometric          | Same as [TensorToGeometric] but using [BladeKind] (eg. "bivector", "even") instead of blade indices.                                         |
| GeometricToTensorWithKind          | Same as [GeometricToTensor] but using [BladeKind] (eg. "bivector", "even") instead of blade indices.         



## Tutorial Notebooks

<img src="https://github.com/albertomariapepe/Torch-GA/blob/main/logo.tiff?raw=true" width="10" height="10"> [GA Fundamentals with Torch-GA](https://github.com/albertomariapepe/Torch-GA/blob/main/tests/torchga.ipynb)


<img src="https://github.com/albertomariapepe/Torch-GA/blob/main/logo.tiff?raw=true" width="10" height="10"> [GA Layers with Torch-GA](https://github.com/albertomariapepe/Torch-GA/blob/main/tests/torchga_test_triangles.ipynb)


<img src="https://github.com/albertomariapepe/Torch-GA/blob/main/logo.tiff?raw=true" width="10" height="10"> [1D Convolutions with Torch-GA](https://github.com/albertomariapepe/Torch-GA/blob/main/tests/torchga_test_conv.ipynb)

<img src="https://github.com/albertomariapepe/Torch-GA/blob/main/logo.tiff?raw=true" width="10" height="10"> [Projective Geometric Algebra](https://github.com/albertomariapepe/Torch-GA/blob/main/tests/torchga_test_pga.ipynb)
