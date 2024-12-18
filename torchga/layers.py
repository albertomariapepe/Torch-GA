import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Union
from torchga.blades import BladeKind
from torchga.torchga import GeometricAlgebra

class GeometricAlgebraLayer(nn.Module):
    def __init__(self, algebra, **kwargs):
        super().__init__()
        self.algebra = algebra
        # Capture additional configuration in kwargs if necessary
        self.extra_args = kwargs

    @classmethod
    def from_config(cls, config):
        # Create an instance of the GeometricAlgebra if not already present in config
        if "algebra" not in config:
            assert "metric" in config, "Config must contain 'metric' if 'algebra' is not provided."
            config["algebra"] = GeometricAlgebra(config["metric"])
            del config["metric"]
        return cls(**config)

    def get_config(self):
        # Serialize the configuration for this layer
        config = {"metric": self.algebra.metric.clone().detach().cpu().numpy().tolist()}
        config.update(self.extra_args)
        return config


class TensorToGeometric(GeometricAlgebraLayer):
    """Layer for converting tensors with given blade indices to
    geometric algebra tensors.
    """

    def __init__(self, algebra: GeometricAlgebra, blade_indices: List[int], **kwargs):
        super().__init__(algebra, **kwargs)
        self.blade_indices = blade_indices.to(dtype=torch.long)

    def forward(self, inputs: Tensor):
        return self.algebra.from_tensor(inputs, blade_indices=self.blade_indices.to(inputs.device))

class TensorWithKindToGeometric(GeometricAlgebraLayer):
    """Layer for converting tensors with given blade kind to
    geometric algebra tensors.
    """

    def __init__(self, algebra: GeometricAlgebra, kind: BladeKind, **kwargs):
        super().__init__(algebra, **kwargs)
        self.kind = kind

    def forward(self, inputs: Tensor):
        return self.algebra.from_tensor_with_kind(inputs, kind=self.kind)

class GeometricToTensor(GeometricAlgebraLayer):
    """Layer for extracting given blades from geometric algebra tensors."""

    def __init__(self, algebra: GeometricAlgebra, blade_indices: List[int], **kwargs):
        super().__init__(algebra, **kwargs)
        self.blade_indices = blade_indices.to(dtype=torch.long)


    def forward(self, inputs: Tensor):
        return inputs.index_select(-1, self.blade_indices.to(inputs.device))

class GeometricToTensorWithKind(GeometricToTensor):
    """Layer for extracting blades of a kind from geometric algebra tensors."""

    def __init__(self, algebra: GeometricAlgebra, kind: BladeKind, **kwargs):
        blade_indices = algebra.get_kind_blade_indices(kind)
        super().__init__(algebra=algebra, blade_indices=blade_indices, **kwargs)

class GeometricProductDense(GeometricAlgebraLayer):
    """Analagous to PyTorch's Linear layer but using multivector-valued matrices
    instead of scalar ones and geometric multiplication instead of standard
    multiplication.
    """

    def __init__(
        self,
        algebra,  # Geometric Algebra instance
        num_input_units: int,
        num_output_units: int,
        blade_indices_kernel: torch.Tensor,
        blade_indices_bias: torch.Tensor = None,
        activation: str="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ):
        super(GeometricProductDense, self).__init__(algebra)

        self.algebra = algebra
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.blade_indices_kernel = blade_indices_kernel
        self.blade_indices_bias = blade_indices_bias
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        if activation == "relu":
            self.activation = nn.ReLU()


        # Initialize kernel and bias (we can use Glorot uniform and zeros for bias)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.blade_indices_kernel = blade_indices_kernel.to(torch.int64)
        self.blade_indices_bias = blade_indices_kernel.to(torch.int64)

        self.kernel = None
        self.bias = None

        # Kernel shape: (kernel_size, input_channels, output_channels, blade_indices_kernel)
        shape_kernel = (
            self.num_output_units,
            self.num_input_units,
            self.blade_indices_kernel.shape[0],
        )

        self.kernel = nn.Parameter(torch.empty(shape_kernel))
        nn.init.xavier_uniform_(self.kernel)  # Glorot uniform initializer

        if self.use_bias:
            # Initialize bias shape: (output_channels, blade_indices_bias)
            shape_bias = (self.num_output_units, self.blade_indices_bias.shape[0])
            self.bias = nn.Parameter(torch.empty(shape_bias))
            nn.init.zeros_(self.bias)  # Zero initializer for bias
        else:
            self.bias = None


    def forward(self, inputs):
        # Convert the kernel to a geometric object (using the algebra system)
        #print(inputs.shape)
        w_geom = self.algebra.from_tensor(self.kernel, self.blade_indices_kernel.to(inputs.device))
        #print(w_geom)
        
        # Expand the input tensor dimensions to align with geometric product operation
        # inputs_expanded will have a shape of [..., 1, I, X] where I is input channels and X is the last dimension.
        inputs_expanded = inputs.unsqueeze(inputs.dim() - 2)  # Equivalent to tf.expand_dims(inputs, axis=-2)
        #inputs_expanded = inputs
        #print(inputs_expanded.shape)

        #print(w_geom.shape)
        #print(self.algebra.reversion(w_geom).shape)
        #print(inputs_expanded.shape)
        
        
        # Perform geometric product element-wise and sum over the common axis (axis=-2)
        # Result shape will be [..., O, X] where O is the output channels and X is the last dimension.
        result = torch.sum(self.algebra.geom_prod(inputs_expanded, w_geom), dim=-2)
        #print(result.shape)
        #print("***")

        #print(result.shape)

        # If bias exists, apply it
        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias.to(inputs.device))
            #print(b_geom)
            result += b_geom

        # Apply activation function
        #print(self.activation)
        result = self.activation(result)
        return result

class GeometricSandwichProductDense(GeometricProductDense):
    """Analagous to PyTorch's Linear layer but using multivector-valued matrices
    instead of scalar ones and geometric sandwich multiplication instead of
    standard multiplication.
    """

    def __init__(
        self,
        algebra,  # Geometric Algebra instance
        num_input_units: int,
        num_output_units: int,
        blade_indices_kernel: torch.Tensor,
        blade_indices_bias: torch.Tensor = None,
        activation: str="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ):
        super(GeometricProductDense, self).__init__(algebra)

        self.algebra = algebra
        self.num_output_units = num_output_units
        self.blade_indices_kernel = blade_indices_kernel
        self.blade_indices_bias = blade_indices_bias
        self.num_input_units = num_input_units
        self.use_bias = use_bias
        

        if activation == "relu":
            self.activation = nn.ReLU()

        # Initialize kernel and bias (we can use Glorot uniform and zeros for bias)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.kernel = None
        self.bias = None

        self.blade_indices_kernel = blade_indices_kernel.to(torch.int64)
        self.blade_indices_bias = blade_indices_kernel.to(torch.int64)

        # Kernel shape: (kernel_size, input_channels, output_channels, blade_indices_kernel)
        shape_kernel = (
            self.num_output_units,  # C (channels) would be assigned when forward is called
            self.num_input_units,
            self.blade_indices_kernel.size(0),
        )

        # Initialize the kernel with the appropriate shape and the selected initializer
        self.kernel = nn.Parameter(torch.empty(shape_kernel))
        nn.init.xavier_uniform_(self.kernel)  # Glorot uniform initializer
        #nn.init.kaiming_uniform_(self.kernel, mode='fan_in', nonlinearity='relu')

        if self.use_bias:
            # Initialize bias shape: (output_channels, blade_indices_bias)
            shape_bias = (self.num_output_units, self.blade_indices_bias.size(0))
            self.bias = nn.Parameter(torch.empty(shape_bias))
            nn.init.zeros_(self.bias)  # Zero initializer for bias
        else:
            self.bias = None

    def forward(self, inputs: Tensor):
        w_geom = self.algebra.from_tensor(self.kernel, self.blade_indices_kernel.to(inputs.device))
    
        
        inputs_expanded = inputs.unsqueeze(inputs.dim() -2)
     
        result = torch.sum(
            self.algebra.geom_prod(
                w_geom,
                self.algebra.geom_prod(inputs_expanded, self.algebra.reversion(w_geom)),
            ),
            dim=-2,
        )
        
        #print(result.shape)
        #print("***")

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias.to(inputs.device))
            #print(b_geom)
            result += b_geom

        if self.activation:
            result = self.activation(result)

        return result


class GeometricProductElementwise(GeometricAlgebraLayer):

    def __init__(
        self,
        algebra,  # Geometric Algebra instance
        num_input_units: int,
        num_output_units: int,
        blade_indices_kernel: torch.Tensor,
        blade_indices_bias: torch.Tensor = None,
        activation: str="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ):
        super(GeometricProductElementwise, self).__init__(algebra)

        self.algebra = algebra
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units
        self.blade_indices_kernel = blade_indices_kernel
        self.blade_indices_bias = blade_indices_bias
        self.use_bias = use_bias
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint

        if activation == "relu":
            self.activation = nn.ReLU()


        # Initialize kernel and bias (we can use Glorot uniform and zeros for bias)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.blade_indices_kernel = blade_indices_kernel.to(torch.int64)
        self.blade_indices_bias = blade_indices_kernel.to(torch.int64)

        self.kernel = None
        self.bias = None

        # Kernel shape: (kernel_size, input_channels, output_channels, blade_indices_kernel)
        shape_kernel = (
            self.num_input_units,
            self.blade_indices_kernel.shape[0],
        )

        self.kernel = nn.Parameter(torch.empty(shape_kernel))
        nn.init.xavier_uniform_(self.kernel)  # Glorot uniform initializer

        if self.use_bias:
            # Initialize bias shape: (output_channels, blade_indices_bias)
            shape_bias = (self.num_input_units, self.blade_indices_bias.shape[0])
            self.bias = nn.Parameter(torch.empty(shape_bias))
            nn.init.zeros_(self.bias)  # Zero initializer for bias
        else:
            self.bias = None

    def forward(self, inputs):
        # Convert the kernel to a geometric object (using the algebra system)
        #print(inputs.shape)
        w_geom = self.algebra.from_tensor(self.kernel, self.blade_indices_kernel)
        #print(w_geom)
        
        # Expand the input tensor dimensions to align with geometric product operation
        # inputs_expanded will have a shape of [..., 1, I, X] where I is input channels and X is the last dimension.
        #inputs_expanded = inputs.unsqueeze(inputs.dim() - 2)  # Equivalent to tf.expand_dims(inputs, axis=-2)
        #inputs_expanded = inputs
        #print(inputs_expanded.shape)

        #print(w_geom.shape)
        #print(self.algebra.reversion(w_geom).shape)
        #print(inputs_expanded.shape)
        
        
        # Perform geometric product element-wise and sum over the common axis (axis=-2)
        # Result shape will be [..., O, X] where O is the output channels and X is the last dimension.
        result = self.algebra.geom_prod(inputs, w_geom)
        #print(result.shape)
        #print("***")

        #print(result.shape)

        # If bias exists, apply it
        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias)
            #print(b_geom.shape)
            result += b_geom
        
        #print("**")

        # Apply activation function
        #print(self.activation)
        result = self.activation(result)
        return result


class GeometricSandwichProductElementwise(GeometricProductElementwise):
    def __init__(
        self,
        algebra,  # Geometric Algebra instance
        num_input_units: int,
        num_output_units: int,
        blade_indices_kernel: torch.Tensor,
        blade_indices_bias: torch.Tensor = None,
        activation: str="relu",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ):
        super(GeometricProductElementwise, self).__init__(algebra)

        self.algebra = algebra
        self.num_output_units = num_output_units
        self.blade_indices_kernel = blade_indices_kernel
        self.blade_indices_bias = blade_indices_bias
        self.num_input_units = num_input_units
        self.use_bias = use_bias
        

        if activation == "relu":
            self.activation = nn.ReLU()

        # Initialize kernel and bias (we can use Glorot uniform and zeros for bias)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.kernel = None
        self.bias = None

        self.blade_indices_kernel = blade_indices_kernel.to(torch.int64)
        self.blade_indices_bias = blade_indices_kernel.to(torch.int64)


        shape_kernel = (
            self.num_input_units,
            self.blade_indices_kernel.size(0),
        )

        # Initialize the kernel with the appropriate shape and the selected initializer
        self.kernel = nn.Parameter(torch.empty(shape_kernel))
        nn.init.xavier_uniform_(self.kernel)  # Glorot uniform initializer
        #nn.init.kaiming_uniform_(self.kernel, mode='fan_in', nonlinearity='relu')

        if self.use_bias:
            # Initialize bias shape: (output_channels, blade_indices_bias)
            shape_bias = (self.num_input_units, self.blade_indices_bias.size(0))
            self.bias = nn.Parameter(torch.empty(shape_bias))
            nn.init.zeros_(self.bias)  # Zero initializer for bias
        else:
            self.bias = None


    def forward(self, inputs: Tensor):
        w_geom = self.algebra.from_tensor(self.kernel, self.blade_indices_kernel)
    
        
        #inputs_expanded = inputs.unsqueeze(inputs.dim() -2)
     
        result = self.algebra.geom_prod(
                w_geom,
                self.algebra.geom_prod(inputs, self.algebra.reversion(w_geom)),
            )
        
        #print(result.shape)
        #print("***")

        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias, self.blade_indices_bias)
            #print(b_geom)
            result += b_geom

        if self.activation:
            result = self.activation(result)

        return result


class GeometricProductConv1D(nn.Module):
    """Analogous to Keras' Conv1D layer but using multivector-valued kernels
    instead of scalar ones and geometric product instead of standard multiplication.
    """

    def __init__(
        self,
        algebra,  # Geometric Algebra instance
        num_input_filters: int,
        num_output_filters: int,
        kernel_size: int,
        stride: int,
        padding: str,
        blade_indices_kernel: torch.Tensor,
        blade_indices_bias: torch.Tensor = None,
        dilations: int = 1,
        activation: callable = None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    ):
        super(GeometricProductConv1D, self).__init__()

        self.algebra = algebra
        self.num_output_filters = num_output_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilations = dilations
        self.blade_indices_kernel = blade_indices_kernel
        self.blade_indices_bias = blade_indices_bias
        self.num_input_filters = num_input_filters
        self.use_bias = use_bias
        self.activation = activation

        # Initialize kernel and bias (we can use Glorot uniform and zeros for bias)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.kernel = None
        self.bias = None

        self.blade_indices_kernel = blade_indices_kernel.to(torch.int64)
        self.blade_indices_bias = blade_indices_kernel.to(torch.int64)

        # Kernel shape: (kernel_size, input_channels, output_channels, blade_indices_kernel)
        shape_kernel = (
            self.kernel_size,
            self.num_input_filters,  # C (channels) would be assigned when forward is called
            self.num_output_filters,
            self.blade_indices_kernel.size(0),
        )

        # Initialize the kernel with the appropriate shape and the selected initializer
        self.kernel = nn.Parameter(torch.empty(shape_kernel))
        nn.init.xavier_uniform_(self.kernel)  # Glorot uniform initializer

        if self.use_bias:
            # Initialize bias shape: (output_channels, blade_indices_bias)
            shape_bias = (self.num_output_filters, self.blade_indices_bias.size(0))
            self.bias = nn.Parameter(torch.empty(shape_bias))
            nn.init.zeros_(self.bias)  # Zero initializer for bias
        else:
            self.bias = None

    def forward(self, inputs):
        # Convert kernel tensor to geometric object using the algebra's `from_tensor` method
        k_geom = self.algebra.from_tensor(self.kernel.to(inputs.device), self.blade_indices_kernel.to(inputs.device))

        # Apply the geometric 1D convolution
        result = self.algebra.geom_conv1d(
            inputs,
            k_geom.to(inputs.device),
            stride=self.stride,
            padding=self.padding,
            dilations=self.dilations,
        )

        # If bias is used, add it after the convolution
        if self.bias is not None:
            b_geom = self.algebra.from_tensor(self.bias.to(inputs.device), self.blade_indices_bias.to(inputs.device))
            result += b_geom.to(inputs.device)

        # Apply activation function
        if self.activation:
            result = self.activation(result)

        return result

    def get_config(self):
        config = {
            "filters": self.num_output_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilations": self.dilations,
            "blade_indices_kernel": self.blade_indices_kernel,
            "blade_indices_bias": self.blade_indices_bias if self.blade_indices_bias is not None else None,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "activity_regularizer": self.activity_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_constraint": self.bias_constraint,
        }
        return config
