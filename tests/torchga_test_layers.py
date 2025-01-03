import unittest as ut
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from io import BytesIO

from torchga.torchga import GeometricAlgebra
from torchga.blades import BladeKind
from torchga.layers import (
    GeometricProductConv1D,
    GeometricProductDense,
    GeometricProductElementwise,
    GeometricSandwichProductDense,
    GeometricSandwichProductElementwise,
    GeometricToTensor,
    GeometricToTensorWithKind,
    TensorToGeometric,
    TensorWithKindToGeometric,
)


ga = GeometricAlgebra(metric=[1, 1, 1])


class TestKerasLayers(ut.TestCase):
    def assertTensorsEqual(self, a, b):
        self.assertTrue(torch.allclose(a, b), f"{a} not equal to {b}")

    def test_tensor_to_geometric(self):
        sta = GeometricAlgebra([1, -1, -1, -1])
        tensor = torch.ones([32, 4])
        gt_geom_tensor = torch.cat(
            [torch.zeros([32, 1]), torch.ones([32, 4]), torch.zeros([32, 11])], dim=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        tensor_to_geom_layer = TensorToGeometric(sta, vector_blade_indices)

        self.assertTensorsEqual(tensor_to_geom_layer(tensor), gt_geom_tensor)

    def test_tensor_with_kind_to_geometric(self):
        sta = GeometricAlgebra([1, -1, -1, -1])
        tensor = torch.ones([32, 4])
        gt_geom_tensor = torch.cat(
            [torch.zeros([32, 1]), torch.ones([32, 4]), torch.zeros([32, 11])], dim=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        tensor_kind_to_geom_layer = TensorWithKindToGeometric(sta, BladeKind.VECTOR)

        self.assertTensorsEqual(tensor_kind_to_geom_layer(tensor), gt_geom_tensor)

    def test_geometric_to_tensor(self):
        sta = GeometricAlgebra([1, -1, -1, -1])
        gt_tensor = torch.ones([32, 4])
        geom_tensor = torch.cat(
            [torch.zeros([32, 1]), torch.ones([32, 4]), torch.zeros([32, 11])], dim=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        geom_to_tensor_layer = GeometricToTensor(sta, vector_blade_indices)

        self.assertTensorsEqual(geom_to_tensor_layer(geom_tensor), gt_tensor)

    def test_geometric_to_tensor_with_kind(self):
        sta = GeometricAlgebra([1, -1, -1, -1])
        gt_tensor = torch.ones([32, 4])
        geom_tensor = torch.cat(
            [torch.zeros([32, 1]), torch.ones([32, 4]), torch.zeros([32, 11])], dim=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        geom_to_tensor_kind_layer = GeometricToTensorWithKind(sta, BladeKind.VECTOR)

        self.assertTensorsEqual(geom_to_tensor_kind_layer(geom_tensor), gt_tensor)

    def test_geometric_product_dense_v_v(self):
        sta = GeometricAlgebra([1, -1, -1, -1])

        geom_tensor = torch.cat(
            [torch.zeros([32, 6, 1]), torch.ones([32, 6, 4]), torch.zeros([32, 6, 11])], dim=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        geom_prod_layer = GeometricProductDense(
            sta,
            8,
            blade_indices_kernel=vector_blade_indices,
            blade_indices_bias=vector_blade_indices,
            bias_initializer=torch.randn,
        )

        result = geom_prod_layer(geom_tensor)

        # vector * vector + vector -> scalar + bivector + vector
        expected_result_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.assertTrue(sta.is_pure(result, expected_result_indices))

    def test_geometric_product_dense_s_mv(self):
        sta = GeometricAlgebra([1, -1, -1, -1])

        geom_tensor = torch.cat([torch.ones([20, 6, 1]), torch.zeros([20, 6, 15])], dim=-1)

        mv_blade_indices = list(range(16))

        geom_prod_layer = GeometricProductDense(
            sta,
            8,
            blade_indices_kernel=mv_blade_indices,
            blade_indices_bias=mv_blade_indices,
        )

        result = geom_prod_layer(geom_tensor)

        # scalar * multivector + multivector -> multivector
        self.assertTrue(torch.all(result != 0.0))

    def test_geometric_product_dense_sequence(self):
        sta = GeometricAlgebra([1, -1, -1, -1])

        tensor = torch.ones([20, 6, 4])

        vector_blade_indices = [1, 2, 3, 4]
        mv_blade_indices = list(range(16))

        scalar_bivector_blade_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        sequence = nn.Sequential(
            TensorToGeometric(sta, blade_indices=vector_blade_indices),
            GeometricProductDense(
                sta,
                8,
                blade_indices_kernel=vector_blade_indices,
                blade_indices_bias=vector_blade_indices,
                bias_initializer=torch.randn,
            ),
            GeometricToTensor(sta, blade_indices=scalar_bivector_blade_indices),
        )

        result = sequence(tensor)

        self.assertEqual(result.shape[-1], len(scalar_bivector_blade_indices))

    def test_geometric_sandwich_product_dense_v_v(self):
        sta = GeometricAlgebra([1, -1, -1, -1])

        geom_tensor = torch.cat(
            [torch.zeros([32, 6, 1]), torch.ones([32, 6, 4]), torch.zeros([32, 6, 11])], dim=-1
        )

        vector_blade_indices = [1, 2, 3, 4]

        result_indices = torch.cat(
            [
                sta.get_kind_blade_indices(BladeKind.VECTOR),
                sta.get_kind_blade_indices(BladeKind.TRIVECTOR),
            ],
            dim=0,
        )

        geom_prod_layer = GeometricSandwichProductDense(
            sta,
            8,
            blade_indices_kernel=vector_blade_indices,
            blade_indices_bias=result_indices,
            bias_initializer=torch.randn,
        )

        result = geom_prod_layer(geom_tensor)

        # vector * vector * ~vector + vector -> vector + trivector

        self.assertTrue(sta.is_pure(result, result_indices))


class TestKerasLayersSerializable(ut.TestCase):
    def assertTensorsEqual(self, a, b):
        self.assertTrue(torch.allclose(a, b), f"{a} not equal to {b}")

    def _test_layer_serializable(self, layer, inputs):
        # Create algebra
        algebra = layer.algebra

        # Create model
        model = nn.Sequential(layer)

        # Predict on inputs to compare later
        model_output = model(inputs)

        # Serialize model to virtual file
        model_file = BytesIO()
        torch.save(model, model_file)

        # Load model from stream
        model_file.seek(0)
        loaded_model = torch.load(model_file)

        # Predict on same inputs as before
        loaded_output = loaded_model(inputs)

        # Check same output for original and loaded model
        self.assertTensorsEqual(model_output, loaded_output)

        # Check same recreated algebra
        self.assertTensorsEqual(algebra.metric, loaded_model[0].algebra.metric)
        self.assertTensorsEqual(algebra.cayley, loaded_model[0].algebra.cayley)

    def test_geom_dense_serializable(self):
        # Create algebra
        sta = GeometricAlgebra([1, -1, -1, -1])
        vector_blade_indices = [1, 2, 3, 4]
        mv_blade_indices = list(range(16))

        # Create model
        self._test_layer_serializable(
            GeometricProductDense(
                sta,
                units=8,
                blade_indices_kernel=mv_blade_indices,
                blade_indices_bias=vector_blade_indices,
            ),
            torch.randn([3, 6, sta.num_blades], seed=0),
        )

    def test_sandwich_dense_serializable(self):
        # Create algebra
        sta = GeometricAlgebra([1, -1, -1, -1])
        vector_blade_indices = [1, 2, 3, 4]

        # Create model
        self._test_layer_serializable(
            GeometricSandwichProductDense(
                sta,
                8,
                blade_indices_kernel=vector_blade_indices,
                blade_indices_bias=vector_blade_indices,
            ),
            torch.randn([3, 6, sta.num_blades], seed=0),
        )

