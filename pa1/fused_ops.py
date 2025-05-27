from typing import Any, Dict, List
import torch
from auto_diff import *
import numpy as np

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        A, B = input_values
        
        matmul_output = torch.matmul(A, B)
        
        dims = tuple(range(-len(node.attrs['normalized_shape']), 0))
        mean = torch.mean(matmul_output, dim=dims, keepdim=True)
        var = torch.var(matmul_output, dim=dims, keepdim=True, unbiased=False)
        normalized = (matmul_output - mean) / torch.sqrt(var + node.attrs['eps'])
        
        node.attrs['mean'] = mean
        node.attrs['var'] = var
        node.attrs['normalized'] = normalized
        node.attrs['matmul_output'] = matmul_output
        
        return normalized

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A, B = node.inputs
        
        matmul_node = matmul(A, B)
        
        dims = tuple(range(-len(node.attrs['normalized_shape']), 0))
        N = float(np.prod([matmul_node.attrs.get('shape', [])[d] for d in dims]))
        
        mean_node = mean(matmul_node, dims, keepdim=True)
        centered = sub(matmul_node, mean_node)
        var_node = mean(power(centered, 2.0), dims, keepdim=True)
        std_node = sqrt(add_by_const(var_node, node.attrs['eps']))
        normalized = div(centered, std_node)
        
        dx_norm = output_grad
        dx_centered = div(dx_norm, std_node)
        
        dvar = mul_by_const(
            sum_op(mul(mul(dx_norm, normalized), div_by_const(std_node, -2.0)), dims, keepdim=True),
            1.0
        )
        
        dmean = mul_by_const(
            sum_op(mul_by_const(dx_centered, -1.0), dims, keepdim=True),
            1.0
        )
        
        dx = add(
            dx_centered,
            add(
                mul_by_const(mul(dvar, mul_by_const(centered, 2.0/N)), 1.0),
                mul_by_const(dmean, 1.0/N)
            )
        )
        
        grad_A = matmul(dx, transpose(B, -2, -1))
        grad_B = matmul(transpose(A, -2, -1), dx)
        
        return [grad_A, grad_B]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={"dim": dim},
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        A, B = input_values
        
        matmul_output = torch.matmul(A, B)
        
        dim = node.attrs['dim']
        max_x = torch.max(matmul_output, dim=dim, keepdim=True).values
        exp_x = torch.exp(matmul_output - max_x)
        softmax_output = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
        
        node.attrs['softmax_output'] = softmax_output
        node.attrs['matmul_output'] = matmul_output
        
        return softmax_output

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        A, B = node.inputs
        
        matmul_node = matmul(A, B)
        softmax_node = softmax(matmul_node, node.attrs['dim'])
        
        sum_term = sum_op(mul(output_grad, softmax_node), dim=node.attrs['dim'], keepdim=True)
        dx = mul(softmax_node, sub(output_grad, sum_term))
        
        grad_A = matmul(dx, transpose(B, -2, -1))
        grad_B = matmul(transpose(A, -2, -1), dx)
        
        return [grad_A, grad_B]

matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()