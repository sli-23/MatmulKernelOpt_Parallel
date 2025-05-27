import numpy as np

from mpiwrapper import mpi
from rng import get_rng, rng_context


class Linear:
    """Simple linear layer y = xW + b"""

    def __init__(self, in_features, out_features):
        # Use default RNG for all other operations - no need for context
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Expert network with one hidden layer and ReLU activation"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Use rank-specific RNG for expert initialization
        with rng_context('expert'):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Routes inputs to experts using softmax-based gating"""

    def __init__(self, input_dim, num_experts):
        # Router should be consistent across all ranks, so use default RNG
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)

        # Softmax for routing probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Select top-k experts
        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)

        # Normalize gates to sum to 1
        gates = gates / np.sum(gates, axis=1, keepdims=True)

        return indices, gates


class ShardedLinear:
    """
    Linear layer that is sharded across processes
    Each process only holds a portion of the weight matrix
    
    Handles cases where out_features is not evenly divisible by the world size
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        # Save the original output features
        self.out_features_global = out_features
        
        # Calculate padded output features to make it divisible by world_size
        self.padded_out_features = out_features
        if out_features % self.world_size != 0:
            self.padded_out_features = ((out_features // self.world_size) + 1) * self.world_size
            
        # Calculate the local output dimension
        self.local_out_features = self.padded_out_features // self.world_size
        
        # Calculate output offset for this rank
        self.output_offset = self.rank * self.local_out_features
        
        # Determine if this rank handles actual features or just padding
        self.is_padding_rank = self.output_offset >= out_features
        
        # Adjust local_out_features if this rank handles the boundary
        if not self.is_padding_rank and (self.output_offset + self.local_out_features > out_features):
            self.local_out_features = out_features - self.output_offset
            
        # Initialize local weights and bias
        if not self.is_padding_rank and self.local_out_features > 0:
            self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
            self.bias = get_rng().randn(self.local_out_features)
        else:
            # Create empty arrays for padding ranks
            self.weight = np.zeros((in_features, 0))
            self.bias = np.zeros(0)

    def __call__(self, x):
        # Handle empty batch case
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features_global))

        # Create a buffer for the full output
        result = np.zeros((x.shape[0], self.out_features_global), dtype=np.float32)
        
        # Only compute output if this rank has actual features to process
        if not self.is_padding_rank and self.local_out_features > 0:
            local_output = np.dot(x, self.weight) + self.bias
        else:
            local_output = np.zeros((x.shape[0], 0))
            
        # Gather all local outputs
        all_outputs = mpi.comm.allgather(local_output)
        
        # Place each part in the correct position in the result
        current_idx = 0
        for i, local_part in enumerate(all_outputs):
            # Skip empty parts
            if local_part.shape[1] == 0:
                continue
                
            # Calculate the actual size to copy (handle boundary case)
            size_to_copy = min(local_part.shape[1], self.out_features_global - current_idx)
            if size_to_copy <= 0:
                continue
                
            # Copy the relevant portion
            result[:, current_idx:current_idx + size_to_copy] = local_part[:, :size_to_copy]
            current_idx += size_to_copy
            
            # Stop if we've filled the entire output
            if current_idx >= self.out_features_global:
                break

        return result


class ShardedExpert:
    """Expert network with one hidden layer and ReLU activation, sharded across processes"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Use rank-specific RNG for expert initialization
        with rng_context('expert'):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class MoE_TP:
    """
    Distributed Mixture of Experts using MPI for tensor parallelism
    
    TP-style MoE:
    - Each process holds a portion of every expert (sharded experts)
    - Router is replicated on all processes
    - All-to-all and all-gather communication patterns for processing
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Total number of experts in the model
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        # Create router (replicated on all processes)
        with rng_context('router'):
            self.router = Router(input_dim, num_experts)

        # Create sharded experts - each expert is sharded across all processes
        with rng_context('expert'):
            self.experts = [ShardedExpert(input_dim, hidden_dim, output_dim)
                            for _ in range(num_experts)]

        print(f"[Rank {self.rank}] Holding portions of all {num_experts} experts")

    def forward(self, x):
        """
        Distributed forward pass through the MoE model using tensor parallelism
        with optimized batch processing
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        indices, gates = self.router(x, self.topk)
        
        for i in range(batch_size):
            for k in range(self.topk):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                expert_output = self.experts[expert_idx](x[i:i+1])
                outputs[i] += gate * expert_output[0]
                
        return outputs

    def __call__(self, x):
        return self.forward(x)


class SimpleMoE:
    """
    Simple reference implementation of Mixture of Experts.
    
    This class implements a basic MoE model that routes inputs to a subset
    of experts and combines their outputs using learned gating weights.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Number of expert networks
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        # Create router network
        with rng_context('router'):
            self.router = Router(input_dim, num_experts)

        # Create expert networks
        with rng_context('expert'):
            self.experts = [Expert(input_dim, hidden_dim, output_dim)
                            for _ in range(num_experts)]

    def forward(self, x):
        """
        Forward pass through the MoE model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """

        batch_size = x.shape[0]
        indices, gates = self.router(x, self.topk)
        outputs = np.zeros((batch_size, self.output_dim))

        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i:i + 1] 
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]

        return outputs

    def __call__(self, x):
        return self.forward(x)


class MoE_EP:
    """
    Distributed Mixture of Experts using MPI for expert parallelism
    
    EP-style MoE: 
    Each process hosts exactly one expert. Router is replicated on all processes.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts 
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.get_rank()

        with rng_context('router'):
            self.router = Router(input_dim, self.num_experts)

        with rng_context('expert_with_rank'):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Distributed forward pass through the MoE model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        indices, gates = self.router(x, self.topk)

        expert_to_tokens = [[] for _ in range(self.num_experts)]
        token_to_experts = [[] for _ in range(batch_size)]
        token_to_gates = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for k in range(self.topk):
                expert_idx = indices[i, k]
                gate_value = gates[i, k]
                expert_to_tokens[expert_idx].append(i)
                token_to_experts[i].append(expert_idx)
                token_to_gates[i].append(gate_value)
        
        my_expert_idx = self.rank
        my_tokens = expert_to_tokens[my_expert_idx]
        
        my_token_inputs = x[my_tokens] if len(my_tokens) > 0 else np.zeros((0, self.input_dim))
        my_token_outputs = self.expert(my_token_inputs) if len(my_tokens) > 0 else np.zeros((0, self.output_dim))
        
        send_data = (my_tokens, my_token_outputs)
        
        all_expert_data = mpi.comm.allgather(send_data)
        
        for expert_idx, (tokens, token_outputs) in enumerate(all_expert_data):
            for i, token_idx in enumerate(tokens):
                expert_pos = token_to_experts[token_idx].index(expert_idx) if expert_idx in token_to_experts[token_idx] else -1
                if expert_pos >= 0:
                    gate_value = token_to_gates[token_idx][expert_pos]
                    outputs[token_idx] += gate_value * token_outputs[i]
        
        return outputs

    def __call__(self, x):
        return self.forward(x)
