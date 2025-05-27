import numpy as np
from mpi4py import MPI
from rng import get_rng, rng_context, register_rng
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP
import time
import json
import os

# Example usage
def run_moe(
    moe_type="tp", 
    batch_size=8, 
    feature_dim=32, 
    hidden_dim=128, 
    output_dim=64, 
    num_experts=None,
    topk=2
):
    """
    Unified function to run different types of MoE models
    
    Args:
        moe_type: Type of MoE to run ("simple", "ep", or "tp")
        batch_size: Number of samples in the batch
        feature_dim: Dimension of input features
        hidden_dim: Hidden dimension for experts
        output_dim: Output dimension
        topk: Number of experts to route each input to
    """
    num_experts = mpi.get_size()
    
    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        if mpi.get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)
    
    model_class = {
        "simple": SimpleMoE,
        "ep": MoE_EP,
        "tp": MoE_TP
    }.get(moe_type, MoE_TP)
    
    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk
    )
    

    _ = moe(X)
    
    N = 5  
    start_time = time.time()
    for _ in range(N):
        outputs = moe(X)
    end_time = time.time()
    avg_duration_ms = 1000 * (end_time - start_time) / N
    
    # Print timing information
    if mpi.get_rank() == 0:
        print(f"Forward pass time for {moe_type} MoE: {avg_duration_ms:.4f} ms")
        print(f"Config: batch_size={batch_size}, feature_dim={feature_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, topk={topk}")

    return dict(
        outputs=outputs,
        avg_duration_ms=avg_duration_ms,
        config=dict(
            moe_type=moe_type,
            batch_size=batch_size,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=num_experts,
            topk=topk
        )
    )
    
    
def run_comprehensive_benchmark():
    """Run comprehensive benchmarks varying different parameters"""
    results = []
    
    rank = mpi.get_rank()
    register_rng("expert_with_rank", np.random.RandomState(rank + 100))
    
    batch_sizes = [4, 16, 64]
    model_dims = [(16, 64, 32), (32, 128, 64), (64, 256, 128)] 
    topk_values = [1, 2]
    
    world_size = mpi.get_size()
    
    for batch_size in batch_sizes:
        feature_dim = 32
        hidden_dim = world_size * ((128 // world_size) or 1)
        output_dim = world_size * ((64 // world_size) or 1)
        
        for moe_type in ["simple", "ep", "tp"]:
            result = run_moe(
                moe_type=moe_type,
                batch_size=batch_size,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                topk=2
            )
            results.append({
                "moe_type": moe_type,
                "batch_size": batch_size,
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "topk": 2,
                "duration_ms": result["avg_duration_ms"]
            })
    
    for feature_dim, hidden_dim, output_dim in model_dims:
        # Ensure dimensions are multiples of world_size for TP
        hidden_dim = world_size * ((hidden_dim // world_size) or 1)
        output_dim = world_size * ((output_dim // world_size) or 1)
        
        for moe_type in ["simple", "ep", "tp"]:
            result = run_moe(
                moe_type=moe_type,
                batch_size=16,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                topk=2
            )
            results.append({
                "moe_type": moe_type,
                "batch_size": 16,
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "topk": 2,
                "duration_ms": result["avg_duration_ms"]
            })
    
    for topk in topk_values:
        for moe_type in ["simple", "ep", "tp"]:
            result = run_moe(
                moe_type=moe_type,
                batch_size=16,
                feature_dim=32,
                hidden_dim=world_size * ((128 // world_size) or 1),
                output_dim=world_size * ((64 // world_size) or 1),
                topk=topk
            )
            results.append({
                "moe_type": moe_type,
                "batch_size": 16,
                "feature_dim": 32,
                "hidden_dim": world_size * ((128 // world_size) or 1),
                "output_dim": world_size * ((64 // world_size) or 1),
                "topk": topk,
                "duration_ms": result["avg_duration_ms"]
            })
     
    if mpi.get_rank() == 0:
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nBenchmark Results Summary:")
        print("--------------------------")
        
        print("\nVarying Batch Size (feature_dim=32, hidden_dim=128, output_dim=64, topk=2):")
        for batch_size in batch_sizes:
            print(f"  Batch Size = {batch_size}:")
            for moe_type in ["simple", "ep", "tp"]:
                matching_results = [r for r in results if r["moe_type"] == moe_type and 
                                    r["batch_size"] == batch_size and
                                    r["feature_dim"] == 32]
                if matching_results:
                    avg_time = matching_results[0]["duration_ms"]
                    print(f"    {moe_type.upper()}: {avg_time:.4f} ms")
        
        print("\nVarying Model Dimensions (batch_size=16, topk=2):")
        for feature_dim, hidden_dim, output_dim in model_dims:
            hidden_dim_adjusted = world_size * ((hidden_dim // world_size) or 1)
            output_dim_adjusted = world_size * ((output_dim // world_size) or 1)
            
            print(f"  Dimensions = ({feature_dim}, {hidden_dim_adjusted}, {output_dim_adjusted}):")
            for moe_type in ["simple", "ep", "tp"]:
                matching_results = [r for r in results if r["moe_type"] == moe_type and 
                                    r["batch_size"] == 16 and
                                    r["feature_dim"] == feature_dim and
                                    r["hidden_dim"] == hidden_dim_adjusted and
                                    r["output_dim"] == output_dim_adjusted]
                if matching_results:
                    avg_time = matching_results[0]["duration_ms"]
                    print(f"    {moe_type.upper()}: {avg_time:.4f} ms")
        
        print("\nVarying Top-k (batch_size=16, feature_dim=32, hidden_dim=128, output_dim=64):")
        for topk in topk_values:
            print(f"  Top-k = {topk}:")
            for moe_type in ["simple", "ep", "tp"]:
                matching_results = [r for r in results if r["moe_type"] == moe_type and 
                                   r["topk"] == topk and
                                   r["batch_size"] == 16]
                if matching_results:
                    avg_time = matching_results[0]["duration_ms"]
                    print(f"    {moe_type.upper()}: {avg_time:.4f} ms")

if __name__ == "__main__":
    run_comprehensive_benchmark()
