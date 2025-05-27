import argparse
import json
import math
from typing import Tuple, Dict, Any

# Constants for memory and compute calculations
BYTES_PER_PARAM_FP16 = 2
ADAM_STATES_PER_PARAM = 2
FLOPS_PER_TOKEN_PER_PARAM = 6
GB_CONVERSION = 1024**3

def model_training_cost_analysis_llama(model_config_path: str) -> Tuple[int, float, float]:
    with open(model_config_path, 'r') as f:
        config = json.load(f)
    
    h = config['hidden_size']
    V = config['vocab_size']
    L = config['num_hidden_layers']
    i_size = config['intermediate_size']
    s = config['max_position_embeddings']
    n_heads = config['num_attention_heads']
    d_head = h // n_heads
    
    emb_params = V * h
    
    q_proj = h * h
    k_proj = h * h
    v_proj = h * h
    o_proj = h * h
    attn_params = q_proj + k_proj + v_proj + o_proj
    
    gate_proj = h * i_size
    up_proj = h * i_size
    down_proj = i_size * h
    mlp_params = gate_proj + up_proj + down_proj
    
    norm_params = 2 * h
    layer_params = attn_params + mlp_params + norm_params
    final_norm_params = h
    total_params = emb_params + (L * layer_params) + final_norm_params
    
    attn_proj_flops = 4 * s * h * h
    attn_qk_flops = s * s * n_heads * d_head
    attn_softmax_flops = s * s * n_heads
    attn_av_flops = s * s * n_heads * d_head
    attn_flops = attn_proj_flops + attn_qk_flops + attn_softmax_flops + attn_av_flops
    
    mlp_up_flops = 2 * s * h * i_size
    mlp_act_flops = 2 * s * i_size
    mlp_down_flops = s * i_size * h
    mlp_flops = mlp_up_flops + mlp_act_flops + mlp_down_flops
    
    norm_flops = 5 * s * h * 2
    residual_flops = 2 * s * h
    
    base_flops_per_layer = attn_flops + mlp_flops + norm_flops + residual_flops
    flops_layer_TF = base_flops_per_layer / 1e12
    
    param_mem = layer_params * BYTES_PER_PARAM_FP16
    chunk_size = s // 8
    attn_mem = (3 * s * h + chunk_size * s) * BYTES_PER_PARAM_FP16
    
    mlp_mem = s * i_size * BYTES_PER_PARAM_FP16
    act_mem = attn_mem + mlp_mem
    grad_mem = layer_params * BYTES_PER_PARAM_FP16
    
    opt_mem = layer_params * ADAM_STATES_PER_PARAM * 2
    mem_GB = (param_mem + act_mem + grad_mem + opt_mem) / GB_CONVERSION
    
    return total_params, flops_layer_TF, mem_GB

def model_training_cost_analysis_deepseek(config_path: str) -> Tuple[int, float, float]:
    with open(config_path) as f:
        config = json.load(f)
    
    h = config['hidden_size']
    L = config['num_hidden_layers']
    s = config['max_position_embeddings']
    
    m_size = config.get('moe_intermediate_size', 0)
    n_exp = config.get('n_routed_experts', 0)
    ep_size = config.get('ep_size', 1)
    top_k = config.get('num_experts_per_tok', 0)
    
    if m_size == 0:
        m_size = h // 2
    if n_exp == 0:
        n_exp = 8
    if top_k == 0:
        top_k = 2
    
    expert_params = (h * m_size + m_size * h) / ep_size
    
    layer_params = 4 * h**2 + h * n_exp + expert_params * n_exp + 2 * h
    
    total_params = int(L * layer_params + h)
    
    tokens_per_expert = (s * top_k) / n_exp
    expert_flops = tokens_per_expert * 2 * h * m_size
    total_flops = s * h * n_exp + expert_flops * n_exp
    tflops = total_flops / 1e12
    
    params_mem = layer_params * BYTES_PER_PARAM_FP16 / ep_size
    optim_mem = params_mem * ADAM_STATES_PER_PARAM * 2
    
    act_mem = (
        s * h * 4 * BYTES_PER_PARAM_FP16 +
        s * top_k * m_size * BYTES_PER_PARAM_FP16 / n_exp / ep_size
    )
    
    mem_gb = (params_mem + optim_mem + act_mem) / GB_CONVERSION
    
    return total_params, round(tflops, 1), round(mem_gb, 1)

def get_optimal_N_D_from_cost(cost_budget: float) -> Tuple[int, int, int, str]:
    gpus = {
        'A100': {'cost_per_hour': 4.0, 'peak_flops': 312e12},
        'V100': {'cost_per_hour': 2.5, 'peak_flops': 125e12},
        'T4': {'cost_per_hour': 1.0, 'peak_flops': 65e12}
    }
    
    mfu = 0.4
    
    best_gpu = max(gpus.keys(), 
                  key=lambda g: gpus[g]['peak_flops'] * mfu / gpus[g]['cost_per_hour'])
    
    gpu_info = gpus[best_gpu]
    effective_flops_per_hour = gpu_info['peak_flops'] * mfu
    training_hours = cost_budget / gpu_info['cost_per_hour']
    training_budget_flops = int(effective_flops_per_hour * training_hours)
    
    k = 20
    
    N = int(math.sqrt(training_budget_flops / (6 * k)))
    
    N_billions = round(N / 1e9, 1)
    N = int(N_billions * 1e9)
    
    D = int(training_budget_flops / (6 * N))
    
    return N, D, training_budget_flops, best_gpu

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")