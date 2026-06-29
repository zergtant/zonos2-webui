from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from zonos2.layers.moe.fused_moe.topk import select_experts


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)

    raise NotImplementedError("The ZeroGPU fallback MoE path does not use block alignment.")
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    is_marlin: bool,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E or (is_marlin and M <= 32):
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
    is_marlin: bool = False,
):
    E, _, N = w2_shape

    config = get_default_config(M, E, N, w1_shape[2], top_k, is_marlin)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, hidden_size = hidden_states.shape
    num_experts = w1.shape[0]
    topk = topk_ids.shape[1]
    routed_scaling_factor = 1.0 if routed_scaling_factor is None else routed_scaling_factor

    # The original path dispatches through sgl_kernel/Triton fused kernels. The
    # ZeroGPU image currently exposes a Torch/CUDA ABI that is incompatible with
    # the released sgl_kernel wheel, so this Space uses an eager fallback.
    source_hidden = hidden_states if not inplace else hidden_states.clone()
    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
        out_hidden_states.zero_()
    else:
        out_hidden_states = torch.zeros_like(hidden_states)

    for expert_idx in range(num_experts):
        route_mask = topk_ids == expert_idx
        if not bool(route_mask.any().item()):
            continue

        token_idx, route_idx = route_mask.nonzero(as_tuple=True)
        expert_input = source_hidden.index_select(0, token_idx)
        route_weight = topk_weights[token_idx, route_idx].to(dtype=expert_input.dtype).unsqueeze(-1)
        if apply_router_weight_on_input:
            expert_input = expert_input * route_weight

        gate_up = F.linear(expert_input, w1[expert_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        if activation == "silu":
            expert_hidden = F.silu(gate) * up
        elif activation == "gelu":
            expert_hidden = F.gelu(gate) * up
        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        expert_out = F.linear(expert_hidden, w2[expert_idx])
        if not apply_router_weight_on_input:
            expert_out = expert_out * route_weight
        if routed_scaling_factor != 1.0:
            expert_out = expert_out * routed_scaling_factor

        if no_combine:
            out_hidden_states[token_idx, route_idx] = expert_out
        else:
            out_hidden_states.index_add_(0, token_idx, expert_out)

    return out_hidden_states


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> None:

    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        True,
        activation,
        apply_router_weight_on_input,
        False,
        routed_scaling_factor,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,
        activation,
        apply_router_weight_on_input,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):

    if inplace:
        assert not no_combine, "no combine + inplace makes no sense"
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            routed_scaling_factor,
        )
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    no_combine: bool = False,
) -> torch.Tensor:

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=gating_output,
        top_k=topk,
        renormalize=renormalize,
    )
    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        no_combine=no_combine,
    )
