"""Implementation of Thinformer reported in the ICML submission."""

import torch
from torch import (
    nn,
    exp_,
    narrow,
    einsum,
    zeros,
    arange,
    empty,
    cat,
    logical_xor,
    amax,
    square,
    cummax,
    tensor,
    softmax,
    inf,
)
from torch import sqrt as torch_sqrt
from torch import log as torch_log
from torch import bool as torch_bool
from torch.linalg import vector_norm
from torch.nn.functional import scaled_dot_product_attention
from math import sqrt, log
try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError:
    flash_attn_func_cuda = None

def log4_largest_power_of_four(n: int) -> int:
    """Returns log_4 of the largest power of four less than or equal to n.

    Args:
        n: An integer input.

    Returns:
        Log base 4 of the largest power of four less than or equal to n.

    """
    return (n.bit_length() - 1) // 2


def exp_kernel(
    key: torch.Tensor, value: torch.Tensor, shift: torch.Tensor, 
    b_sqd: torch.Tensor
) -> torch.Tensor:
    """Returns tensor of weighted exponential kernel matrices
    kernel_mat[b,a,h]
        = exp(key[b,a,:,h] @ key[b,a,:,h].T - shift[b,0,h])
        * (value[b,a,:,h] @ value[b,a,:,h].T + b_sqd[b,0,h])

    Note: Assumes key has already been scaled appropriately by
    sqrt(softmax_temp)

    Args:
        key: tensor of shape [B, A, S, H, E]
        value: tensor of shape [B, A, S, H, D]
        shift: tensor broadcastable to shape [B, 1, H, 1, 1]
        b_sqd: tensor of shape [B, 1, H, 1, 1]

    Returns tensor of shape [B, A, H, S, S]

    """
    key_term = einsum("bashe,barhe->bahsr", key, key)
    key_term = key_term - shift
    key_term = exp_(key_term)
    val_term = einsum("bashd,barhd->bahsr", value, value)
    val_term = val_term + b_sqd
    val_term = val_term * key_term
    return val_term


def halve_K(
    K: torch.Tensor, delta: float = 0.5, symmetrize: bool = False
) -> torch.Tensor:
    """Batched kernel halving
    Returns kernel halving coreset indices of length S//2.
    Assumes S is even.

    Args:
        K: kernel matrix with shape [B,A,H,S,S]
        delta: kernel halving is run with failure probabilities delta_i = delta/n
        symmetrize: if False, returns initial coreset for each batch;
            if True, returns initial coreset or its complement uniformly at
            random for each batch

    Returns:
      tensor of shape [B,A,H,S//2]

    """
    B, A, H, S, _ = K.shape
    device = K.device
    num_points_in_coreset = S // 2
    log_multiplier = 0.5 + log(2 * S / delta)

    uniforms = empty(B, A, H, num_points_in_coreset, device=device).uniform_(
        -log_multiplier, log_multiplier
    )

    Kdiff = K[..., ::2] - K[..., 1::2]
    Kdiff = Kdiff[..., ::2, :] - Kdiff[..., 1::2, :]
    coreset_sum_diff = Kdiff[..., 0, :]
    rt_diag = torch_sqrt(Kdiff.diagonal(dim1=-2, dim2=-1))
    uniforms = uniforms * rt_diag * cummax(rt_diag, dim=-1).values

    swap_points = empty(B, A, H, num_points_in_coreset, device=device, dtype=torch_bool)
    swap_points[..., 0] = zeros(B, A, H, device=device, dtype=torch_bool)

    for i in range(1, num_points_in_coreset):
        swap_points_i = uniforms[..., i] <= coreset_sum_diff[..., i]
        swap_points[..., i] = swap_points_i
        coreset_sum_diff = coreset_sum_diff + (
            1.0 - 2.0 * swap_points_i.unsqueeze(-1)
        ) * Kdiff.select(-2, i)
    if symmetrize:
        swap_points = logical_xor(
            swap_points, empty(1, device=device, dtype=torch_bool).bernoulli_()
        )
    return swap_points + arange(0, S, step=2, device=device).expand_as(swap_points)


def halve(
    key: torch.Tensor,
    value: torch.Tensor,
    shift: torch.Tensor,
    b_sqd: torch.Tensor,
    halve_prob: float,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Weighted exponential kernel halving

    Returns a tensor containing, for each (b, a, h),
    half of the rows of X[b,a,:,h]. The half is selected by
    applying kernel halving to the sum of the weighted exponential
    kernel matrices for each (b,a,h).

    Args:
        key: tensor of shape [B, A, S, H, E]
        value: tensor of shape [B, A, S, H, D]
        shift: tensor of shape [B, 1, H, 1, 1] accepted by exp_kernel;
            typically represents the maximum squared key vector two norm
            to prevent exponential overflow via log-sum-exp trick
        halve_prob: halve_K is run with delta = halve_prob * S^2
        b_sqd: tensor of shape [B, 1, H, 1, 1] accepted by exp_kernel;
            typically represents the maximum squared value vector inf norm
        symmetrize: if False, returns initial coreset for each (b,a,h);
            if True, returns initial coreset or its complement uniformly at
            random for each (b,a,h)

    Returns:
        key coreset tensor of shape [B, A, S//2, H, E] and
        value coreset tensor of shape [B, A, S//2, H, D]

    """
    B, A, S, H, E = key.shape
    D = value.shape[-1]
    # Compute attention kernel matrix
    kernel_mat = exp_kernel(key, value, shift, b_sqd)
    # Select half of (key, value) pairs using kernel halving
    delta = halve_prob * S * S
    coreset = halve_K(kernel_mat, delta, symmetrize=symmetrize)
    # Return corresponding half of keys and values
    S_over_2 = S // 2
    coreset = coreset.transpose(2, 3).unsqueeze(-1)
    return (key.gather(2, coreset.expand(B, A, S_over_2, H, E)), 
        value.gather(2, coreset.expand(B, A, S_over_2, H, D)))


# @torch.compile(mode="reduce-overhead", fullgraph=True)
def _khcompress(
    key: torch.Tensor,
    value: torch.Tensor,
    four_to_g_plus_1: int,
    num_halving_rounds: int,
    halve_prob: float,
    symmetrize: bool,
) -> torch.Tensor:
    """KH-Compress(g)

    Produces a coreset of size 2^g sqrt(S). NOTE: S must be a power of 4.

    Args:
        key: Tensor of shape [B, S, H, E]
        value: Tensor of shape [B, S, H, D]
        four_to_g_plus_1: 4^{g+1} for the Compress oversampling parameter, g
        num_halving_rounds: number of Compress halving rounds
        halve_prob: parameters for halve()
        symmetrize: if True, randomly choose between coreset and its complement

    Returns:
        key coreset tensor of shape [B, S//2^m, H, E] and 
        value coreset tensor of shape [B, S//2^m, H, D]

    """
    # Compute kernel parameters:
    # Max squared key vector two norm
    shift = amax(square(key).sum(dim=3, keepdim=True), dim=1, keepdim=True)
    shift = shift.permute((0, 2, 1, 3))
    shift = shift.unsqueeze(1)
    # Max squared value vector inf norm
    b_sqd = vector_norm(
        value, dim=(1, 3), ord=inf, keepdim=True).unsqueeze(-1) ** 2

    # Combine keys and values
    B, _, H, E = key.shape
    D = value.shape[-1]

    # Execute Compress in bottom-up fashion by iteratively dividing input
    # into consecutive buckets of size bucket_size and halving each bucket.
    # Note that bucket_size grows by a factor of 2 on each halving round.
    bucket_size = four_to_g_plus_1
    for i in range(num_halving_rounds):
        key = key.view(B, -1, bucket_size, H, E)
        value = value.view(B, -1, bucket_size, H, D)
        key, value = halve(key, value, shift, b_sqd, halve_prob, symmetrize=symmetrize)
        bucket_size *= 2

    return key.view(B, -1, H, E), value.view(B, -1, H, D)


def khcompress(
    key: torch.Tensor,
    value: torch.Tensor,
    log4_n: int,
    g: int = 0,
    delta: float = 0.5,
) -> torch.Tensor:
    """KH-Compress(g)

    Computes a coreset of size n_out = min(n, 2^g sqrt(n)), where n is the 
    largest power of 4 less than or equal to S, the input sequence length.

    NOTE: This implementation first standard thins the input sequence to a size n,
    then computes a coreset of size 2^g sqrt(n) by calling the _khcompress function.

    Args:
        key: Tensor of shape [B, S, H, E]
        value: Tensor of shape [B, S, H, D]
        log4_n: log base 4 of n
        g: Oversampling factor, int >= 0
        delta: Kernel halving failure parameter, scalar in [0,1]

    Returns:
        key coreset Tensor of shape [B, n_out, H, E] and
        value coreset Tensor of shape [B, n_out, H, D]
        
    """
    S = key.shape[1]
    n = 4**log4_n
    four_to_g_plus_1 = int(4 ** (g + 1))
    
    # Standard thin keys and values down to sequence length n
    stride = S // n
    n_times_stride = n * stride
    key = key[:, 0 : n_times_stride : stride]
    value = value[:, 0 : n_times_stride : stride]

    # If n <= 2^g sqrt(n), no further thinning is needed
    # Since n is a power of four, equivalently, check if 
    # n < 4^{g+1}
    if n < four_to_g_plus_1:
        return key, value

    # Compute number of Compress halving rounds
    num_halving_rounds = int(log4_n - g)
    
    # Compute KH-Compress base halving probability parameter
    halve_prob = delta / four_to_g_plus_1 / num_halving_rounds / n

    # NOTE: To prevent overwriting, we call 
    # torch.compiler.cudagraph_mark_step_begin() 
    # torch.compiler.cudagraph_mark_step_begin()
    key, value = _khcompress(
        key, value,
        four_to_g_plus_1,
        num_halving_rounds,
        halve_prob,
        symmetrize=True)
    return key, value

def add_self_attentions(attn1, lse1, attn2, lse2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
    output:
        - attn
        = (attn1 * exp(lse1) + attn2 * exp(lse2)) / (exp(lse1) + exp(lse2))
        = (attn1 + attn2 * exp(lse2 - lse1)) / (1 + exp(lse2-lse1))
        = attn1 * c + attn2 * (1-c), where c=1/(1 + exp(lse2-lse1)),
        - lse 
        = log(exp(lse1) + exp(lse2)) 
        = log(exp(lse1) * (1 + exp(lse2 - lse1))) 
        = lse1 + log(1 + exp(lse2 - lse1)) = lse1 - log(c)
    """
    # import pdb; pdb.set_trace()
    c = (1 / (1 + (lse2 - lse1).exp())).to(dtype=attn1.dtype)
    attn = c * attn1 + (1-c) * attn2
    lse = lse1 - (c + torch.finfo(lse1.dtype).eps).log()
    return attn, lse


# @torch.compile(mode="reduce-overhead", fullgraph=True)
def full_forward(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implements the multihead softmax attention assuming
    query and key have already been scaled appropriately
    (e.g., that key has been multiplied by
    softmax_temp = self.scale or 1 / sqrt(E))

    Arguments:
    ---------
        query: (B, T, H, E) The tensor containing the query
        key: (B, S, H, E) The tensor containing the key
        value: (B, S, H, D) The tensor containing the value
        causal: bool, optional (default=False) Whether to apply causal masking

    Returns:
        output: (B, T, H, D) The tensor containing the output
        lse: (B, T, S, H) Logsumexp of QK

    """
    QK = einsum("bthe,bshe->bhts", query, key)
    
    if causal:
        # Create causal mask where True means "masked out"
        mask = torch.triu(torch.ones(query.shape[1], key.shape[1], dtype=torch.bool, device=query.device), diagonal=1)
        # Apply mask by setting masked positions to -inf before softmax
        QK = QK.masked_fill(mask, float('-inf'))
    
    A = softmax(QK, dim=-1)
    output = einsum("bhts,bshd->bthd", A, value)
    lse = torch.logsumexp(QK, dim=-1, keepdim=True)

    return output, lse


class ThinformerAttention(nn.Module):
    """Implementation of Thinformer Attention module."""

    def __init__(
        self,
        g: int = 2,
        scale: float | None = None,
        use_torch_spda: bool = False,
        **kwargs: dict,
    ):
        """Initialize the ThinformerAttention module.

        Args:
            g (int): oversampling parameter, a nonnegative integer
            scale (float): scale for dot-product attention. 
              If `None`, scale is chosen as 1/sqrt(key.shape[-1]) in forward.
            use_torch_spda (bool): if True, use 
              torch.nn.functional.scaled_dot_product_attention,
              which automatically optimizes the attention computation for GPUs, 
              for the final attention computation.
            kwargs: placeholder for other arguments

        """
        super().__init__()
        self.g = g
        self.scale = scale
        self.use_torch_spda = use_torch_spda

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """The forward pass of the ThinformerAttention module comprises
        two steps:
        1. Select a subset of the key and value pairs using KH-Compress.
           The size of this subset is 2^g sqrt(S), where S is the number
           of key-value pairs and g is the oversampling parameter.
        2. For each query vector, only attend to the selected key and value
           pairs. The computational complexity of this step is 2^g sqrt(S) * T.

        NOTE: the second step can be computed using a generic scaled dot-
        product attention implementation or an optimized implementation,
        e.g., torch.nn.functional.scaled_dot_product_attention.

        Args:
            query: (B, T, H, E) The tensor containing the queries
            key: (B, S, H, E) The tensor containing the keys
            value: (B, S, H, D) The tensor containing the values

        Returns:
            output: (B, T, H, D) The tensor approximating
              softmax(query * tranpose(key) * softmax_temp) * value
              for softmax_temp = self.scale or 1 / sqrt(E)
            A: (B, T, H, S) The attention weights approximating
              softmax(query * tranpose(key) * softmax_temp)

        """
        # Compute squareroot of softmax temperature 
        # for the attention that A = softmax(query * key * softmax_temp)
        E = key.shape[-1]
        sqrt_softmax_temp = sqrt(self.scale or 1 / sqrt(E))
        S = key.shape[1]
        log4_n = log4_largest_power_of_four(S)
        key, value = khcompress(key * sqrt_softmax_temp, value, 
                                log4_n, g=self.g)
        key = key * sqrt_softmax_temp

        if self.use_torch_spda:
            # NOTE: we set scale=1.0 since we have already scaled the query and key
            out = scaled_dot_product_attention(
                query=query.transpose(1, 2),
                key=key.transpose(1, 2),
                value=value.transpose(1, 2),
                dropout_p=0.0,
                scale=1.0,
                is_causal=False,
            ).transpose(1, 2)
            # NOTE: returning the attention weights is not supported in the current
            # implementation of torch.nn.functional.scaled_dot_product_attention
            return out, None
        else:
            return full_forward(query, key, value)

class ThinformerHyperAttention(nn.Module):
    """Implementation of Thinformer Attention module."""

    def __init__(
        self,
        g: int = 2,
        scale: float | None = None,
        use_torch_spda: bool = False,
        min_seq_len=4096,
        **kwargs: dict,
    ):
        """Initialize the ThinformerAttention module.

        Args:
            g (int): oversampling parameter, a nonnegative integer
            scale (float): scale for dot-product attention. 
              If `None`, scale is chosen as 1/sqrt(key.shape[-1]) in forward.
            use_torch_spda (bool): if True, use 
              torch.nn.functional.scaled_dot_product_attention,
              which automatically optimizes the attention computation for GPUs, 
              for the final attention computation.
            kwargs: placeholder for other arguments

        """
        super().__init__()
        self.g = g
        self.scale = scale
        self.use_torch_spda = use_torch_spda
        self.min_seq_len = min_seq_len

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale=None,
        causal=False,
        return_lse=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inputs:
            query: (B, H, T, E) The tensor containing the queries
            key: (B, H, S, E) The tensor containing the keys
            value: (B, H, S, D) The tensor containing the values
            scale: float, optional, default=None
            causal: bool, optional, default=False
        """

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape

        # Without causal masking
        if not causal: 
            attn, lse = self.forward_no_causal_mask(query, key, value)

        # With causal masking
        else:
            if n_key <= self.min_seq_len:
                # NOTE: key, value have shape [B, S, H, D]
                # query has shape [B, T, H, E]
                attn, lse = self.flash_full_forward(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), causal=True)
                # NOTE: attn has shape [B, T, H, S]
                # lse has shape [B, T, S, H]
            else:
            
                # If n_query is odd we pad inputs by adding all-zero rows
                if n_query % 2:
                    query = torch.nn.functional.pad(query, (0,0,0,1), mode='constant',value=0.)
                    key = torch.nn.functional.pad(key, (0,0,0,1), mode='constant',value=0.)
                    value = torch.nn.functional.pad(value, (0,0,0,1), mode='constant',value=0.)

                # NOTE: should be using transpose instead of view?
                q_bd = query.view(batch_size, query.shape[2]//2, 2*n_heads, query.shape[-1]).transpose(1, 2)
                k_bd = key.view(batch_size, key.shape[2]//2, 2*n_heads, key.shape[-1]).transpose(1, 2)
                v_bd = value.view(batch_size, key.shape[2]//2, 2*n_heads, value.shape[-1]).transpose(1, 2)
        
                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)
                
                if attn_bd.shape[2] not in attn_bd.stride():
                    attn_bd = attn_bd.contiguous()
                if False:
                    attn_bd = attn_bd.view(batch_size, n_heads, -1, dim)
                else:
                    attn_bd = attn_bd.reshape(batch_size, n_heads, -1, dim)

                if lse_bd.shape[2] not in lse_bd.stride():
                    lse_bd = lse_bd.contiguous()
                lse_bd = lse_bd.view(batch_size, n_heads, -1, 1)

                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2]//2:, :].transpose(1, 2),
                    key[:, :, :key.shape[2]//2, :].transpose(1, 2), 
                    value[:, :, :key.shape[2]//2, :].transpose(1, 2))
                attn_unmasked = attn_unmasked.transpose(1, 2)
                # lse_unmasked = lse_unmasked.transpose(1, 2)

                attn_up, lse_up = attn_bd[:,:,:query.shape[2]//2,:], lse_bd[:,:,:query.shape[2]//2,:]
                attn_down, lse_down = add_self_attentions(
                    attn_bd[:,:,query.shape[2]//2:,:],
                    lse_bd[:,:,query.shape[2]//2:,:],
                    attn_unmasked,
                    lse_unmasked)

                attn = torch.cat((attn_up, attn_down), dim=-2)
                lse = torch.cat((lse_up, lse_down), dim=-2)

                # If n_query was odd exclude the last rows
                if n_query % 2:
                    attn = attn[:,:,:-1,:]
                    lse = lse[:,:,:-1,:]

        if not return_lse:
            return attn
        else:
            return attn, lse


    def flash_full_forward(self,
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flash_attn_func_cuda is None:
            raise ImportError("Please install flash_attn (pip install flash-attn --no-build-isolation)")
        out, lse, _ = flash_attn_func_cuda(
            query.transpose(1,2), key.transpose(1,2), value.transpose(1,2),
            softmax_scale=self.scale, causal=causal, return_attn_probs=True)
        out = out.transpose(1,2)
        lse = lse.unsqueeze(-1)
        return out, lse

    def forward_no_causal_mask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the ThinformerAttention module comprises
        two steps:
        1. Select a subset of the key and value pairs using KH-Compress.
           The size of this subset is 2^g sqrt(S), where S is the number
           of key-value pairs and g is the oversampling parameter.
        2. For each query vector, only attend to the selected key and value
           pairs. The computational complexity of this step is 2^g sqrt(S) * T.

        NOTE: the second step can be computed using a generic scaled dot-
        product attention implementation or an optimized implementation,
        e.g., torch.nn.functional.scaled_dot_product_attention.

        Args:
            query: (B, T, H, E) The tensor containing the queries
            key: (B, S, H, E) The tensor containing the keys
            value: (B, S, H, D) The tensor containing the values

        Returns:
            output: (B, T, H, D) The tensor approximating
              softmax(query * tranpose(key) * softmax_temp) * value
              for softmax_temp = self.scale or 1 / sqrt(E)
            A: (B, T, H, S) The attention weights approximating
              softmax(query * tranpose(key) * softmax_temp)

        """
        # Compute squareroot of softmax temperature 
        # for the attention that A = softmax(query * key * softmax_temp)
        E = key.shape[-1]
        sqrt_softmax_temp = sqrt(self.scale or 1 / sqrt(E))
        S = key.shape[1]
        log4_n = log4_largest_power_of_four(S)
        key, value = khcompress(key * sqrt_softmax_temp, value, 
                                log4_n, g=self.g)
        key = key * sqrt_softmax_temp

        if self.use_torch_spda:
            raise NotImplemented("Not completed") # type: ignore
        else:
            return self.flash_full_forward(query, key, value)