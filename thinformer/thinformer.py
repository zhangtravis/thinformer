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
    log,
    tensor,
    softmax,
    inf,
)
from torch.linalg import vector_norm
from torch import sqrt as torch_sqrt
from torch import bool as torch_bool
from torch.nn.functional import scaled_dot_product_attention
from math import sqrt


def largest_power_of_four(n: int) -> int:
    """Returns the largest power of four less than or equal to n.

    Args:
        n: An integer input.

    Returns:
        The largest power of four less than or equal to n.

    """
    return 4 ** ((n.bit_length() - 1) // 2)


def exp_kernel(
    key: torch.Tensor, value: torch.Tensor, shift: torch.Tensor, b_sqd: torch.Tensor
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
    log_multiplier = 0.5 + log(2 * tensor(S) / delta)

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
    X: torch.Tensor,
    shift: torch.Tensor,
    halve_prob: float,
    b_sqd: torch.Tensor,
    symmetrize: bool = True,
) -> torch.Tensor:
    """Weighted exponential kernel halving

    Returns a tensor containing, for each (b, a, h),
    half of the rows of X[b,a,:,h]. The half is selected by
    applying kernel halving to the sum of the weighted exponential
    kernel matrices for each (b,a,h).

    Args:
        X: tensor of shape [B, A, S, H, 2*E]
        shift: tensor of shape [B, 1, H, 1, 1] accepted by exp_kernel;
            max K row sum for X = concat(K, V), to prevent overflow via log-sum-exp trick
        halve_prob: halve_K is run with delta = halve_prob * S^2
        symmetrize: if False, returns initial coreset for each (b,a,h);
            if True, returns initial coreset or its complement uniformly at
            random for each (b,a,h)
        b_sqd: tensor of shape [B, 1, H, 1, 1];
            max infnorm of V, for X = concat(K, V)

    Assumes key and value components of X have the same shape

    Returns tensor of shape [B, A, S//2, H, 2*E]

    """
    feature_dim = 4
    B, A, S, H, two_E = X.shape
    E = two_E // 2
    key = narrow(X, feature_dim, 0, E)
    value = narrow(X, feature_dim, E, E)
    kernel_mat = exp_kernel(key, value, shift, b_sqd)
    delta = halve_prob * S * S
    coreset = halve_K(kernel_mat, delta, symmetrize=symmetrize)
    return X.gather(
        2, coreset.transpose(2, 3).unsqueeze(-1).expand(B, A, S // 2, H, two_E)
    )


@torch.compile(mode="reduce-overhead", fullgraph=True)
def _khcompress(
    X: torch.Tensor,
    four_to_g_plus_1: int,
    m: int,
    shift: torch.Tensor,
    halve_prob: float,
    symmetrize: bool,
    final_halve_prob: float,
) -> torch.Tensor:
    """KH-Compress(g)

    Produces a coreset of size 2^g sqrt(S). NOTE: S must be a power of 4.

    Args:
        X: tensor of shape [B, S, H, E_plus_D]
        four_to_g_plus_1: 4^{g+1} for the Compress oversampling parameter, g
        m: number of thinning steps
        shift: parameters for kernel function
        halve_prob: parameters for halve function
        symmetrize: if True, randomly choose between coreset and its complement
        final_halve_prob: halve_prob for final halving round

    Returns:
        tensor of shape [B, S//2^m, H, E_plus_D]

    """
    B, S, H, E_plus_D = X.shape
    # max infnorm of V, for X = concat(K, V)
    b_sqd = (
        vector_norm(
            X[:, :, :, E_plus_D // 2 :], dim=(1, 3), ord=inf, keepdim=True
        ).unsqueeze(-1)
        ** 2
    )

    for i in range(m):
        bucket_size = 2**i * four_to_g_plus_1
        X = X.view(B, -1, bucket_size, H, E_plus_D)
        X = halve(X, shift, halve_prob, b_sqd, symmetrize=symmetrize)

    X = halve(
        X.view(B, 1, -1, H, E_plus_D),
        shift,
        final_halve_prob,
        b_sqd,
        symmetrize=False,
    )
    return X.view(B, -1, H, E_plus_D)


def khcompress(
    X: torch.Tensor,
    log2_n: int,
    g: int = 0,
    shift: torch.Tensor = 0,
    delta: float = 0.5,
) -> torch.Tensor:
    """KH-Compress(g)

    Computes a coreset of size 2^g sqrt(n), where n is the largest power of 4
    less than or equal to S, the input sequence length.

    NOTE: This implementation first thins the input sequence to a size n,
    then computes a coreset of size 2^g sqrt(n) by calling the _khcompress function.

    Args:
        X: Tensor of shape [B, S, H, E+D]
        log2_n: log base 2 of n
        g: Oversampling factor, int >= 0
        shift: Tensor broadcastable to shape [B, 1, H, 1, 1] accepted by exp_kernel
        delta: Kernel halving failure parameter, scalar in [0,1]

    Returns:
        Tensor of shape [B, 2^g sqrt(n), H, E_plus_D]

    """
    log2_num_bins = 2  # num_bins = 4
    B, S, H, E_plus_D = X.shape
    log2_bin_size = log2_n - log2_num_bins
    bin_size = 2**log2_bin_size
    n = 2**log2_n
    stride = S // n
    X = X[:, 0 : n * stride : stride]
    four_to_g_plus_1 = 4 ** (g + 1)
    if bin_size < four_to_g_plus_1:
        return X

    log2_sqrt_bin_size_minus_g = log2_bin_size // 2 - g
    halve_prob = delta / four_to_g_plus_1 / log2_sqrt_bin_size_minus_g / n
    m = log2_num_bins // 2
    thin_frac = m / (m + (2**m) * log2_sqrt_bin_size_minus_g)

    halve_prob *= 1 - thin_frac
    thin_delta = delta * thin_frac

    # NOTE: thin_S = 2^g sqrt(num_bins) sqrt(n)
    #              = 2^(g+1) sqrt(n)  (when num_bins = 4)
    thin_n_sqd = four_to_g_plus_1 * n
    # NOTE: To prevent overwriting, we call torch.compiler.cudagraph_mark_step_begin() before each function invocation.
    torch.compiler.cudagraph_mark_step_begin()
    X = _khcompress(
        X,
        int(four_to_g_plus_1),
        int(log2_sqrt_bin_size_minus_g),
        shift,
        halve_prob,
        symmetrize=True,
        final_halve_prob=thin_delta / thin_n_sqd,
    )
    return X


@torch.compile(mode="reduce-overhead", fullgraph=True)
def full_forward(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, None]:
    """Implements the multihead softmax attention assuming
    query and key have already been scaled appropriately
    (e.g., that key has been multiplied by
    softmax_temp = self.scale or 1 / sqrt(E))

    Arguments:
    ---------
        query: (B, T, H, E) The tensor containing the query
        key: (B, S, H, E) The tensor containing the key
        value: (B, S, H, D) The tensor containing the value

    Returns:
        output: (B, T, H, D) The tensor containing the output
        A: (B, T, H, S) The attention weights

    """
    QK = einsum("bthe,bshe->bhts", query, key)
    A = softmax(QK, dim=-1)
    output = einsum("bhts,bshd->bthd", A, value)

    return output, A


class ThinformerAttention(nn.Module):
    """Implementation of Thinformer Attention module."""

    def __init__(
        self,
        g: int = 1,
        scale: float | None = None,
        use_torch_spda: bool = False,
        **kwargs: dict,
    ):
        """Initialize the ThinformerAttention module.

        Args:
            g (int): oversampling parameter, a nonnegative integer
            scale (float): scale for dot-product attention. If `None`, the scale is set to 1/sqrt(E).
            use_torch_spda (bool): if True, use torch.nn.functional.scaled_dot_product_attention,
                which automatically optimizes the attention computation for GPUs, for the
                final attention computation.
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
            query: (B, T, H, E) The tensor containing the query
            key: (B, S, H, E) The tensor containing the key
            value: (B, S, H, D) The tensor containing the value

        Returns:
            output: (B, T, H, D) The tensor containing the output
            A: (B, T, H, S) The attention weights

        """
        E = key.shape[-1]
        sqrt_softmax_temp = sqrt(self.scale or 1 / sqrt(E))
        X = cat((key * sqrt_softmax_temp, value), dim=3)
        # max K row sum for X = concat(K, V), to prevent overflow via log-sum-exp trick
        shift = amax(square(X[..., :E]).sum(dim=3, keepdim=True), dim=1, keepdim=True)
        shift = shift.permute((0, 2, 1, 3))
        shift = shift.unsqueeze(1)
        n = largest_power_of_four(X.shape[1])
        X = khcompress(X, log2_n=(n.bit_length() - 1), g=self.g, shift=shift)

        E = key.shape[3]
        D = value.shape[3]
        key = narrow(X, 3, 0, E)
        value = narrow(X, 3, E, D)
        key *= sqrt_softmax_temp

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
