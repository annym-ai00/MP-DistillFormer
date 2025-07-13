import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

def b_splines(x, grid, k):
    """Generate B-splines.

    Args:
        x (torch.Tensor): [batch_size_size, in_dim]
        grid (torch.Tensor): [in_dim, grid_size + 2*k + 1]
        k (int): degree of B-splines

    Returns:
        (torch.Tenosr): [batch_size, in_dim, grid_size + k]
    """

    x = x.unsqueeze(-1)

    value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    for p in range(1, k + 1):
        value = (x - grid[:, : -(p + 1)]) / (
            grid[:, p:-1] - grid[:, : -(p + 1)]
        ) * value[:, :, :-1] + (grid[:, p + 1 :] - x) / (
            grid[:, p + 1 :] - grid[:, 1:(-p)]
        ) * value[
            :, :, 1:
        ]

    return value


def curve2coeff(x, y, grid, k, eps=1e-8):
    """Calculate coefficients of B-splines.

    Args:
        x_eval (torch.Tensor): [batch_size, in_dim]
        y_eval (torch.Tensor): [batch_size, in_dim, out_dim]
        grid (torch.Tensor): [in_dim, grid_size + 2*k + 1]
        k (int): degree of B-splines

    Returns:
        (torch.Tensor): [out_dim, in_dim, grid_size + k]
    """

    splines = b_splines(x, grid, k).transpose(0, 1)

    device = splines.device
    if device != "cpu":
        splines = splines.cpu()
        y = y.cpu()

    # [in_dim, batch_size, grid_size + k] @ [in_dim, grid_size + k, out_dim] - [in_dim, batch_size, out_dim] = 0
    value = torch.linalg.lstsq(splines, y.transpose(0, 1)).solution

    value = value.to(device)

    # [in_dim, grid_size + k, out_dim] -> [out_dim, in_dim, grid_size + k]
    value = value.permute(2, 0, 1)

    return value

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        grid_size=5,
        k=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        scale_spline=None,
        base_fun=torch.nn.SiLU(),
        grid_eps=0.02,
        grid_range=[-1, +1],
        bias=False,
        bias_trainable=True,
        scale_spline_trainable=True,
        scale_base_trainable=True,
        device="cpu",
    ):
        torch.nn.Module.__init__(self)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = k
        self.base_fun = base_fun
        self.grid_eps = grid_eps
        self.size = in_dim * out_dim
        self.grid_size = grid_size
        self.device = device

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-k, grid_size + k + 1, device=device) * step + grid_range[0]
        ).repeat(self.in_dim, 1)
        self.register_buffer("grid", grid)  # grid [in_dim, grid_size + 2*k + 1]

        if scale_spline is not None:
            self.scale_spline = torch.nn.Parameter(
                torch.full(
                    (
                        out_dim,
                        in_dim,
                    ),
                    fill_value=scale_spline,
                    device=device,
                ),
                requires_grad=scale_spline_trainable,
            )
        else:
            self.register_buffer("scale_spline", torch.tensor([1.0], device=device))

        noise = (
            (torch.rand(grid_size + 1, in_dim, out_dim, device=device) - 1 / 2)
            * noise_scale
            / self.grid_size  # TODO: (np.sqrt(in_dim) * np.sqrt(grid_size)) ?
        )
        self.coeff = torch.nn.Parameter(
            curve2coeff(
                x=self.grid.T[k:-k],  # [grid_size + 1, in_dim]
                y=noise,  # [grid_size + 1, in_dim, out_dim]
                grid=self.grid,
                k=k,
            ).contiguous()
        )  # [out_dim, in_dim, grid_size + k]

        self.scale_base = torch.nn.Parameter(
            (
                1 / (in_dim**0.5)
                + (torch.randn(self.out_dim, self.in_dim, device=device) * 2 - 1)
                * noise_scale_base
            ),
            requires_grad=scale_base_trainable,
        )

        if bias is True:
            self.bias = torch.nn.Parameter(
                torch.rand(out_dim), requires_grad=bias_trainable
            )
        else:
            self.bias = None

    def forward(self, x):
        shape = x.shape[:-1]
        x = x.view(-1, self.in_dim)
        # x [batch, in_dim]

        splines = b_splines(x, self.grid, self.k)  # [batch_size, in_dim, grid_size + k]

        ####### Efficient KAN forward #########

        batch_size = x.shape[0]
        y_b = F.linear(self.base_fun(x), self.scale_base)
        # [batch_size, in_dim] @ [out_dim, in_dim]^T = [batch_size, out_dim]

        y_spline = F.linear(
            splines.view(batch_size, -1),
            (self.coeff * self.scale_spline.unsqueeze(-1)).view(self.out_dim, -1),
        )  # [batch_size, in_dim * (grid_size + k)] @ [out_dim, in_dim * (grid_size + k)]^T = [batch, out_dim]

        y = y_b + y_spline

        if self.bias is not None:
            y = y + self.bias

        y = y.view(*shape, self.out_dim)

        return y

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.coeff.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        batch_size = x.shape[0]

        # [batch_size, in_dim, grid_size + k]
        splines = b_splines(x, self.grid, self.k)

        # TODO: Is this correct?
        orig_coeff = self.coeff * self.scale_spline.unsqueeze(-1)

        # [in_dim, batch_size, grid_size + k] @ [in_dim, grid_size + k, out_dim] = [in_dim, batch_size, out_dim] -> [batch_size, in_dim, out_dim]
        y = (splines.permute(1, 0, 2) @ orig_coeff.permute(1, 2, 0)).permute(1, 0, 2)

        # sort activations in ascending order for each input dimension
        x_sorted = torch.sort(x, dim=0)[0]  # [batch_size, in_dim]

        # sample grid points from sorted activations
        # [grid_size + 1, in_dim]
        grid_adaptive = x_sorted[
            torch.linspace(
                0,
                batch_size - 1,
                self.grid_size + 1,
                dtype=torch.int64,
                device=self.device,
            )
        ]

        # find max and min values for each input dimension + margin and sample grid points uniformly from this range
        uniform_step = (
            x_sorted[-1] - x_sorted[0] + 2 * margin
        ) / self.grid_size  # [in_dim]
        # [grid_size + 1, in_dim]
        grid_uniform = (
            torch.arange(self.grid_size + 1, device=self.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # combine adaptive and uniform grid points
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        # grid is [grid_size + 1, in_dim] but self.grid is [in_dim, grid_size + 2*k + 1], so we need to expand grid
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.k, 0, -1, device=self.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.k + 1, device=self.device).unsqueeze(1),
            ],
            dim=0,
        )

        # set new grid, transpose [grid_size + 2*k + 1, in_dim] -> [in_dim, grid_size + 2*k + 1]
        self.grid.copy_(grid.T)
        self.coeff.data.copy_(curve2coeff(x, y, self.grid, self.k))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding = padding, stride = 1, bias = False)
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, proj_kernel, padding = padding, stride = kv_proj_stride, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)

        return self.to_out(out)

    
class ConvTB(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head = 64, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, proj_kernel = proj_kernel, kv_proj_stride = kv_proj_stride, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_mult, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SMoBlock(nn.Module):

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                      1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = out + x
        return out

class ConvTE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            LayerNorm(out_channels)
        )

    def forward(self, x):
        return self.block(x)
    
class  TeSMo_KAN(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        
        s1_emb_dim=32,
        s1_emb_kernel=3,
        s1_emb_stride=2,
        s1_proj_kernel=3,
        s1_kv_proj_stride=2,
        s1_heads=1,
        s1_depth=1,
        s1_mlp_mult=4,
        
        s2_emb_dim=64,
        s2_emb_kernel=3,
        s2_emb_stride=2,
        s2_proj_kernel=3,
        s2_kv_proj_stride=2,
        s2_heads=3,
        s2_depth=2,
        s2_mlp_mult=4,
        
        s3_emb_dim=192,
        s3_emb_kernel=3,
        s3_emb_stride=2,
        s3_proj_kernel=3,
        s3_kv_proj_stride=2,
        s3_heads=6,
        s3_depth=10,
        s3_mlp_mult=4,
        
        dropout = 0.1,
        expansion = 1
    ):
        super().__init__()
        kwargs = dict(locals())
        
        self.s1_layers = nn.Sequential(      
            ConvTE(3, s1_emb_dim, kernel_size=s1_emb_kernel, stride=s1_emb_stride),
            SMoBlock(s1_emb_dim, s1_emb_dim, 1, expansion),
            ConvTB(dim=s1_emb_dim, proj_kernel=s1_proj_kernel, kv_proj_stride=s1_kv_proj_stride, depth=s1_depth, heads=s1_heads, mlp_mult=s1_mlp_mult, dropout=dropout),
        )

        self.s2_layers = nn.Sequential(
            ConvTE(s1_emb_dim, s2_emb_dim, kernel_size=s2_emb_kernel, stride=s2_emb_stride),
            SMoBlock(s2_emb_dim, s2_emb_dim, 1, expansion),
            ConvTB(dim=s2_emb_dim, proj_kernel=s2_proj_kernel, kv_proj_stride=s2_kv_proj_stride, depth=s2_depth, heads=s2_heads, mlp_mult=s2_mlp_mult, dropout=dropout),
        )

        self.s3_layers = nn.Sequential(
            ConvTE(s2_emb_dim, s3_emb_dim, kernel_size=s3_emb_kernel, stride=s3_emb_stride),
            SMoBlock(s3_emb_dim, s3_emb_dim, 1, expansion),
            ConvTB(dim=s3_emb_dim, proj_kernel=s3_proj_kernel, kv_proj_stride=s3_kv_proj_stride, depth=s3_depth, heads=s3_heads, mlp_mult=s3_mlp_mult, dropout=dropout),   
        )
        
        self.to_logits = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            KANLinear(s3_emb_dim, num_classes)
        )

    def forward(self, x):
        x = self.s1_layers(x)
        x = self.s2_layers(x)     
        x = self.s3_layers(x)
 
        return self.to_logits(x)