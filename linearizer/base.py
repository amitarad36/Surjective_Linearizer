from abc import abstractmethod
from typing import final

import torch.nn


class G(torch.nn.Module):
    """Abstract base class for an encoder.

    Maps between image space and the induced latent space.
    Subclasses must implement forward (encode) and inverse (decode).
    """

    def __init__(self, in_ch, image_resolution):
        super().__init__()
        self.dim = in_ch * image_resolution ** 2

    @abstractmethod
    def forward(self, x, **kwargs):
        """Encode x from image space to latent space."""
        pass

    @abstractmethod
    def inverse(self, z, **kwargs):
        """Decode z from latent space back to image space."""
        pass


class LinearModule(torch.nn.Module):
    """Abstract base class for the linear operator A in the induced latent space."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x, **kwargs):
        """Apply the linear map A to x."""
        pass

    # optional
    def inverse(self, z, **kwargs):
        """Apply the inverse of A to z. Optional — not all linear operators are invertible."""
        pass


class Linearizer(torch.nn.Module):
    """Composes two encoders gx, gy and a linear operator A into a full model.

    The forward pass implements: f(x) = gy_inverse( A( gx(x) ) )
    The inverse pass implements: f_inv(y) = gx_inverse( A_inv( gy(y) ) )

    Both forward and inverse are @final — subclasses override the individual
    encode/decode/A methods, not the composition itself.
    """

    def __init__(self, gx: G, linear_network: LinearModule, gy: G = None):
        super().__init__()
        if gy is None:
            gy = gx
        self.net_gx = gx
        self.net_gy = gy
        self.linear_network = linear_network

    def gx(self, x, **kwargs):
        """Encode x using the input-space encoder gx."""
        return self.net_gx(x, **kwargs)

    def gy(self, y, **kwargs):
        """Encode y using the output-space encoder gy."""
        return self.net_gy(y, **kwargs)

    def gx_inverse(self, g_x, **kwargs):
        """Decode from gx latent space back to image space."""
        return self.net_gx.inverse(g_x, **kwargs)

    def gy_inverse(self, g_y, **kwargs):
        """Decode from gy latent space back to image space."""
        return self.net_gy.inverse(g_y, **kwargs)

    def A(self, g_x, **kwargs):
        """Apply the linear operator A in the induced latent space."""
        return self.linear_network(g_x, **kwargs)

    # optional
    def A_inverse(self, g_y, **kwargs):
        """Apply the inverse of A. Optional — only valid if A is invertible."""
        return self.linear_network.inverse(g_y, **kwargs)

    @final
    def forward(self, x, **kwargs):
        """Full forward pass: gy_inverse( A( gx(x) ) )."""
        g_x = self.gx(x, **kwargs)
        g_y = self.A(g_x, **kwargs)
        y_pred = self.gy_inverse(g_y, **kwargs)
        return y_pred

    @final
    def inverse(self, y, **kwargs):
        """Full inverse pass: gx_inverse( A_inv( gy(y) ) )."""
        g_y = self.gy(y, **kwargs)
        g_x = self.A_inverse(g_y, **kwargs)
        x_pred = self.gx_inverse(g_x, **kwargs)
        return x_pred
