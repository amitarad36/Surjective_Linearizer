from typing_extensions import override

from linearizer.base import Linearizer, G
from linearizer.linear_network import OneStepLinearModule


class OneStepLinearizer(Linearizer):
    """Linearizer variant that uses SPNN's surjective encoder with pseudo-inverse decoding.

    Overrides the encode/decode calls to use .pinv() instead of .inverse(),
    making it compatible with surjective encoders (SPNN) that have a
    Moore-Penrose pseudo-inverse rather than an exact inverse.
    """

    def __init__(self, gx: G, linear_network: OneStepLinearModule, gy: G = None):
        super().__init__(gx=gx, linear_network=linear_network, gy=gy)

    @override
    def gx(self, x, **kwargs):
        """Encode x through the data encoder gx."""
        return self.net_gx(x)

    @override
    def gy(self, y, **kwargs):
        """Encode y through the noise encoder gy."""
        return self.net_gy(y)

    @override
    def gx_inverse(self, g_x, **kwargs):
        """Decode from gx latent space using the pseudo-inverse g†."""
        return self.net_gx.pinv(g_x)

    @override
    def gy_inverse(self, g_y, **kwargs):
        """Decode from gy latent space using the pseudo-inverse g†."""
        return self.net_gy.pinv(g_y)
