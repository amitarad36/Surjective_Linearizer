from typing_extensions import override

from linearizer.base import Linearizer, G
from linearizer.linear_network import OneStepLinearModule


class OneStepLinearizer(Linearizer):
    def __init__(self, gx: G, linear_network: OneStepLinearModule, gy: G = None):
        super().__init__(gx=gx, linear_network=linear_network, gy=gy)

    @override
    def gx(self, x, **kwargs):
        return self.net_gx(x)

    @override
    def gy(self, y, **kwargs):
        return self.net_gy(y)

    @override
    def gx_inverse(self, g_x, **kwargs):
        return self.net_gx.pinv(g_x)

    @override
    def gy_inverse(self, g_y, **kwargs):
        return self.net_gy.pinv(g_y)
