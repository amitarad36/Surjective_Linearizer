from models.spnn import SPNN
from linearizer.linear_network import TimeDependentLoRALinearLayer


def get_g(img_ch, img_size, latent_size):
    return SPNN(img_ch=img_ch, num_classes=latent_size, hidden=128, scale_bound=2.0, img_size=img_size)


def get_linear_network(latent_size, lora_rank=8, t_size=128):
    return TimeDependentLoRALinearLayer(latent_size, lora_rank, t_size)
