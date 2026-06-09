"""Compare inference time: Surjective Linearizer (SPNN) vs Original Linearizer (InverseUnet)."""
import sys
import argparse
import time
import torch
import wandb

SURJ_PATH = '/home/amit.arad/work/Surjective_Linearizer'
LIN_PATH  = '/home/amit.arad/work/Linearizer-main'

# SURJ_PATH must be inserted last so it ends up at index 0 — otherwise
# linearizer.py from LIN_PATH shadows our linearizer/ package.
sys.path.insert(0, LIN_PATH)
sys.path.insert(0, f'{LIN_PATH}/one_step')
sys.path.insert(0, f'{LIN_PATH}/common')
sys.path.insert(0, SURJ_PATH)


def bench(fn, warmup=3, runs=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs


def build_surj_model(checkpoint, latent_size, lora_rank, img_size, in_ch, device):
    from linearizer.one_step import OneStepLinearizer
    from training.flow_matching import FlowMatcher
    from utils.model_utils import get_g, get_linear_network

    g = get_g(img_ch=in_ch, img_size=img_size, latent_size=latent_size)
    linear_network = get_linear_network(latent_size=latent_size, lora_rank=lora_rank)
    linearizer = OneStepLinearizer(gx=g, gy=None, linear_network=linear_network)
    linearizer.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    linearizer = linearizer.to(device).eval()

    fm = FlowMatcher(linearizer, latent_size=latent_size).to(device).eval()
    return fm, linearizer


def build_lin_model(num_layers, in_ch, img_size, latent_size, lora_rank, device):
    from modules.invertable_network import InverseUnet
    from linearizer import Linearizer
    from linearizer.linear_network import TimeDependentLoRALinearLayer
    from song__unet import creat_song_unet

    gx = InverseUnet(num_layers, in_ch, img_size, creat_song_unet, model_channels=16)
    linear_network = TimeDependentLoRALinearLayer(latent_size, lora_rank, t_size=128)
    lin = Linearizer(gx=gx, linear_network=linear_network).to(device).eval()
    return lin


def get_B_linearizer(lin, latent_size, T, device):
    """Build B for the original Linearizer (full image-space latent)."""
    from linearizer.linear_network import TimeDependentLoRALinearLayer
    I = torch.eye(latent_size, device=device)
    B = I.clone()
    dt = 1.0 / T
    with torch.no_grad():
        for i in range(T - 2, -1, -1):
            t_k = i * dt
            t_tensor = torch.ones(1, device=device) * t_k
            A_t_k = lin.linear_network.get_lin_t(t_tensor).squeeze(0)
            t_k2 = t_k + 0.5 * dt
            t_tensor2 = torch.ones(1, device=device) * t_k2
            A_t_k2 = lin.linear_network.get_lin_t(t_tensor2).squeeze(0)
            k1 = (dt / (1.0 - t_k))  * (A_t_k  - I)
            k2 = (dt / (1.0 - t_k2)) * (A_t_k2 - I)
            k3 = k2
            t_k4 = t_k + dt
            t_tensor4 = torch.ones(1, device=device) * t_k4
            A_t_k4 = lin.linear_network.get_lin_t(t_tensor4).squeeze(0)
            k4 = (dt / (1.0 - t_k4)) * (A_t_k4 - I)
            M_k = I + (k1 + 2*k2 + 2*k3 + k4) / 6
            B = M_k @ B
    return B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   type=str, required=True)
    parser.add_argument('--latent_size',  type=int, default=128)
    parser.add_argument('--lora_rank',    type=int, default=8)
    parser.add_argument('--img_size',     type=int, default=64)
    parser.add_argument('--in_ch',        type=int, default=3)
    parser.add_argument('--T',            type=int, default=100)
    parser.add_argument('--lin_layers',   type=int, default=2,
                        help='Number of InvUnet layer pairs for the Linearizer gx')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    wandb.init(project='surjective-linearizer', name='benchmark_compare', config=vars(args))

    noise = torch.randn(100, args.in_ch, args.img_size, args.img_size, device=device)
    lin_latent_size = args.in_ch * args.img_size * args.img_size  # full image-space latent

    def count_params(model):
        return sum(p.numel() for p in model.parameters())

    def count_spnn_components(spnn):
        """Split SPNN params into gx (t, s, mix) and g_dagger (r networks)."""
        gx_p, gdagger_p = 0, 0
        for block in spnn.pinn.blocks:
            if hasattr(block, 'r'):  # ConvPINNBlock
                for name, param in block.named_parameters():
                    if name.startswith('r.'):
                        gdagger_p += param.numel()
                    else:
                        gx_p += param.numel()
            else:  # PixelUnshuffleBlock — no learnable params
                gx_p += sum(p.numel() for p in block.parameters())
        return gx_p, gdagger_p

    # ---- Surjective Linearizer ---- #
    print('\n--- Surjective Linearizer (SPNN, latent=128) ---')
    fm, linearizer = build_surj_model(args.checkpoint, args.latent_size, args.lora_rank,
                                       args.img_size, args.in_ch, device)
    surj_params = count_params(linearizer)
    surj_gx_p, surj_gdagger_p = count_spnn_components(linearizer.net_gx)
    surj_lin_p = count_params(linearizer.linear_network)
    print(f'Total params:  {surj_params:,}')
    print(f'  gx:          {surj_gx_p:,}')
    print(f'  g_dagger:    {surj_gdagger_p:,}')
    print(f'  A(t) linear: {surj_lin_p:,}')
    t0 = time.perf_counter()
    B_surj = fm.get_sampling_terms(device, T=args.T, sampling_method='rk')
    torch.cuda.synchronize()
    surj_B_build = time.perf_counter() - t0
    print(f'B build time:  {surj_B_build:.3f}s   (latent={args.latent_size}, B={args.latent_size}x{args.latent_size})')

    with torch.no_grad():
        surj_inf = bench(lambda: fm.sample_one_step(noise, device=device, B=B_surj))
    print(f'Inference:     {surj_inf*1000:.2f}ms / 100 images  ({surj_inf/100*1000:.3f}ms per image)')

    # ---- Original Linearizer (InverseUnet) ---- #
    print(f'\n--- Original Linearizer (InverseUnet, latent={lin_latent_size}) ---')
    lin = build_lin_model(args.lin_layers, args.in_ch, args.img_size,
                          lin_latent_size, args.lora_rank, device)
    lin_params = count_params(lin)
    lin_gx_p = count_params(lin.net_gx)   # gx = gx_inverse (same weights, exact inverse)
    lin_lin_p = count_params(lin.linear_network)
    print(f'Total params:  {lin_params:,}')
    print(f'  gx (=gx_inv, exact): {lin_gx_p:,}')
    print(f'  g_dagger:            0  (exact inverse, no extra params)')
    print(f'  A(t) linear:         {lin_lin_p:,}')
    t0 = time.perf_counter()
    B_lin = get_B_linearizer(lin, lin_latent_size, args.T, device)
    torch.cuda.synchronize()
    lin_B_build = time.perf_counter() - t0
    print(f'B build time:  {lin_B_build:.3f}s   (latent={lin_latent_size}, B={lin_latent_size}x{lin_latent_size})')

    with torch.no_grad():
        def lin_infer():
            g_x = lin.gy(noise, mode='gx')
            g_x = (g_x.reshape(g_x.shape[0], -1) @ B_lin).reshape(g_x.shape)
            return lin.gx_inverse(g_x, mode='gx')
        lin_inf = bench(lin_infer)
    print(f'Inference:     {lin_inf*1000:.2f}ms / 100 images  ({lin_inf/100*1000:.3f}ms per image)')

    # ---- Summary ---- #
    print('\n========== SUMMARY ==========')
    print(f'{"":30} {"Surj. Lin.":>15} {"Orig. Lin.":>15}')
    print(f'{"Latent size":30} {args.latent_size:>15} {lin_latent_size:>15}')
    print(f'{"Total params":30} {surj_params:>15,} {lin_params:>15,}')
    print(f'{"  gx params":30} {surj_gx_p:>15,} {lin_gx_p:>15,}')
    print(f'{"  g_dagger params":30} {surj_gdagger_p:>15,} {"0 (exact inv)":>15}')
    print(f'{"  A(t) params":30} {surj_lin_p:>15,} {lin_lin_p:>15,}')
    print(f'{"B build (s)":30} {surj_B_build:>15.3f} {lin_B_build:>15.3f}')
    print(f'{"Inference 100 imgs (ms)":30} {surj_inf*1000:>15.2f} {lin_inf*1000:>15.2f}')
    print(f'{"Per image (ms)":30} {surj_inf/100*1000:>15.3f} {lin_inf/100*1000:>15.3f}')

    wandb.log({
        'surj/latent_size':        args.latent_size,
        'surj/params_total':       surj_params,
        'surj/params_gx':          surj_gx_p,
        'surj/params_gdagger':     surj_gdagger_p,
        'surj/params_linear':      surj_lin_p,
        'lin/params_total':        lin_params,
        'lin/params_gx':           lin_gx_p,
        'lin/params_gdagger':      0,
        'lin/params_linear':       lin_lin_p,
        'surj/B_build_s':          surj_B_build,
        'surj/inference_100img_ms': surj_inf * 1000,
        'surj/per_image_ms':       surj_inf / 100 * 1000,
        'lin/latent_size':         lin_latent_size,
        'lin/B_build_s':           lin_B_build,
        'lin/inference_100img_ms': lin_inf * 1000,
        'lin/per_image_ms':        lin_inf / 100 * 1000,
    })
    wandb.finish()


if __name__ == '__main__':
    main()
