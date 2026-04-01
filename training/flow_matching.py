import torch
import torch.nn as nn

from piq import LPIPS

from linearizer.one_step import OneStepLinearizer


class FlowMatcher(nn.Module):
    """Wraps OneStepLinearizer for Conditional Flow Matching training and sampling.

    Training: learns A(t) to predict the target latent g_x1 from any interpolated
    point g_xt on the straight-line path between g_x0 (noise) and g_x1 (data).

    Sampling: integrates the ODE in latent space using Euler or RK4, then decodes.
    One-step sampling: collapses the full ODE into a single precomputed matrix B.
    """

    def __init__(self, linearizer: OneStepLinearizer, latent_size):
        super().__init__()
        self.linearizer = linearizer
        self.latent_size = latent_size
        self.lpips = LPIPS(replace_pooling=True, reduction="none")

    def forward(self, x1, x0=None, noise_level=0.0):
        """Compute training loss for a batch of real images x1."""
        return self.training_losses(x1, x0, noise_level)

    def training_losses(self, x1, x0=None, noise_level=0.0):
        """Compute the full flow matching loss.

        Encodes x0 (noise) and x1 (data) into latent space, interpolates a random
        point g_xt along the straight-line path, predicts g_x1 via A(t), and
        combines MSE in latent space with LPIPS reconstruction losses.
        """
        batch_size = x1.shape[0]
        device = x1.device

        if x0 is None:
            x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=device)

        # --- project into induced space --- #
        # Single shared g: both noise and data are encoded through gx.
        g_x0 = self.linearizer.gx(x0)
        g_x1 = self.linearizer.gx(x1)

        # --- predict in the induced space --- #
        g_xt = (1.0 - t)[:, None] * g_x0 + t[:, None] * g_x1
        g_x1_p = self.linearizer.A(g_xt, t=t)

        # --- calculate losses --- #
        induced_space_loss = ((g_x1_p - g_x1) ** 2).mean()

        # add regularizing noise
        g_x0 = g_x0 + torch.randn_like(g_x1) * noise_level
        g_x1 = g_x1 + torch.randn_like(g_x1) * noise_level

        # reconstruction losses
        x0_tag = self.linearizer.gx_inverse(g_x0)
        x0_rec_loss = ((x0 - x0_tag) ** 2).mean()
        x1_tag = self.linearizer.gx_inverse(g_x1)
        x1_rec_loss = self.lpips(x1, self.linearizer.gx_inverse(g_x1)).mean()
        x1_pred_rec_loss = self.lpips(x1, self.linearizer.gx_inverse(g_x1_p)).mean()
        loss_r_x0_tag = x0_tag.pow(2).mean()
        loss_r_x1_tag = x1_tag.pow(2).mean()

        loss = induced_space_loss + x0_rec_loss + x1_rec_loss + x1_pred_rec_loss + loss_r_x0_tag + loss_r_x1_tag
        return loss

    def sample(self, x, device, steps=100, method='euler', return_path=False):
        """Generate images by integrating the flow ODE in latent space.

        Encodes noise x via gx, integrates using Euler or RK4 for the given
        number of steps, then decodes the final latent via gx_inverse.
        """
        self.linearizer.eval()
        with torch.no_grad():
            g_x = self.linearizer.gx(x)
            dt = 1.0 / steps

            if return_path:
                path = [g_x]

            if method == 'euler':
                for i in range(0, steps - 1):
                    t = torch.full((g_x.shape[0],), i * dt, device=device)
                    g_t_model = self.linearizer.A(g_x, t=t)
                    g_vt = (g_t_model - g_x) / (1 - t)[:, None]
                    g_x = g_x + g_vt * dt
                    if return_path:
                        path.append(g_x)

            elif method == 'rk':
                for i in range(0, steps - 1):
                    t = torch.full((g_x.shape[0],), i * dt, device=device)

                    g_t_model = self.linearizer.A(g_x, t=t)
                    k1 = (g_t_model - g_x) / (1 - t)[:, None]

                    g_x_k2 = g_x + 0.5 * dt * k1
                    t_k2 = t + 0.5 * dt
                    g_t_model_k2 = self.linearizer.A(g_x_k2, t=t_k2)
                    k2 = (g_t_model_k2 - g_x_k2) / (1 - t_k2)[:, None]

                    g_x_k3 = g_x + 0.5 * dt * k2
                    g_t_model_k3 = self.linearizer.A(g_x_k3, t=t_k2)
                    k3 = (g_t_model_k3 - g_x_k3) / (1 - t_k2)[:, None]

                    g_x_k4 = g_x + dt * k3
                    t_k4 = t + dt
                    g_t_model_k4 = self.linearizer.A(g_x_k4, t=t_k4)
                    k4 = (g_t_model_k4 - g_x_k4) / (1 - t_k4)[:, None]

                    g_x = g_x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

                    if return_path:
                        path.append(g_x)

            g_x = self.linearizer.gx_inverse(g_x)

        if return_path:
            return g_x, path
        return g_x

    def sample_one_step(self, x, device, sampling_method='rk', T=100, B=None):
        """Generate images in a single matrix multiplication using precomputed B.

        Encodes noise x, applies the precomputed composed matrix B (which
        represents the full T-step ODE in one shot), and decodes the result.
        """
        self.linearizer.eval()
        with torch.no_grad():
            g_x = self.linearizer.gx(x)
            if B is None:
                B = self.get_sampling_terms(device, sampling_method=sampling_method, T=T)
            B = B.to(device)
            g_x = (g_x.reshape(g_x.shape[0], -1) @ B).reshape(g_x.shape)
            g_x = self.linearizer.gx_inverse(g_x)
        return g_x

    def get_sampling_terms(self, device, T=100, sampling_method='euler'):
        """Precompute the combined matrix B = M_{T-1} @ ... @ M_0.

        B collapses T Euler or RK4 steps into a single matrix, enabling
        one-step inference. Computed once offline before sampling.
        """
        with torch.no_grad():
            I = torch.eye(self.latent_size).to(device)
            B = I
            dt = 1.0 / T

            if sampling_method == 'euler':
                for i in range(T - 2, -1, -1):
                    t_k = i * dt
                    A_t_k = self.linearizer.linear_network.get_lin_t(torch.ones(1).to(device) * t_k).squeeze(0)
                    M_k = I + (dt / (1.0 - t_k)) * (A_t_k - I)
                    B = M_k @ B

            elif sampling_method == 'rk':
                for i in range(T - 2, -1, -1):
                    t_k = i * dt
                    A_t_k = self.linearizer.linear_network.get_lin_t(torch.ones(1).to(device) * t_k).squeeze(0)
                    k1 = (dt / (1.0 - t_k)) * (A_t_k - I)

                    t_k2 = t_k + 0.5 * dt
                    A_t_k2 = self.linearizer.linear_network.get_lin_t(torch.ones(1).to(device) * t_k2).squeeze(0)
                    k2 = (dt / (1.0 - t_k2)) * (A_t_k2 - I)
                    k3 = (dt / (1.0 - t_k2)) * (A_t_k2 - I)

                    t_k4 = t_k + dt
                    A_t_k4 = self.linearizer.linear_network.get_lin_t(torch.ones(1).to(device) * t_k4).squeeze(0)
                    k4 = (dt / (1.0 - t_k4)) * (A_t_k4 - I)

                    M_k = I + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                    B = M_k @ B

            return B


def train_flow_matching(linearizer, dataloader, epochs=10, lr=1e-4, noise_level=0.0,
                        eval_epoch=10, steps=100, num_of_ch=1, sampling_method='rk',
                        save_folder='', img_size=32, latent_size=10):
    """Run the full flow matching training loop.

    Wraps the linearizer in a FlowMatcher, sets up Adam optimizer and multi-GPU
    DataParallel if available, then trains for the given number of epochs.
    Saves model checkpoints and generated sample grids every eval_epoch epochs.
    Resumes automatically from the latest checkpoint if one exists.
    """
    import os
    from utils.sampling_utils import sample_and_save

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Available devices: {torch.cuda.device_count()}")

    linearizer = linearizer.to(device)
    fm = FlowMatcher(linearizer, latent_size=latent_size)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        fm = torch.nn.DataParallel(fm)

    fm = fm.to(device)
    optimizer = torch.optim.Adam([{"params": linearizer.parameters(), "lr": lr}],
                                 betas=(0.9, 0.999), weight_decay=0.0)

    models_save_path = f'{save_folder}/models'
    artifacts_save_path = f'{save_folder}/artifacts'
    os.makedirs(models_save_path, exist_ok=True)
    os.makedirs(artifacts_save_path, exist_ok=True)

    # --- resume from checkpoint if available --- #
    start_epoch = 0
    ckpt_path = f'{models_save_path}/checkpoint.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        fm_eval = fm.module if isinstance(fm, torch.nn.DataParallel) else fm
        fm_eval.linearizer.load_state_dict(ckpt['linearizer'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        fm.train()
        for batch_idx, (x1, _) in enumerate(dataloader):
            x1 = x1.to(device)
            optimizer.zero_grad()
            loss = fm(x1=x1, x0=None, noise_level=noise_level)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs} completed, Avg Loss: {avg_loss:.4f}')

        # save checkpoint every epoch for resume support
        fm_eval = fm.module if isinstance(fm, torch.nn.DataParallel) else fm
        torch.save({'epoch': epoch, 'linearizer': fm_eval.linearizer.state_dict(),
                    'optimizer': optimizer.state_dict()}, ckpt_path)

        if epoch % eval_epoch == 0:
            fm.eval()
            torch.save(fm_eval.linearizer, f'{models_save_path}/lin_{epoch}.pth')
            print(f"Model saved to {models_save_path}/lin_{epoch}.pth")
            sample_and_save(fm=fm_eval, num_of_images=16, device=device, steps=steps,
                            epoch=epoch, num_of_ch=num_of_ch, sampling_method=sampling_method,
                            img_size=img_size, save_dir=artifacts_save_path)
