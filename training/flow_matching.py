import torch
import torch.nn as nn
import wandb

from piq import LPIPS

from linearizer.one_step import OneStepLinearizer


class FlowMatcher(nn.Module):
    """Wraps OneStepLinearizer for Conditional Flow Matching training and sampling.

    Training: learns A(t) to predict the target latent g_x1 (endpoint prediction)
    from any interpolated point g_xt on the straight-line path between g_x0 (noise)
    and g_x1 (data).

    Sampling: integrates the ODE in latent space using Euler or RK4, then decodes.
    Velocity is derived from the endpoint prediction as v = (A(g_xt, t) - g_xt) / (1 - t).
    One-step sampling: collapses the full ODE into a single precomputed matrix B.
    """

    def __init__(self, linearizer: OneStepLinearizer, latent_size, var_match_lambda=0.0):
        super().__init__()
        self.linearizer = linearizer
        self.latent_size = latent_size
        self.var_match_lambda = var_match_lambda
        self.lpips = LPIPS(replace_pooling=True, reduction="none")

    def forward(self, x1, x0=None, noise_level=0.0):
        """Compute training loss for a batch of real images x1."""
        return self.training_losses(x1, x0, noise_level)

    def training_losses(self, x1, x0=None, noise_level=0.0):
        """Compute the full flow matching loss.

        Encodes x0 (noise) and x1 (data) into latent space, interpolates a random
        point g_xt along the straight-line path, predicts g_x1 via A(t), and
        combines MSE in latent space with LPIPS reconstruction losses.

        A(t) uses endpoint prediction: A(g_xt, t) ≈ g_x1.
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
        g_x1_p = self.linearizer.A(g_xt, t=t)  # predicts g_x1 (endpoint)

        # --- latent diversity diagnostics --- #
        low_t_mask  = t < 0.33
        high_t_mask = t > 0.66
        diag = {
            'var/g_x0':      g_x0.var(dim=0).mean().item(),
            'var/g_x1':      g_x1.var(dim=0).mean().item(),
            'var/g_xt':      g_xt.var(dim=0).mean().item(),
            'var/g_x1_pred': g_x1_p.var(dim=0).mean().item(),
        }
        if low_t_mask.sum() > 1:
            diag['var/g_xt_low_t']  = g_xt[low_t_mask].var(dim=0).mean().item()
            diag['var/pred_low_t']  = g_x1_p[low_t_mask].var(dim=0).mean().item()
        if high_t_mask.sum() > 1:
            diag['var/g_xt_high_t'] = g_xt[high_t_mask].var(dim=0).mean().item()
            diag['var/pred_high_t'] = g_x1_p[high_t_mask].var(dim=0).mean().item()
        wandb.log(diag)

        # --- calculate losses --- #
        # endpoint prediction loss: A(g_xt, t) should equal g_x1
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
        # loss_r_x0_tag = x0_tag.pow(2).mean()
        # loss_r_x1_tag = x1_tag.pow(2).mean()

        # variance matching: only penalize when g(noise) is MORE collapsed than g(faces)
        var_match_loss = torch.relu(g_x1.var(dim=0).detach() - g_x0.var(dim=0)).pow(2).mean()
        wandb.log({'loss/var_match': var_match_loss.item()})

        loss = (induced_space_loss + x0_rec_loss + x1_rec_loss + x1_pred_rec_loss +
                8 * var_match_loss)
        return loss

    def sample(self, x, device, steps=100, method='euler', return_path=False):
        """Generate images by integrating the flow ODE in latent space.

        Encodes noise x via gx, integrates using Euler or RK4 for the given
        number of steps, then decodes the final latent via gx_inverse.

        Velocity is derived from the endpoint prediction:
            v = (A(g_xt, t) - g_xt) / (1 - t)
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

                    # k1
                    g_t_model = self.linearizer.A(g_x, t=t)
                    k1 = (g_t_model - g_x) / (1 - t)[:, None]

                    # k2
                    g_x_k2 = g_x + 0.5 * dt * k1
                    t_k2 = t + 0.5 * dt
                    g_t_model_k2 = self.linearizer.A(g_x_k2, t=t_k2)
                    k2 = (g_t_model_k2 - g_x_k2) / (1 - t_k2)[:, None]

                    # k3
                    g_x_k3 = g_x + 0.5 * dt * k2
                    g_t_model_k3 = self.linearizer.A(g_x_k3, t=t_k2)
                    k3 = (g_t_model_k3 - g_x_k3) / (1 - t_k2)[:, None]

                    # k4
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

        Each step uses the endpoint-prediction velocity: v = (A - I)*g_x / (1-t),
        giving M_k = I + dt/(1-t) * (A_t - I).
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
                        save_folder='', img_size=32, latent_size=10, var_match_lambda=0.0):
    """Run the full flow matching training loop.

    Wraps the linearizer in a FlowMatcher, sets up Adam optimizer and multi-GPU
    DataParallel if available, then trains for the given number of epochs.
    Saves model checkpoints and generated sample grids every eval_epoch epochs.

    var_match_lambda: weight for the variance matching loss term.
                     Sweep suggested values: 0, 1, 2, 4, 8, 16.
    """
    import os
    from utils.sampling_utils import sample_and_save

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Available devices: {torch.cuda.device_count()}")

    linearizer = linearizer.to(device)
    fm = FlowMatcher(linearizer, latent_size=latent_size, var_match_lambda=var_match_lambda)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        fm = torch.nn.DataParallel(fm)

    fm = fm.to(device)
    optimizer = torch.optim.Adam([{"params": linearizer.parameters(), "lr": lr}],
                                 betas=(0.9, 0.999), weight_decay=0.0)

    artifacts_save_path = f'{save_folder}/artifacts'
    os.makedirs(artifacts_save_path, exist_ok=True)

    fixed_noise = torch.randn(16, num_of_ch, img_size, img_size, device=device)

    for epoch in range(epochs):
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
                wandb.log({'batch_loss': loss.item(), 'epoch': epoch + 1})

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs} completed, Avg Loss: {avg_loss:.4f}')
        wandb.log({'avg_loss': avg_loss, 'epoch': epoch + 1})

        fm_eval = fm.module if isinstance(fm, torch.nn.DataParallel) else fm

        if epoch % eval_epoch == 0:
            fm.eval()
            sample_and_save(fm=fm_eval, num_of_images=16, device=device, steps=steps,
                            epoch=epoch, num_of_ch=num_of_ch, sampling_method=sampling_method,
                            img_size=img_size, save_dir=artifacts_save_path, fixed_noise=fixed_noise)
            wandb.log({
                'samples_one': wandb.Image(f'{artifacts_save_path}/one_{epoch}.png'),
                'samples_multi': wandb.Image(f'{artifacts_save_path}/multi_{epoch}.png'),
                'epoch': epoch + 1,
            })
