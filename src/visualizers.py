import typing
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from PIL import Image
import imageio
from tqdm import tqdm
from src.constants import POLLUTANT_COLUMNS, OUTPUT_DIR, TQDM_DISABLE

if typing.TYPE_CHECKING:
    from src.trainer import Trainer

class MDNVisualizer:
    def __init__(self, trainer: "Trainer"):
        self.trainer = trainer

    def plot_timeseries_from_val(self, indeces, num_targets=None, title=None):
        results = self.trainer.predict_from_val(indeces)
        predictions = results["predictions"]
        ground_truths = results["ground_truths"]
        mus = results["mus"]
        sigmas = results["sigmas"]
        alphas = results["alphas"]
        timestamps = results["timestamps"]
        unnormalized_targets = results["unnormalized_targets"]
        means = results["means"]
        stds = results["stds"]

        x = np.array(timestamps)

        if num_targets is None:
            num_targets = len(POLLUTANT_COLUMNS)
        num_mixtures = mus.shape[1]

        # Grid config: 2 columns
        ncols = 2
        nrows = math.ceil(num_targets / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows), sharex=True)
        axs = axs.flatten()

        for target_idx in range(num_targets):
            ax = axs[target_idx]

            y_pred_norm = predictions[:, target_idx]
            y_true = unnormalized_targets[:, target_idx]

            mu = mus[:, :, target_idx]
            sigma = sigmas[:, :, target_idx]
            alpha = alphas

            mixture_var = np.sum(alpha * (sigma**2 + mu**2), axis=1) - np.sum(alpha * mu, axis=1)**2
            mixture_std = np.sqrt(np.clip(mixture_var, 1e-8, None))

            mean = means[0][target_idx]
            std = stds[0][target_idx]

            y_pred = y_pred_norm * std + mean
            mixture_std_scaled = mixture_std * std
            mu_scaled = mu * std + mean

            z_90 = 1.645
            z_95 = 1.960

            ax.plot(x, y_true, label="Ground Truth", color="black", linewidth=2)
            ax.plot(x, y_pred, label="Prediction (E[y])", color="blue", linestyle="--")

            # 90% CI (clip lower bound to 0)
            ax.fill_between(
                x,
                np.clip(y_pred - z_90 * mixture_std_scaled, a_min=0, a_max=None),
                y_pred + z_90 * mixture_std_scaled,
                color='blue', alpha=0.12, label="90% CI"
            )

            # 95% CI (clip lower bound to 0)
            ax.fill_between(
                x,
                np.clip(y_pred - z_95 * mixture_std_scaled, a_min=0, a_max=None),
                y_pred + z_95 * mixture_std_scaled,
                color='blue', alpha=0.07, label="95% CI"
            )

            ax.set_title(f"{POLLUTANT_COLUMNS[target_idx].replace('components.', '').replace('_','.').upper()} Forecast")
            ax.set_ylabel("Concentration (ug/m³)")
            ax.grid(True)

        # Remove unused subplots
        for i in range(num_targets, len(axs)):
            fig.delaxes(axs[i])

        # axs[-1].set_xlabel("Timestamp")
        if title:
            plt.suptitle(title, fontsize=16)
      
        fig.legend(
                *axs[0].get_legend_handles_labels(),
                loc="upper right",
                ncol=2
                # bbox_to_anchor=(0.82, 0.08),
                # frameon=False,
            )
        plt.tight_layout()
        # plt.show()

    def plot_mixture_distributions_at_timestep(self, timestep_index: int, num_targets=None, save_path=None):
        result = self.trainer.predict_from_val([timestep_index])

        mus = result["mus"][0]
        sigmas = result["sigmas"][0]
        alphas = result["alphas"][0]
        means = result["means"][0]
        stds = result["stds"][0]
        timestamp = result["timestamps"][0]

        if num_targets is None:
            num_mixtures, num_targets = mus.shape
        else:
            num_mixtures = mus.shape[0]
        ncols = 2
        nrows = math.ceil(num_targets / ncols)

        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows), sharex=False)
        
        # fig.suptitle("Mixture Distributions", fontsize=16)
        axs = axs.flatten()

        mixture_line = None
        sum_line = None
        

        for target_idx in range(num_targets):
            ax = axs[target_idx]

            mu = mus[:, target_idx]
            sigma = sigmas[:, target_idx]
            alpha = alphas

            mean = means[target_idx]
            std = stds[target_idx]

            mu_scaled = mu * std + mean
            sigma_scaled = sigma * std
            x_range = np.linspace(-5, 5, 1000)
            x_scaled = x_range * std + mean

            total_pdf = np.zeros_like(x_scaled)

            for k in range(num_mixtures):
                pdf = alpha[k] * norm.pdf(x_scaled, loc=mu_scaled[k], scale=sigma_scaled[k])
                total_pdf += pdf

                # Save the first mixture line for the legend
                line = ax.plot(x_scaled, pdf, linestyle="--", alpha=0.6)
                if target_idx == 0 and k == 0:
                    mixture_line = line[0]

            # Plot the combined mixture sum
            line = ax.plot(x_scaled, total_pdf, color="black", linewidth=2)
            if target_idx == 0:
                sum_line = line[0]

            mu_center = np.sum(mu_scaled * alpha)
            std_combined = np.sqrt(np.sum(alpha * sigma_scaled**2))
            ax.set_xlim(mu_center - 4 * std_combined, mu_center + 4 * std_combined)
            ax.set_xlabel("Concentration (mg/m³)")
            ax.set_title(f"{POLLUTANT_COLUMNS[target_idx]}".replace("components.", "").replace("_", ".").upper())
            ax.set_ylabel("Density")
            ax.grid(True)

        for idx in range(num_targets, len(axs)):
            fig.delaxes(axs[idx])

       
        if mixture_line and sum_line:
            fig.legend(
                handles=[mixture_line, sum_line],
                labels=["Mixtures", "Mixture Sum"],
                loc="upper right",
                # bbox_to_anchor=(0.93, 0.15),
                frameon=False,
                ncols=2,
            )

        fig.suptitle(
            f"Mixture Distributions",
            fontsize=16, fontweight='bold',
            # ha='right', va='bottom'
        )

        fig.text(
            0.1, 0.90, f"{timestamp.strftime('%-I %p')}",
            fontsize=12, fontweight='bold',
            ha='right', va='bottom'
        )


        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()

    def generate_mixture_gif(self, start: int, end: int, num_targets :int ,  output_path: str = "mixture_evolution.gif"):
        frame_paths = []
        for t in tqdm(range(start, end), desc="Generating frames", disable=TQDM_DISABLE):
            frame_file = OUTPUT_DIR / f"frame_{t:03d}.png"
            self.plot_mixture_distributions_at_timestep(timestep_index=t, num_targets=num_targets, save_path=frame_file)
            frame_paths.append(frame_file)

        base_size = Image.open(frame_paths[0]).size
        images = [Image.open(fp).resize(base_size) for fp in frame_paths]
        imageio.mimsave(OUTPUT_DIR/output_path, images, duration=0.8)

        for fp in frame_paths:
            os.remove(fp)

        print(f"GIF saved to: {output_path}")

    def generate_insights_from_timestep(self, timestep_index):
        result = self.trainer.predict_from_val([timestep_index])

        mus = result["mus"][0]          # (num_mixtures, num_targets)
        sigmas = result["sigmas"][0]    # (num_mixtures, num_targets)
        alphas = result["alphas"][0]    # (num_mixtures,)
        means = result["means"][0]      # (num_targets,)
        stds = result["stds"][0]        # (num_targets,)
        timestamp = result["timestamps"][0]  # datetime

        num_mixtures, num_targets = mus.shape

        expected_values = []
        ci_90_bounds = []
        ci_95_bounds = []

        for t in range(num_targets):
            mu = mus[:, t] * stds[t] + means[t]
            sigma = sigmas[:, t] * stds[t]
            alpha = alphas

            expected = np.sum(alpha * mu)
            expected_values.append(expected)

            mixture_var = np.sum(alpha * (sigma**2 + mu**2)) - expected**2
            std_dev = np.sqrt(mixture_var)

            z90 = norm.ppf(0.95)
            z95 = norm.ppf(0.975)

            ci_90 = (
                max(0, expected - z90 * std_dev),
                expected + z90 * std_dev
            )

            ci_95 = (
                max(0, expected - z95 * std_dev),
                expected + z95 * std_dev
            )

            ci_90_bounds.append(ci_90)
            ci_95_bounds.append(ci_95)

        mus_unnorm = mus * stds[None, :] + means[None, :]
        sigmas_unnorm = sigmas * stds[None, :]

        timestamp_str = timestamp.strftime("%B %d, %Y at %-I:%M %p") if hasattr(timestamp, "strftime") else str(timestamp)
        report_lines = []

        for i in range(num_targets):
            pollutant = POLLUTANT_COLUMNS[i].replace("components.", "").replace("_", ".").upper()
            expected = expected_values[i]
            ci90_lower, ci90_upper = ci_90_bounds[i]
            ci95_lower, ci95_upper = ci_95_bounds[i]

            report = (
                f"At {timestamp_str}, the expected concentration of {pollutant} was estimated at "
                f"{expected:.2f} ug/m³.\n"
                f"We are 90% confident that the true value lies between {ci90_lower:.2f} and {ci90_upper:.2f} mg/m³,\n"
                f"and 95% confident it lies between {ci95_lower:.2f} and {ci95_upper:.2f} mg/m³.\n"
            )
            report_lines.append(report)

        return {
            "timestamp": timestamp,
            "expected_values": expected_values,
            "mixtures": {
                "mu": mus_unnorm,
                "sigma": sigmas_unnorm,
                "alpha": alphas
            },
            "ci_90": ci_90_bounds,
            "ci_95": ci_95_bounds,
            "text_report": report_lines
        }