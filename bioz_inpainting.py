"""Run leave-one-subject-out wrist BioZ gap reconstruction."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, cholesky, toeplitz
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

RATE = 125
WINDOW = 1000
GAPS = (0.5, 1, 2)
DECIMATION = 10
PADDING = RATE
BANDPASS = butter(3, [0.5, 12], fs=RATE, btype="bandpass", output="sos")
DILATIONS = (1, 4, 16)


def preprocess(raw):
    """Filter a padded native-rate window and retain its central eight seconds."""
    return sosfiltfilt(BANDPASS, resample_poly(raw, 1, DECIMATION))[
        PADDING:PADDING + WINDOW
    ].astype(np.float32)


class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(channels, 32, 7, padding=3), nn.SiLU())
        self.middle = nn.Sequential(
            nn.AvgPool1d(4),
            *[layer for i, dilation in enumerate(DILATIONS) for layer in (
                nn.Conv1d(32 if i == 0 else 64, 64, 15,
                          padding=7 * dilation, dilation=dilation), nn.SiLU(),
            )],
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(96, 32, 7, padding=3), nn.SiLU(), nn.Conv1d(32, 1, 1)
        )

    def forward(self, x):
        skip = self.encoder(x)
        return self.decoder(torch.cat((skip, self.middle(skip)), dim=1))


def load_trials(root):
    subjects = {}
    for path in sorted(root.rglob("*_bioz.csv")):
        subject = next((m[1] for part in path.parts
                        if (m := re.fullmatch(r"subject(\d+)_day\d+", part))), None)
        if subject is None or not any(
            re.search(r"baseline|rest", part, re.I) for part in path.parts
        ):
            continue
        with path.open(encoding="utf-8-sig") as stream:
            first_row = next(csv.reader(stream))
        try:
            float(first_row[0])
            skip = 0
        except ValueError:
            skip = 1
        data = np.loadtxt(path, delimiter=",", usecols=(0, 1), skiprows=skip, ndmin=2)
        if len(data) < WINDOW * 10 or not np.isfinite(data).all():
            continue
        deltas = np.diff(data[:, 0])
        if not np.allclose(deltas, 1 / 1250, atol=0.00005, rtol=0):
            continue
        # Retain raw data: masking must precede both antialiasing and filtering.
        size = (WINDOW + 2 * PADDING) * DECIMATION
        for start in range(0, len(data) - size + 1, WINDOW // 2 * DECIMATION):
            segment = data[start:start + size, 1]
            subjects.setdefault(subject, []).append(segment.copy())
    return {subject: np.stack(waves) for subject, waves in subjects.items()}


def make_examples(waves, rng, seconds=None):
    targets, contexts, masks, scales = [], [], [], []
    for raw in waves:
        length = round(RATE * (seconds if seconds is not None else rng.choice(GAPS)))
        start = rng.integers(RATE, WINDOW - length - RATE + 1)
        mask = np.zeros(WINDOW, dtype=bool)
        mask[start:start + length] = True
        raw_start = (PADDING + start) * DECIMATION
        raw_end = raw_start + length * DECIMATION
        # Replace every missing native-rate sample using visible endpoints only.
        # Neither resample_poly nor sosfiltfilt can then see the hidden values.
        filled = raw.copy()
        filled[raw_start:raw_end] = np.linspace(
            raw[raw_start - 1], raw[raw_end], raw_end - raw_start + 2
        )[1:-1]
        visible_wave = preprocess(filled)
        wave = preprocess(raw)  # Intact signal is used only as the target.
        scale = max(float(visible_wave[~mask].std()), 1e-6)
        mean = visible_wave[~mask].mean()
        target = (wave - mean) / scale
        targets.append(target)
        contexts.append(np.where(mask, 0, (visible_wave - mean) / scale))
        masks.append(mask)
        scales.append(scale)
    arrays = [np.stack(values)[:, None, :] for values in (targets, contexts, masks)]
    return TensorDataset(*(torch.from_numpy(array) for array in arrays),
                         torch.tensor(scales, dtype=torch.float32))


class ConditionalVAE(nn.Module):
    latent_size = 16

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.AvgPool1d(10), nn.Flatten(), nn.Linear(3 * WINDOW // 10, 128),
            nn.SiLU(), nn.Linear(128, 2 * self.latent_size),
        )
        self.decoder = UNet(2 + self.latent_size)

    def sample(self, condition, latent):
        latent_channels = latent[:, :, None].expand(-1, -1, condition.size(-1))
        return self.decoder(torch.cat((condition, latent_channels), dim=1))

    def loss(self, condition, target, mask):
        hidden_target = torch.where(mask, target, 0)
        mean, log_variance = self.encoder(torch.cat((condition, hidden_target), dim=1)).chunk(2, dim=1)
        log_variance = log_variance.clamp(-8, 4)
        latent = mean + torch.randn_like(mean) * (log_variance / 2).exp()
        reconstruction = (self.sample(condition, latent) - target)[mask].square().mean()
        kl = 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).mean()
        return reconstruction + 0.1 * kl


class GaussianProcess:
    """Stationary kernel from training-window autocovariance."""

    def __init__(self, waves):
        centered = waves - waves.mean(axis=1, keepdims=True)
        normalized = centered / centered.std(axis=1, keepdims=True).clip(1e-6)
        spectrum = np.fft.rfft(normalized, n=2 * WINDOW, axis=1)
        power = np.square(np.abs(spectrum)).mean(axis=0)
        self.autocovariance = np.fft.irfft(power)[:WINDOW] / WINDOW
        self.cache = {}

    def posterior(self, observed, gap):
        length = int(gap.sum())
        start = int(np.flatnonzero(gap)[0])
        if length not in self.cache:
            local = np.arange(2 * RATE + length)
            inside = (local >= RATE) & (local < RATE + length)
            covariance = toeplitz(self.autocovariance[:len(local)])
            cross = covariance[np.ix_(inside, ~inside)]
            visible = covariance[np.ix_(~inside, ~inside)] + 0.01 * np.eye(2 * RATE)
            weights = cho_solve(cho_factor(visible), cross.T).T
            posterior = covariance[np.ix_(inside, inside)] - weights @ cross.T
            factor = cholesky(posterior + 1e-8 * np.eye(length), lower=True)
            self.cache[length] = weights, factor, inside
        weights, factor, inside = self.cache[length]
        output = observed.astype(np.float64)
        context = output[start - RATE:start + length + RATE][~inside]
        return weights @ context, factor

    def mean(self, observed, gap):
        output = observed.astype(np.float64).copy()
        output[gap] = self.posterior(observed, gap)[0]
        return output

    def sample(self, observed, gap, rng):
        mean, factor = self.posterior(observed, gap)
        output = observed.astype(np.float64).copy()
        output[gap] = mean + factor @ rng.standard_normal(int(gap.sum()))
        return output


def train(waves, epochs, device, rng, epoch_callback=None, model_callback=None):
    direct, flow = UNet(2).to(device), UNet(4).to(device)
    diffusion = UNet(4).to(device)
    optimizer = torch.optim.AdamW(
        [*direct.parameters(), *flow.parameters(), *diffusion.parameters()], lr=0.001
    )
    vae = ConditionalVAE().to(device)
    gan = UNet(3).to(device)
    discriminator = UNet(3).to(device)
    generator_optimizer = torch.optim.AdamW([*vae.parameters(), *gan.parameters()], lr=0.001)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.001)
    adversarial_loss = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        epoch_losses = np.zeros(6)
        seen = 0
        loader = DataLoader(make_examples(waves, rng), batch_size=32, shuffle=True)
        for target, context, mask, _ in loader:
            target, context, mask = (item.to(device) for item in (target, context, mask))
            condition = torch.cat((context, mask), dim=1)
            t = torch.rand((len(target), 1, 1), device=device)
            noise = torch.randn_like(target)
            state = torch.where(mask, torch.lerp(noise, target, t), context)
            velocity = flow(torch.cat((condition, state, t.expand_as(state)), dim=1))
            direct_error = (direct(condition) - target)[mask].square().mean()
            flow_error = (velocity - (target - noise))[mask].square().mean()
            angle = t * (math.pi / 2)
            alpha, sigma = angle.cos(), angle.sin()
            noisy = torch.where(mask, alpha * target + sigma * noise, context)
            predicted_v = diffusion(torch.cat((condition, noisy, t.expand_as(noisy)), dim=1))
            diffusion_error = (predicted_v - (alpha * noise - sigma * target))[mask].square().mean()
            optimizer.zero_grad()
            (direct_error + flow_error + diffusion_error).backward()
            optimizer.step()
            proposal = gan(torch.cat((condition, noise), dim=1))
            fake = torch.where(mask, proposal, context)
            real = torch.where(mask, target, context)
            real_logits = discriminator(torch.cat((condition, real), dim=1))[mask]
            fake_logits = discriminator(torch.cat((condition, fake.detach()), dim=1))[mask]
            discriminator_optimizer.zero_grad()
            discriminator_error = (adversarial_loss(real_logits, torch.ones_like(real_logits))
                                   + adversarial_loss(fake_logits, torch.zeros_like(fake_logits)))
            discriminator_error.backward()
            discriminator_optimizer.step()

            discriminator.requires_grad_(False)
            fake_logits = discriminator(torch.cat((condition, fake), dim=1))[mask]
            gan_error = (proposal - target)[mask].square().mean()
            gan_error = gan_error + 0.1 * adversarial_loss(fake_logits, torch.ones_like(fake_logits))
            generator_optimizer.zero_grad()
            vae_error = vae.loss(condition, target, mask)
            (vae_error + gan_error).backward()
            generator_optimizer.step()
            discriminator.requires_grad_(True)
            epoch_losses += len(target) * np.array([
                float(value.detach()) for value in
                (direct_error, flow_error, diffusion_error, vae_error, gan_error, discriminator_error)
            ])
            seen += len(target)
        if not np.isfinite(epoch_losses).all():
            raise FloatingPointError(f"Nonfinite training loss at epoch {epoch + 1}")
        if epoch_callback is not None:
            epoch_callback(epoch + 1, epoch_losses / seen)
        if model_callback is not None and model_callback(epoch + 1, (direct, flow, diffusion, vae, gan)):
            break
    return direct.eval(), flow.eval(), diffusion.eval(), vae.eval(), gan.eval(), GaussianProcess(
        np.stack([preprocess(raw) for raw in waves])
    )


def peak_stats(reference, output, mask, tolerance=0.15, prominence=0.35):
    settings = dict(distance=round(0.28 * RATE), prominence=prominence)
    truth = [p for p in find_peaks(reference, **settings)[0] if mask[p]]
    predicted = [p for p in find_peaks(output, **settings)[0] if mask[p]]
    i = j = matches = error = 0
    while i < len(truth) and j < len(predicted):
        delta = predicted[j] - truth[i]
        if abs(delta) <= round(tolerance * RATE):
            error += abs(delta)
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    return error, matches, len(truth) - matches, len(predicted) - matches


@torch.inference_mode()
def evaluate(models, examples, device, steps, seed, window_callback=None):
    direct, flow, diffusion, vae, gan, gp = models
    names = ("linear", "unet", "flow", "diffusion", "cvae", "cgan", "gp")
    totals = {name: np.zeros(5) for name in names}
    index = np.arange(WINDOW)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    window_index = 0
    for target, context, mask, scales in DataLoader(examples, batch_size=32):
        context_device, mask_device = context.to(device), mask.to(device)
        condition = torch.cat((context_device, mask_device), dim=1)
        deterministic = torch.where(mask_device, direct(condition), context_device)
        noise = torch.randn(context_device.shape, device=device, generator=generator)
        state = torch.where(mask_device, noise, context_device)
        diffused = state.clone()
        latent = torch.randn((len(target), vae.latent_size), device=device, generator=generator)
        variational = torch.where(mask_device, vae.sample(condition, latent), context_device)
        adversarial = torch.where(mask_device, gan(torch.cat((condition, noise), dim=1)), context_device)
        for step in range(steps):
            time = torch.full_like(state, step / steps)
            velocity = flow(torch.cat((condition, state, time), dim=1))
            state = torch.where(mask_device, state + velocity / steps, context_device)
            # Deterministic DDIM update in the v-parameterization.
            diffusion_time = torch.full_like(diffused, 1 - step / steps)
            predicted_v = diffusion(torch.cat((condition, diffused, diffusion_time), dim=1))
            delta = math.pi / (2 * steps)
            diffused = torch.where(
                mask_device,
                math.cos(delta) * diffused - math.sin(delta) * predicted_v,
                context_device,
            )
        for reference, observed, gap, scale, unet, generated, denoised, decoded, synthesized in zip(
            target[:, 0].numpy(), context[:, 0].numpy(), mask[:, 0].numpy(),
            scales.numpy(), deterministic[:, 0].cpu().numpy(), state[:, 0].cpu().numpy(),
            diffused[:, 0].cpu().numpy(), variational[:, 0].cpu().numpy(),
            adversarial[:, 0].cpu().numpy(),
        ):
            linear = np.interp(index, index[~gap], observed[~gap])
            gaussian = gp.sample(observed, gap, rng)
            for name, output in zip(names, (linear, unet, generated, denoised, decoded, synthesized, gaussian)):
                mae = np.abs(output[gap] - reference[gap]).mean() * scale
                metrics = (float(mae), *peak_stats(reference, output, gap))
                if not np.isfinite(metrics).all():
                    raise FloatingPointError(f"Nonfinite evaluation for {name}, window {window_index}")
                totals[name] += metrics
                if window_callback is not None:
                    window_callback(window_index, int(np.flatnonzero(gap)[0]), name, metrics)
            window_index += 1
    for name, (mae, timing, matches, missed, extra) in totals.items():
        ms = 1000 * timing / (matches * RATE) if matches else ""
        yield name, len(examples), mae / len(examples), ms, int(matches), int(missed), int(extra)


def smoke_data():
    rng = np.random.default_rng(22)
    t = np.arange((WINDOW + 2 * PADDING) * DECIMATION) / (RATE * DECIMATION)
    return {
        str(subject): np.stack([
            np.sin(2 * np.pi * (1 + 0.05 * subject) * t + phase)
            + 0.15 * np.sin(4 * np.pi * t + phase)
            + rng.normal(0, 0.02, len(t))
            for phase in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        ]).astype(np.float32)
        for subject in range(3)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test == (args.data is not None):
        parser.error("Choose exactly one of --data and --smoke-test.")
    if min(args.epochs, args.steps) < 1:
        parser.error("--epochs and --steps must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed > 2**32 // 1000 - 1 for seed in args.seeds):
        parser.error("--seeds must be distinct nonnegative integers below 4294967.")
    if args.data is not None and not args.data.is_dir():
        parser.error("--data must be an extracted dataset directory.")
    subjects = smoke_data() if args.smoke_test else load_trials(args.data)
    if len(subjects) < 3:
        parser.error("Need usable baseline/rest recordings from at least three subjects.")
    print("Windows per subject:", {key: len(value) for key, value in subjects.items()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import scipy
    manifest_path = args.output.with_suffix(".run.json")
    metadata = {
        "status": "running", "command": sys.argv, "epochs": args.epochs,
        "steps": args.steps, "seeds": args.seeds, "batch_size": 32,
        "architecture": "dilated_unet_v2", "dilations": DILATIONS,
        "training_seed_rule": "1000 * run_seed + zero_based_fold",
        "sampling_seed_rule": "1000 * run_seed + zero_based_fold",
        "test_mask_seed_rule": "42 + zero_based_fold; fixed across runs",
        "windows_per_subject": {key: len(value) for key, value in subjects.items()},
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
        "determinism": "Seeds fixed; deterministic cuDNN; bitwise GPU reproducibility not guaranteed",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(args.data.rglob("*_bioz.csv"))} if args.data else {},
        "completed_folds": [],
    }
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.perf_counter()
    with (args.output.open("w", newline="", encoding="utf-8") as stream,
          args.output.with_suffix(".windows.csv").open("w", newline="", encoding="utf-8") as windows_stream,
          args.output.with_suffix(".training.csv").open("w", newline="", encoding="utf-8") as training_stream):
        writer = csv.writer(stream)
        writer.writerow(("dataset", "seed", "held_out_subject", "gap_seconds", "method", "windows",
                         "mae_mohm_or_smoke_units", "peak_timing_mae_ms", "matched_peaks",
                         "missed_peaks", "extra_peaks"))
        window_writer = csv.writer(windows_stream)
        window_writer.writerow(("seed", "held_out_subject", "gap_seconds", "window_index", "gap_start", "method",
                                "mae", "timing_error_samples", "matched_peaks", "missed_peaks", "extra_peaks"))
        training_writer = csv.writer(training_stream)
        training_writer.writerow(("seed", "held_out_subject", "epoch", "unet_mse", "flow_mse", "diffusion_mse",
                                  "cvae_objective", "cgan_objective", "discriminator_objective"))
        for run_seed in args.seeds:
            for fold, (subject, test) in enumerate(subjects.items()):
                fold_seed = 1000 * run_seed + fold
                torch.manual_seed(fold_seed)
                training = np.concatenate([waves for key, waves in subjects.items() if key != subject])

                def log_epoch(epoch, losses):
                    training_writer.writerow((run_seed, subject, epoch, *losses))
                    training_stream.flush()

                models = train(training, args.epochs, device, np.random.default_rng(fold_seed), log_epoch)
                for seconds in GAPS:
                    examples = make_examples(test, np.random.default_rng(42 + fold), seconds)

                    def log_window(index, start, name, metrics):
                        window_writer.writerow((run_seed, subject, seconds, index, start, name, *metrics))

                    for metrics in evaluate(models, examples, device, args.steps, seed=fold_seed,
                                            window_callback=log_window):
                        writer.writerow(("synthetic_smoke" if args.smoke_test else "physionet",
                                         run_seed, subject, seconds, *metrics))
                stream.flush()
                windows_stream.flush()
                metadata["completed_folds"].append([run_seed, subject])
                metadata["elapsed_seconds"] = time.perf_counter() - started
                manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                print(f"Finished seed {run_seed}, held-out subject {subject} on {device}", flush=True)
                del models
        metadata["status"] = "complete"
        metadata["elapsed_seconds"] = time.perf_counter() - started
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
