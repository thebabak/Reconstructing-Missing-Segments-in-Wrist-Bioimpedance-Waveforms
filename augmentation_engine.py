"""Numerically checked batching and early-stop execution for the augmentation sweep."""
import math
import weakref
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import resample_poly, sosfiltfilt
from bioz_inpainting import RATE, WINDOW, GAPS, PADDING, DECIMATION, BANDPASS, UNet, ConditionalVAE
from validated_benchmark import CNN_NAMES

_targets = {}


def preprocess_batch(raw):
    return sosfiltfilt(BANDPASS, resample_poly(raw, 1, DECIMATION, axis=1), axis=1)[
        :, PADDING:PADDING+WINDOW].astype(np.float32)


def make_examples(waves, rng, seconds=None):
    identity = id(waves)
    entry = _targets.get(identity)
    if entry is None or entry[0]() is not waves:
        target = preprocess_batch(waves)
        reference = weakref.ref(waves, lambda unused, k=identity: _targets.pop(k, None))
        _targets[identity] = (reference, target)
    else:
        target = entry[1]
    filled = waves.copy()
    masks = np.zeros((len(waves), WINDOW), dtype=bool)
    for row, raw in enumerate(waves):
        length = round(RATE*(seconds if seconds is not None else rng.choice(GAPS)))
        start = rng.integers(RATE, WINDOW-length-RATE+1)
        masks[row, start:start+length] = True
        a = (PADDING+start)*DECIMATION; b = a+length*DECIMATION
        filled[row, a:b] = np.linspace(raw[a-1], raw[b], b-a+2)[1:-1]
    visible = preprocess_batch(filled)
    targets, contexts, scales = [], [], []
    for wave, context, mask in zip(target, visible, masks):
        scale = max(float(context[~mask].std()), 1e-6)
        center = context[~mask].mean()
        targets.append((wave-center)/scale)
        contexts.append(np.where(mask, 0, (context-center)/scale))
        scales.append(scale)
    return TensorDataset(torch.from_numpy(np.stack(targets)[:,None,:]),
                         torch.from_numpy(np.stack(contexts)[:,None,:]),
                         torch.from_numpy(masks[:,None,:]), torch.tensor(scales, dtype=torch.float32))


def train(waves, epochs, device, rng, epoch_callback=None, model_callback=None, active=None):
    """Skip independent models after stopping/capture, preserving random draws."""
    direct, flow = UNet(2).to(device), UNet(4).to(device)
    diffusion = UNet(4).to(device)
    optimizer = torch.optim.AdamW([*direct.parameters(), *flow.parameters(), *diffusion.parameters()], lr=.001)
    vae = ConditionalVAE().to(device); gan = UNet(3).to(device); discriminator = UNet(3).to(device)
    generator_optimizer = torch.optim.AdamW([*vae.parameters(), *gan.parameters()], lr=.001)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=.001)
    adversarial = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs+1):
        enabled = {name: active(name, epoch) if active else True for name in CNN_NAMES}
        totals = {}; seen = 0
        for target, context, mask, _ in DataLoader(make_examples(waves, rng), batch_size=32, shuffle=True):
            target, context, mask = (x.to(device) for x in (target, context, mask))
            condition = torch.cat((context, mask), dim=1)
            t = torch.rand((len(target),1,1), device=device); noise = torch.randn_like(target)
            losses = {}
            if enabled['flow']:
                state = torch.where(mask, torch.lerp(noise, target, t), context)
                velocity = flow(torch.cat((condition, state, t.expand_as(state)), dim=1))
            if enabled['unet']:
                losses['unet'] = (direct(condition)-target)[mask].square().mean()
            if enabled['flow']:
                losses['flow'] = (velocity-(target-noise))[mask].square().mean()
            if enabled['diffusion']:
                angle = t*(math.pi/2); alpha, sigma = angle.cos(), angle.sin()
                noisy = torch.where(mask, alpha*target+sigma*noise, context)
                velocity = diffusion(torch.cat((condition, noisy, t.expand_as(noisy)), dim=1))
                losses['diffusion'] = (velocity-(alpha*noise-sigma*target))[mask].square().mean()
            optimizer.zero_grad()
            terms = [losses[m] for m in ('unet','flow','diffusion') if m in losses]
            if terms:
                objective = terms[0]
                for term in terms[1:]: objective = objective + term
                objective.backward(); optimizer.step()
            if enabled['cgan']:
                proposal = gan(torch.cat((condition, noise), dim=1))
                fake = torch.where(mask, proposal, context); real = torch.where(mask, target, context)
                real_logits = discriminator(torch.cat((condition, real), dim=1))[mask]
                fake_logits = discriminator(torch.cat((condition, fake.detach()), dim=1))[mask]
                discriminator_optimizer.zero_grad()
                discriminator_error = adversarial(real_logits, torch.ones_like(real_logits)) + adversarial(fake_logits, torch.zeros_like(fake_logits))
                discriminator_error.backward(); discriminator_optimizer.step()
                discriminator.requires_grad_(False)
                fake_logits = discriminator(torch.cat((condition, fake), dim=1))[mask]
                losses['cgan'] = (proposal-target)[mask].square().mean() + .1*adversarial(fake_logits, torch.ones_like(fake_logits))
                losses['discriminator'] = discriminator_error
            generator_optimizer.zero_grad()
            if enabled['cvae']:
                losses['cvae'] = vae.loss(condition, target, mask)
            else:
                # The original loop draws a VAE latent after GAN's forward pass.
                # Preserve that random-stream position for all still-active models.
                torch.randn((len(target), vae.latent_size), device=device, dtype=target.dtype)
            terms = [losses[m] for m in ('cvae','cgan') if m in losses]
            if terms:
                objective = terms[0]
                for term in terms[1:]: objective = objective + term
                objective.backward(); generator_optimizer.step()
            discriminator.requires_grad_(True)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.) + len(target)*float(value.detach())
            seen += len(target)
        values = {name: value/seen for name, value in totals.items()}
        assert all(np.isfinite(value) for value in values.values())
        if epoch_callback:
            epoch_callback(epoch, values)
        if model_callback and model_callback(epoch, (direct,flow,diffusion,vae,gan)):
            break
    return direct.eval(), flow.eval(), diffusion.eval(), vae.eval(), gan.eval()
