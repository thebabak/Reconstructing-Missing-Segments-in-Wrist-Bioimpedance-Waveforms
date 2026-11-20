"""One-subject real-data test of training-only synthetic BioZ augmentation.

Synthetic signals are mathematical surrogates, not new measured participants.
The unchanged validated benchmark remains the manuscript's primary experiment.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from bioz_inpainting import (RATE, WINDOW, PADDING, DECIMATION, GAPS, load_trials,
                            preprocess, make_examples, train, peak_stats)
from validated_benchmark import (CNN_NAMES, EpochSelector, subject_split,
                                 validation_scores, train_saits, predict)

NATIVE_RATE = RATE * DECIMATION
NATIVE_LENGTH = (WINDOW + 2 * PADDING) * DECIMATION
NEURAL = (*CNN_NAMES, 'saits')
CONDITIONS = ('real_only_matched', 'real_plus_synthetic')


class SyntheticBioZ:
    """Training-fitted harmonic surrogates with a colored stochastic residual."""
    harmonics = 4

    def fit(self, raw_waves):
        waves = np.stack([preprocess(raw) for raw in raw_waves]).astype(np.float64)
        t = np.arange(WINDOW) / RATE
        frequencies = np.arange(.7, 3.0001, .025)
        designs = []
        for frequency in frequencies:
            angle = 2 * np.pi * frequency * t
            design = np.column_stack([np.ones(WINDOW), *[
                value for k in range(1, self.harmonics + 1)
                for value in (np.sin(k * angle), np.cos(k * angle))]])
            designs.append(design)
        coefficients, fitted_frequencies, residuals = [], [], []
        for wave in waves:
            fits = [np.linalg.solve(x.T @ x + 1e-4 * np.eye(x.shape[1]), x.T @ wave)
                    for x in designs]
            errors = [np.mean((x @ beta - wave) ** 2) for x, beta in zip(designs, fits)]
            index = int(np.argmin(errors))
            coefficients.append(fits[index][1:])
            fitted_frequencies.append(frequencies[index])
            residuals.append(wave - designs[index] @ fits[index])
        residuals = np.asarray(residuals)
        self.coefficients = np.asarray(coefficients)
        self.frequencies = np.asarray(fitted_frequencies)
        self.residual_scales = residuals.std(axis=1)
        self.residual_power = np.mean(np.abs(np.fft.rfft(residuals, axis=1)) ** 2, axis=0)
        self.offsets = np.asarray(raw_waves).mean(axis=1)
        assert np.isfinite(self.coefficients).all()
        return self

    def sample(self, count, seed):
        rng = np.random.default_rng(seed)
        t = np.arange(NATIVE_LENGTH) / NATIVE_RATE
        fft_frequency = np.fft.rfftfreq(NATIVE_LENGTH, 1 / NATIVE_RATE)
        power = np.interp(fft_frequency, np.fft.rfftfreq(WINDOW, 1 / RATE), self.residual_power)
        power[(fft_frequency < .5) | (fft_frequency > 12)] = 0
        donors = rng.integers(0, len(self.frequencies), count)
        output = []
        for donor in donors:
            frequency = np.clip(self.frequencies[donor] * np.exp(rng.normal(0, .04)), .7, 3)
            phase = (2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi)
                     + .05 * np.sin(2 * np.pi * rng.uniform(.08, .25) * t + rng.uniform(0, 2*np.pi)))
            envelope = np.exp(rng.normal(0, .12)) * (
                1 + .1 * np.sin(2*np.pi*rng.uniform(.08, .3)*t + rng.uniform(0, 2*np.pi)))
            signal = np.zeros_like(t)
            for k, (sine, cosine) in enumerate(self.coefficients[donor].reshape(-1, 2), 1):
                signal += np.exp(rng.normal(0, .06)) * (sine*np.sin(k*phase) + cosine*np.cos(k*phase))
            spectrum = np.sqrt(power) * (rng.normal(size=len(power)) + 1j*rng.normal(size=len(power)))
            spectrum[0] = 0; spectrum[-1] = spectrum[-1].real
            residual = np.fft.irfft(spectrum, n=NATIVE_LENGTH)
            residual *= self.residual_scales[donor] * rng.uniform(.8, 1.2) / max(residual.std(), 1e-12)
            output.append(self.offsets[donor] + envelope * signal + residual)
        values = np.asarray(output, dtype=np.float64)
        assert values.shape == (count, NATIVE_LENGTH) and np.isfinite(values).all()
        return values, donors

    def save(self, path):
        np.savez_compressed(path, coefficients=self.coefficients, frequencies=self.frequencies,
                            residual_scales=self.residual_scales, residual_power=self.residual_power,
                            offsets=self.offsets)


def training_set(real, synthetic, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    assert real.shape == synthetic.shape
    # Both conditions have the same epoch size and maximum optimizer-step budget.
    return np.concatenate((real, real if condition == 'real_only_matched' else synthetic))


def check_split(inner_ids, validation_id, outer_ids, test_id):
    assert test_id not in outer_ids and validation_id != test_id
    assert validation_id not in inner_ids and test_id not in inner_ids
    assert set(inner_ids + [validation_id]) == set(outer_ids)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def train_condition(condition, inner, outer, validation, seed, fold, args, paths, log):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_seed = 1000 * seed + fold
    selector = EpochSelector(CNN_NAMES)

    def training_logger(stage):
        def write(epoch, values):
            values = (values,) if np.isscalar(values) else values
            names = ('saits',) if len(values) == 1 else (*CNN_NAMES, 'discriminator')
            for method, value in zip(names, values):
                log('training', (condition, stage, epoch, method, value))
        return write

    def select(epoch, models):
        if epoch % args.check_every and epoch != args.max_epochs:
            return False
        scores = validation_scores(dict(zip(CNN_NAMES, models)), validation, device, args.steps, 70000 + fold)
        done = selector.update(epoch, scores)
        for method, score in scores.items():
            log('validation', (condition, epoch, method, score, int(selector.stopped[method])))
        print(f'{condition}: inner CNN epoch {epoch}; ' + ', '.join(
            f'{m}={scores[m]:.2f}' for m in CNN_NAMES), flush=True)
        return done

    torch.manual_seed(training_seed)
    models = train(inner, args.max_epochs, device, np.random.default_rng(training_seed),
                   training_logger('inner'), select)
    del models
    saits_selector = EpochSelector(('saits',))

    def select_saits(epoch, model):
        if epoch % args.check_every and epoch != args.max_epochs:
            return False
        scores = validation_scores({'saits': model}, validation, device, args.steps, 70000 + fold)
        done = saits_selector.update(epoch, scores)
        log('validation', (condition, epoch, 'saits', scores['saits'], int(done)))
        print(f'{condition}: inner SAITS epoch {epoch}; MAE={scores["saits"]:.2f}', flush=True)
        return done

    saits = train_saits(inner, args.max_epochs, device, training_seed + 100000,
                        select_saits, training_logger('inner'))
    del saits
    epochs = {**selector.epochs, **saits_selector.epochs}
    selection = []
    for method in NEURAL:
        rule = saits_selector if method == 'saits' else selector
        record = {'condition': condition, 'method': method, 'selected_epoch': epochs[method],
                  'validation_mae': rule.best[method], 'stopped_by_patience': bool(rule.stopped[method])}
        selection.append(record)
        print('SELECTED', record, flush=True)

    snapshots = {}

    def capture(epoch, models):
        for name, model in zip(CNN_NAMES, models):
            if epochs[name] == epoch:
                snapshots[name] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f'{condition}: refit CNN epoch {epoch}/{max(epochs[m] for m in CNN_NAMES)}', flush=True)
        return False

    torch.manual_seed(training_seed)
    fitted = train(outer, max(epochs[m] for m in CNN_NAMES), device, np.random.default_rng(training_seed),
                   training_logger('refit'), capture)
    models = dict(zip(CNN_NAMES, fitted[:5]))
    for name, model in models.items():
        model.load_state_dict(snapshots[name]); model.eval()
    models['saits'] = train_saits(outer, epochs['saits'], device, training_seed + 100000,
                                 lambda e, m: False, training_logger('refit'))
    checkpoint = {'models': {m: {k: v.detach().cpu() for k, v in model.state_dict().items()}
                              for m, model in models.items()}, 'selection': selection}
    torch.save(checkpoint, paths / f'{condition}.pt')
    print(f'FINISHED training {condition}', flush=True)
    return selection


def write_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('work/physionet_bioz'))
    parser.add_argument('--output', type=Path, default=Path('outputs/bioz_augmentation_subject1_seed2026'))
    parser.add_argument('--test-subject', default='1')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-epochs', type=int, default=120)
    parser.add_argument('--check-every', type=int, default=5)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()
    if min(args.max_epochs, args.check_every, args.steps) < 1:
        parser.error('Positive budgets required')
    manifest_path = args.output.with_suffix('.run.json')
    if manifest_path.exists():
        parser.error('Use a new output prefix; previous experiments are preserved')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    subjects = load_trials(args.data)
    if args.test_subject not in subjects:
        parser.error('Unknown test participant')
    inner_ids, validation_id, outer_ids = subject_split(subjects, args.test_subject, args.seed)
    check_split(inner_ids, validation_id, outer_ids, args.test_subject)
    fold = sorted(subjects, key=int).index(args.test_subject)
    inner_real = np.concatenate([subjects[s] for s in inner_ids])
    outer_real = np.concatenate([subjects[s] for s in outer_ids])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = Path('work') / (args.output.name + '_artifacts'); paths.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sources = [Path(__file__), Path('bioz_inpainting.py'), Path('validated_benchmark.py'),
               *Path('vendor/saits').glob('*.py')]
    manifest = {'status': 'running', 'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'test_subject': args.test_subject, 'validation_subject': validation_id,
                'inner_training_subjects': inner_ids, 'outer_training_subjects': outer_ids,
                'real_windows_inner': len(inner_real), 'real_windows_outer': len(outer_real),
                'test_windows': len(subjects[args.test_subject]), 'synthetic_to_real_ratio': 1,
                'methods': NEURAL, 'conditions': CONDITIONS, 'completed_conditions': [],
                'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'data_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.rglob('*_bioz.csv')},
                'synthetic_definition': 'Four-harmonic model fit to real training windows only; resampled parameter donors, random phase, frequency and amplitude perturbations, slow modulation, colored stochastic residual. Generated directly on 1250-Hz 10-s native grid; same mask-before-filter preprocessing as real data.',
                'comparison': '1:1 real/synthetic versus twice-presented real windows; equal dataset size and maximum per-epoch update budget, method-specific real-validation epoch selection.',
                'scope': 'One previously studied test participant, one run seed; exploratory augmentation experiment, not independent cohort confirmation.',
                'versions': {'torch': torch.__version__, 'numpy': np.__version__},
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    save_json(manifest_path, manifest)
    generated = {}
    for stage, real, ids, offset in (('inner', inner_real, inner_ids, 800000), ('outer', outer_real, outer_ids, 900000)):
        generator = SyntheticBioZ().fit(real)
        synthetic, donors = generator.sample(len(real), args.seed + offset)
        generator.save(paths / f'{stage}_generator.npz')
        np.savez_compressed(paths / f'{stage}_synthetic.npz', raw=synthetic, donor_index=donors)
        generated[stage] = synthetic
        manifest[f'{stage}_generator'] = {'source_subjects': ids, 'fit_windows': len(real), 'generated_windows': len(synthetic),
                                         'seed': args.seed + offset, 'synthetic_sha256': hashlib.sha256(synthetic.tobytes()).hexdigest(),
                                         'filtered_real_median_sd': float(np.median([preprocess(r).std() for r in real])),
                                         'filtered_synthetic_median_sd': float(np.median([preprocess(r).std() for r in synthetic]))}
        save_json(manifest_path, manifest)
        print('GENERATED', stage, manifest[f'{stage}_generator'], flush=True)
    validation = [make_examples(subjects[validation_id], np.random.default_rng(31415 + fold), g) for g in GAPS]
    streams = {}
    for name, header in (('training', ['condition','stage','epoch','method','objective']),
                         ('validation', ['condition','epoch','method','validation_mae','stopped'])):
        stream = args.output.with_suffix('.' + name + '.csv').open('w', newline='')
        writer = csv.writer(stream); writer.writerow(header); streams[name] = (stream, writer)

    def log(name, row):
        stream, writer = streams[name]; writer.writerow(row); stream.flush()

    selections = []
    for condition in CONDITIONS:
        selected = train_condition(condition, training_set(inner_real, generated['inner'], condition),
                                   training_set(outer_real, generated['outer'], condition), validation,
                                   args.seed, fold, args, paths, log)
        selections.extend(selected)
        write_csv(args.output.with_suffix('.selection.csv'), selections)
        manifest['completed_conditions'].append(condition)
        manifest['elapsed_seconds'] = time.perf_counter() - started
        save_json(manifest_path, manifest)
    for stream, _ in streams.values():
        stream.close()
    # No test scores are computed or exposed until both training conditions finish.
    from bioz_inpainting import UNet, ConditionalVAE
    from validated_benchmark import new_saits
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records, windows = [], []
    for condition in CONDITIONS:
        checkpoint = torch.load(paths / f'{condition}.pt', map_location='cpu', weights_only=True)
        models = {'unet': UNet(2), 'flow': UNet(4), 'diffusion': UNet(4), 'cvae': ConditionalVAE(),
                  'cgan': UNet(3), 'saits': new_saits(device)}
        for name, model in models.items():
            model.load_state_dict(checkpoint['models'][name]); model.to(device).eval()
        for mask_set, mask_seed in enumerate((42, 142, 242)):
            for gap in GAPS:
                examples = make_examples(subjects[args.test_subject], np.random.default_rng(mask_seed + fold), gap)
                target, context, mask, scales = [v.numpy()[:,0] if v.ndim == 3 else v.numpy() for v in examples.tensors]
                outputs = predict(models, examples, device, args.steps, 1000*args.seed + fold + 100*mask_set)
                np.savez_compressed(paths / f'{condition}_mask{mask_set}_gap{gap}.npz', target=target,
                                    context=context, mask=mask, scales=scales, **outputs)
                for method, output in outputs.items():
                    assert np.isfinite(output).all()
                    np.testing.assert_array_equal(output[~mask], context[~mask])
                    errors = (np.abs(output-target)*mask).sum(axis=1)/mask.sum(axis=1)*scales
                    stats = np.array([peak_stats(t, o, g) for t, o, g in zip(target, output, mask)])
                    timing, tp, fn, fp = stats.sum(axis=0)
                    records.append({'condition': condition, 'method': method, 'mask_set': mask_set,
                                    'gap_seconds': gap, 'windows': len(target), 'mae': float(errors.mean()),
                                    'timing_error_samples': int(timing), 'matched_peaks': int(tp),
                                    'missed_peaks': int(fn), 'extra_peaks': int(fp),
                                    'f1': float(2*tp/(2*tp+fn+fp)) if 2*tp+fn+fp else 0.})
                    for index, (error, values) in enumerate(zip(errors, stats)):
                        windows.append({'condition': condition, 'method': method, 'mask_set': mask_set,
                                        'gap_seconds': gap, 'window_index': index, 'mae': float(error),
                                        'timing_error_samples': int(values[0]), 'matched_peaks': int(values[1]),
                                        'missed_peaks': int(values[2]), 'extra_peaks': int(values[3])})
        del models
    write_csv(args.output.with_suffix('.csv'), records)
    write_csv(args.output.with_suffix('.windows.csv'), windows)
    summaries, comparisons = [], []
    for condition in CONDITIONS:
        for method in NEURAL:
            for gap in GAPS:
                group = [r for r in records if r['condition']==condition and r['method']==method and r['gap_seconds']==gap]
                summaries.append({'condition': condition, 'method': method, 'gap_seconds': gap,
                                  'mae': float(np.mean([r['mae'] for r in group])),
                                  'f1': float(np.mean([r['f1'] for r in group]))})
    lookup = {(r['condition'], r['method'], r['gap_seconds']): r for r in summaries}
    for method in NEURAL:
        for gap in GAPS:
            real, mixed = [lookup[c, method, gap] for c in CONDITIONS]
            comparisons.append({'method': method, 'gap_seconds': gap, 'real_only_matched_mae': real['mae'],
                                'real_plus_synthetic_mae': mixed['mae'], 'mixed_minus_real_mae': mixed['mae']-real['mae'],
                                'real_only_matched_f1': real['f1'], 'real_plus_synthetic_f1': mixed['f1'],
                                'mixed_minus_real_f1': mixed['f1']-real['f1']})
    write_csv(args.output.with_suffix('.summary.csv'), summaries)
    write_csv(args.output.with_suffix('.comparison.csv'), comparisons)
    manifest['status'] = 'complete'; manifest['elapsed_seconds'] = time.perf_counter() - started
    manifest['test_scored_after_all_training'] = True
    manifest['test_records'] = len(records); manifest['test_window_records'] = len(windows)
    save_json(manifest_path, manifest)
    print('COMPLETE', json.dumps(comparisons, indent=2), flush=True)


if __name__ == '__main__':
    main()
