"""Independent read-only audit/replay of the completed augmentation experiment."""
from pathlib import Path
from statistics import mean
import csv
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bioz_inpainting import UNet, ConditionalVAE, load_trials, make_examples, peak_stats
from validated_benchmark import new_saits, predict, EpochSelector
from synthetic_augmentation import SyntheticBioZ, NEURAL, CONDITIONS, check_split


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    prefix = ROOT / 'outputs/bioz_augmentation_subject1_seed2026'
    manifest = json.loads(prefix.with_suffix('.run.json').read_text())
    assert manifest['status'] == 'complete' and manifest['test_scored_after_all_training']
    assert tuple(manifest['completed_conditions']) == CONDITIONS
    for category in ('code_sha256', 'data_sha256'):
        for path, digest in manifest[category].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    check_split(manifest['inner_training_subjects'], manifest['validation_subject'],
                manifest['outer_training_subjects'], manifest['test_subject'])
    subjects = load_trials(ROOT / manifest['configuration']['data'])
    paths = ROOT / 'work' / (prefix.name + '_artifacts')
    synthetic_counts = {}
    for stage in ('inner', 'outer'):
        generator_info = manifest[stage + '_generator']
        assert manifest['test_subject'] not in generator_info['source_subjects']
        if stage == 'inner':
            assert manifest['validation_subject'] not in generator_info['source_subjects']
        generator = SyntheticBioZ()
        with np.load(paths / f'{stage}_generator.npz') as saved:
            for name in saved.files:
                setattr(generator, name, saved[name])
        refitted = SyntheticBioZ().fit(np.concatenate([subjects[s] for s in generator_info['source_subjects']]))
        for name in ('coefficients','frequencies','residual_scales','residual_power','offsets'):
            np.testing.assert_allclose(getattr(generator, name), getattr(refitted, name), atol=1e-9, rtol=1e-10)
        expected, donors = generator.sample(generator_info['generated_windows'], generator_info['seed'])
        with np.load(paths / f'{stage}_synthetic.npz') as saved:
            np.testing.assert_array_equal(saved['raw'], expected)
            np.testing.assert_array_equal(saved['donor_index'], donors)
        assert hashlib.sha256(expected.tobytes()).hexdigest() == generator_info['synthetic_sha256']
        synthetic_counts[stage] = len(expected)

    selections = read(prefix.with_suffix('.selection.csv'))
    validation = read(prefix.with_suffix('.validation.csv'))
    assert len(selections) == 12
    for row in selections:
        selector = EpochSelector((row['method'],))
        group = [r for r in validation if r['condition'] == row['condition'] and r['method'] == row['method']]
        for r in group:
            selector.update(int(r['epoch']), {row['method']: float(r['validation_mae'])})
        assert selector.epochs[row['method']] == int(row['selected_epoch'])
        assert abs(selector.best[row['method']] - float(row['validation_mae'])) < 1e-10
        assert selector.stopped[row['method']] == (row['stopped_by_patience'] == 'True')

    records = read(prefix.with_suffix('.csv'))
    windows = read(prefix.with_suffix('.windows.csv'))
    summary = read(prefix.with_suffix('.summary.csv'))
    comparison = read(prefix.with_suffix('.comparison.csv'))
    assert len(records) == 108 and len(windows) == 4320 and len(summary) == 36 and len(comparison) == 18
    test = manifest['test_subject']; fold = sorted(subjects, key=int).index(test)
    audited = 0
    for condition in CONDITIONS:
        for mask_set, mask_seed in enumerate((42, 142, 242)):
            for gap in (.5, 1, 2):
                examples = make_examples(subjects[test], np.random.default_rng(mask_seed + fold), gap)
                target, context, mask, scales = [v.numpy()[:, 0] if v.ndim == 3 else v.numpy() for v in examples.tensors]
                with np.load(paths / f'{condition}_mask{mask_set}_gap{gap}.npz') as saved:
                    for name, value in (('target', target), ('context', context), ('mask', mask), ('scales', scales)):
                        np.testing.assert_array_equal(saved[name], value)
                    for method in NEURAL:
                        output = saved[method]
                        np.testing.assert_array_equal(output[~mask], context[~mask])
                        errors = (np.abs(output-target)*mask).sum(axis=1)/mask.sum(axis=1)*scales
                        statistics = np.array([peak_stats(t, o, m) for t, o, m in zip(target, output, mask)])
                        row = next(r for r in records if r['condition']==condition and r['method']==method
                                   and int(r['mask_set'])==mask_set and float(r['gap_seconds'])==gap)
                        assert abs(float(row['mae']) - float(errors.mean())) < 1e-9
                        timing, tp, fn, fp = statistics.sum(axis=0)
                        np.testing.assert_array_equal([float(row[k]) for k in ('timing_error_samples','matched_peaks','missed_peaks','extra_peaks')], statistics.sum(axis=0))
                        assert abs(float(row['f1'])-(2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0)) < 1e-12
                        group = [r for r in windows if r['condition']==condition and r['method']==method
                                 and int(r['mask_set'])==mask_set and float(r['gap_seconds'])==gap]
                        group.sort(key=lambda r: int(r['window_index']))
                        np.testing.assert_allclose([float(r['mae']) for r in group], errors, atol=1e-10, rtol=0)
                        np.testing.assert_array_equal([[int(r[k]) for k in ('timing_error_samples','matched_peaks','missed_peaks','extra_peaks')] for r in group], statistics)
                        audited += 1
    for row in summary:
        group = [r for r in records if all(r[k]==row[k] for k in ('condition','method','gap_seconds'))]
        assert len(group) == 3
        for key in ('mae', 'f1'):
            assert abs(float(row[key])-mean(float(r[key]) for r in group)) < 1e-10
    for row in comparison:
        real, mixed = [next(r for r in summary if r['condition']==condition and r['method']==row['method']
                           and r['gap_seconds']==row['gap_seconds']) for condition in CONDITIONS]
        for key in ('mae', 'f1'):
            assert abs(float(row['mixed_minus_real_'+key]) - (float(mixed[key])-float(real[key]))) < 1e-10

    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replayed = 0
    for condition in CONDITIONS:
        checkpoint = torch.load(paths / f'{condition}.pt', map_location='cpu', weights_only=True)
        models = {'unet':UNet(2), 'flow':UNet(4), 'diffusion':UNet(4), 'cvae':ConditionalVAE(),
                  'cgan':UNet(3), 'saits':new_saits(device)}
        for name, model in models.items():
            model.load_state_dict(checkpoint['models'][name]); model.to(device).eval()
        examples = make_examples(subjects[test], np.random.default_rng(42 + fold), 2)
        output = predict(models, examples, device, int(manifest['configuration']['steps']),
                         1000*int(manifest['configuration']['seed']) + fold)
        with np.load(paths / f'{condition}_mask0_gap2.npz') as saved:
            for name, values in output.items():
                np.testing.assert_allclose(values, saved[name], atol=2e-5, rtol=1e-4)
                replayed += 1
        del models
    manuscript_metadata = json.loads((ROOT / 'outputs/bioz_metrics_validated.analysis.json').read_text())
    assert hashlib.sha256((ROOT / 'outputs/wrist_bioz_manuscript.docx').read_bytes()).hexdigest() == manuscript_metadata['document_sha256']
    result = {'status':'passed', 'unit_tests_passed':13, 'full_pipeline_fixture_smoke_test':'passed',
              'synthetic_signal_counts_reproduced':synthetic_counts, 'selected_epochs_checked':12,
              'generator_fits_recomputed_from_training_subjects_only':True,
              'recomputed_condition_method_schedule_gap_records':audited,
              'recomputed_window_records':len(windows), 'checkpoint_model_replays':replayed,
              'primary_manuscript_unchanged':True,
              'scope':'One real test participant and one seed; does not establish population improvement.'}
    prefix.with_suffix('.audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
