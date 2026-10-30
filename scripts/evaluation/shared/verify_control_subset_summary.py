#!/usr/bin/env python3
"""Independent standard-library check of the exported four-model aggregates."""
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


def main(root):
    with (root / 'summary_mean_sd.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    metrics = sorted({r['metric'] for r in summary})
    groups = defaultdict(list)
    seen = set()
    with gzip.open(root / 'per_condition_all_runs.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            identity = (row['model'], row['run'], row['condition_index'])
            assert identity not in seen
            seen.add(identity)
            for scope in ['all272'] + (['multi255'] if int(row['n_true_clusters']) > 1 else []):
                for metric in metrics:
                    value = float(row[metric])
                    assert math.isfinite(value)
                    groups[(row['model'], int(row['n_controls']), scope, metric, row['run'])].append(value)
    runs = defaultdict(list)
    for (model, size, scope, metric, run), values in groups.items():
        assert len(values) == (272 if scope == 'all272' else 255)
        runs[(model, size, scope, metric)].append(statistics.mean(values))
    assert len(seen) == 11968
    assert len(summary) == 648 and len(metrics) == 27
    max_difference = 0.0
    for row in summary:
        key = (row['model'], int(row['n_controls']), row['scope'], row['metric'])
        values = runs[key]
        assert len(values) == int(row['n_repeats'])
        reference = runs[(row['model'], 10691, row['scope'], row['metric'])][0]
        checks = [(statistics.mean(values), float(row['mean'])),
                  (reference, float(row['full_reference'])),
                  (statistics.mean(values) - reference, float(row['difference_from_full']))]
        if len(values) > 1:
            checks.append((statistics.stdev(values), float(row['sd_across_draws'])))
        else:
            assert not row['sd_across_draws']
        for a, b in checks:
            difference = abs(a-b)
            max_difference = max(max_difference, difference)
            assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (key, a, b)
    report = dict(valid=True, independent_implementation='Python standard library csv/statistics',
        condition_records=len(seen), aggregate_rows=len(summary), metrics=len(metrics),
        max_absolute_difference=max_difference)
    (root / 'independent_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
