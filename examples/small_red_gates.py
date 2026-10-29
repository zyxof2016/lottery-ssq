"""Exploratory causal gates over fixed small-red candidates; no automatic adoption.

Input: cargo run --release --example small_red_candidates -- data/ssq.csv
Each choice is made before appending the current outcome to the history.
"""
import collections
import hashlib
import json
import sys

source = json.load(open(sys.argv[1]))
assert source['red_sizes'] == [12, 14, 15, 16]
rows = source['rows']
trial = source['models'].index('shrink_recent50')
names = ['weighted', 'small_recent', 'guard156', 'guard312', 'two_window',
         'two_window_margin', 'last156_full_only', 'last312_full_only']
history = collections.deque(maxlen=312)
results = {name: [] for name in names}
for row in rows:
    hits = [row['hits'][0], row['hits'][trial]]

    def counts(length):
        past = list(history)[-length:]
        return [[sum(outcome[model][size] == 6 for outcome in past)
                 for size in range(4)] for model in range(2)]

    short, medium, long = counts(60), counts(156), counts(312)
    guard156 = (len(history) >= 156 and medium[1][3] >= medium[0][3] + 1
                and sum(medium[1][1:3]) >= sum(medium[0][1:3]))
    guard312 = (len(history) >= 312 and long[1][3] >= long[0][3] + 1
                and sum(long[1][1:3]) >= sum(long[0][1:3]))
    two = (len(history) >= 156 and short[1][3] >= short[0][3]
           and medium[1][3] >= medium[0][3]
           and sum(medium[1][1:4]) > sum(medium[0][1:4]))
    margin = two and (short[1][3] > short[0][3] or medium[1][3] > medium[0][3])
    choices = [False, True, guard156, guard312, two, margin,
               len(history) >= 156 and medium[1][3] > medium[0][3],
               len(history) >= 312 and long[1][3] > long[0][3]]
    for name, choose_trial in zip(names, choices):
        results[name].append(hits[int(choose_trial)])
    history.append(hits)

start = next((i for i, row in enumerate(rows) if row['issue'] >= 25111), len(rows))
end = min(start + 156, len(rows))
spans = {'all_oos': (0, len(rows)), 'development': (0, start),
         'confirmation_exploratory': (start, end), 'tracking': (end, len(rows)),
         'recent60': (max(0, len(rows) - 60), len(rows)),
         'recent156': (max(0, len(rows) - 156), len(rows))}
windows = {}
for window, (first, last) in spans.items():
    cells = []
    for model in names:
        for position, size in enumerate(source['red_sizes']):
            full = sum(hits[position] == 6 for hits in results[model][first:last])
            cells.append({'model': model, 'red_count': size, 'full_red_periods': full,
                          'n': last - first})
    windows[window] = {'n': last - first, 'cells': cells}

json.dump({'kind': 'exploratory_causal_small_red_gates', 'actual_recommendation': False,
           'automatic_promotion': False, 'input_data_sha256': source['data_sha256'],
           'input_source_sha256': source['source_sha256'],
           'source_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
           'candidate_gate_count': 6, 'models': names, 'windows': windows},
          sys.stdout, ensure_ascii=False, indent=2)
print()
