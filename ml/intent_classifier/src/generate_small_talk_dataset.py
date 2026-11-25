import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = 'E09_07_07_03_01'
FIELDS = ['id', 'text', 'intent', 'context', 'scenario', 'style', 'context_required', 'source']
QUOTAS = {
    'GREETING': (35, 12, 3, 0),
    'GOODBYE': (30, 15, 5, 0),
    'GRATITUDE': (30, 14, 5, 1),
    'AGREEMENT': (25, 15, 8, 2),
    'DISAGREEMENT': (25, 15, 8, 2),
    'UNCERTAINTY': (25, 15, 8, 2),
    'ACKNOWLEDGMENT': (35, 12, 3, 0),
    'AFFIRMATIVE_ANSWER': (30, 14, 5, 1),
    'NEGATIVE_ANSWER': (30, 14, 5, 1),
    'QUESTION': (10, 20, 14, 6),
    'CLARIFICATION': (15, 18, 12, 5),
    'STATEMENT': (10, 16, 14, 10),
}
STYLES = {'casual', 'neutral', 'hesitant', 'emphatic', 'fragmented', 'polite'}
SCENARIOS = {'initial_contact', 'information_presentation', 'human_evaluation', 'machine_response', 'discussion_and_clarification', 'evidence_presentation', 'final_reaction', 'conversation_closing'}
SOURCE = 'digital_mirror_small_talk_v1'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def diagnostic_normalize(text):
    return ' '.join(re.findall(r"\b\w+(?:'\w+)*\b", text.lower().replace('’', "'")))


def length_bucket(text):
    n = len(text.split())
    return 0 if n <= 3 else 1 if n <= 6 else 2 if n <= 10 else 3


def assemble(batch_dir, output):
    paths = [batch_dir / f'b{i:02d}_{intent.lower()}.json' for i, intent in enumerate(QUOTAS, 1)]
    require(set(batch_dir.glob('*.json')) == set(paths), 'Unexpected or missing batch files')
    hashes = {p: sha256(p) for p in paths}
    rows = []
    print(f'=== {EXPERIMENT}: DATASET ASSEMBLY ===')
    try:
        for path, (intent, quota) in zip(paths, QUOTAS.items()):
            batch = json.loads(path.read_text(encoding='utf-8'))
            require(isinstance(batch, list) and len(batch) == 50, f'{path.name}: invalid batch size')
            for row in batch:
                require(isinstance(row, dict) and set(row) == set(FIELDS), 'Invalid schema')
                require(all(isinstance(row[k], str) for k in FIELDS if k != 'context_required'), 'Invalid string field')
                require(type(row['context_required']) is bool, 'Invalid boolean field')
                require(row['id'] == f'NST-{len(rows)+1:04d}', 'Invalid ID sequence')
                require(row['intent'] == intent, 'Incorrect intent')
                require(row['source'] == SOURCE, 'Incorrect source')
                require(bool(diagnostic_normalize(row['text'])), 'Empty text')
                require(row['style'] in STYLES and row['scenario'] in SCENARIOS, 'Invalid metadata')
                rows.append(row)
            counts = Counter(length_bucket(r['text']) for r in batch)
            actual = tuple(counts[i] for i in range(4))
            require(actual == quota, f'{path.name}: length quota mismatch {actual}')
            print(f'{path.name}: PASS | 50 records | lengths={actual}')
        require(len(rows) == 600, 'Incorrect total')
        require(len({r['id'] for r in rows}) == 600, 'Duplicate IDs')
        groups = defaultdict(list)
        for row in rows:
            groups[diagnostic_normalize(row['text'])].append(row)
        duplicates = {k: v for k, v in groups.items() if len(v) > 1}
        conflicts = {k: v for k, v in duplicates.items() if len({r['intent'] for r in v}) > 1}
        print('\n=== DISTRIBUTION ===')
        print('Records:', len(rows))
        print('Class counts:', dict(Counter(r['intent'] for r in rows)))
        print('Length counts (1-3, 4-6, 7-10, 11+):', tuple(sum(length_bucket(r['text']) == i for r in rows) for i in range(4)))
        print('Styles:', dict(Counter(r['style'] for r in rows)))
        print('Scenarios:', dict(Counter(r['scenario'] for r in rows)))
        print('Context present:', sum(bool(r['context'].strip()) for r in rows))
        print('Context required:', sum(r['context_required'] for r in rows))
        print('\n=== DUPLICATE DIAGNOSTICS ===')
        print('Unique normalized texts:', len(groups))
        print('Duplicate groups:', len(duplicates))
        print('Conflicting groups:', len(conflicts))
        for key, group in duplicates.items():
            print(repr(key), [(r['id'], r['intent']) for r in group])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with output.open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            require(reader.fieldnames == FIELDS, 'CSV schema mismatch')
            loaded = list(reader)
        require(loaded == [{k: str(r[k]) for k in FIELDS} for r in rows], 'CSV reload mismatch')
        print('\nCSV reload: PASS')
        print('Output:', output)
        print('Output SHA-256:', sha256(output))
        print('No records removed. No model trained.')
    finally:
        unchanged = sum(sha256(p) == h for p, h in hashes.items())
        print(f'Input integrity: {unchanged}/{len(paths)}')
        require(unchanged == len(paths), 'Input integrity failure')
    print('ASSEMBLY: PASS')


if __name__ == '__main__':
    raw_dir = ML_ROOT / 'data' / 'raw' / 'digital_mirror_small_talk'
    assemble(raw_dir / 'batches', raw_dir / 'small_talk_intents_v1.csv')
