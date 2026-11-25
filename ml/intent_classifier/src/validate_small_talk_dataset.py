"""E09.07.7.3.2: read-only dataset diagnostics and masked review export."""
import csv
import hashlib
import math
import random
import re
import statistics as st
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from generate_small_talk_dataset import FIELDS, QUOTAS, STYLES, SCENARIOS, SOURCE, length_bucket

ROOT = Path(__file__).resolve().parents[1]
EXP = 'E09_07_07_03_02'
OUT = ROOT / 'experiments'
PATHS = {
    'small_talk': ROOT / 'data/raw/digital_mirror_small_talk/small_talk_intents_v1.csv',
    'digital_mirror': ROOT / 'data/raw/digital_mirror/digital_mirror_intents_v1.csv',
    'contrastive': ROOT / 'data/raw/digital_mirror_contrastive/contrastive_intents_v1.csv',
}
EXPECTED_HASH = '03c4261d33b01f2ea42634dc96bc48a6022b6006bf48ddfe6bad52fabb4a7ffd'
WINDOW = 50
SEED = 42


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(text):
    return re.findall(r"\b\w+(?:'\w+)*\b", text.lower().replace('’', "'"))


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(name, rows, fields):
    path = OUT / f'{name}.csv'
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', path.name)


def check(condition, message):
    if not condition:
        raise ValueError(message)
    print('PASS:', message)


def lexical(rows, dataset, scope):
    # Average MATTR over ten reproducible utterance-order permutations.
    docs = [tokens(r['text']) for r in rows]
    lengths = [len(d) for d in docs]
    counts = Counter(w for d in docs for w in d)
    total = sum(counts.values())
    mattrs = []
    for seed in range(SEED, SEED + 10):
        order = list(range(len(docs)))
        random.Random(seed).shuffle(order)
        words = [w for i in order for w in docs[i]]
        if len(words) >= WINDOW:
            window = Counter(words[:WINDOW])
            values = [len(window) / WINDOW]
            for i in range(WINDOW, len(words)):
                old = words[i-WINDOW]
                window[old] -= 1
                if not window[old]:
                    del window[old]
                window[words[i]] += 1
                values.append(len(window)/WINDOW)
            mattrs.append(st.mean(values))
    return dict(dataset=dataset, scope=scope, records=len(rows), tokens=total,
                vocabulary=len(counts), ttr=len(counts)/total,
                mattr_50=st.mean(mattrs) if mattrs else '',
                hapax_ratio=sum(v == 1 for v in counts.values())/len(counts),
                mean_length=st.mean(lengths), median_length=st.median(lengths),
                std_length=st.pstdev(lengths), min_length=min(lengths), max_length=max(lengths),
                short_share=sum(1 <= n <= 3 for n in lengths)/len(rows))


def patterns(rows, dataset, scope):
    result = []
    docs = [tokens(r['text']) for r in rows]
    for n in (2, 3, 4):
        for kind in ('prefix', 'ngram'):
            counts = Counter()
            for d in docs:
                if len(d) >= n:
                    grams = {tuple(d[:n])} if kind == 'prefix' else {tuple(d[i:i+n]) for i in range(len(d)-n+1)}
                    counts.update(grams)
            for gram, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                if count >= 2:
                    result.append(dict(dataset=dataset, scope=scope, kind=kind, n=n, pattern=' '.join(gram), records=count, share=count/len(rows)))
    return result


def similarities(rows, dataset):
    sets = [set(tokens(r['text'])) for r in rows]
    totals = defaultdict(Counter)
    flagged = []
    for i,j in combinations(range(len(rows)),2):
        a,b = rows[i],rows[j]
        score = len(sets[i] & sets[j])/len(sets[i] | sets[j])
        same = a['intent'] == b['intent']
        bucket = 'very_high' if score >= .9 else 'high' if score >= .7 else 'moderate' if score >= .5 else 'low'
        for scope in ('ALL', 'within_class' if same else 'between_classes'):
            totals[scope][bucket] += 1
        if score >= .7:
            flagged.append(dict(dataset=dataset,id_a=a['id'],id_b=b['id'],text_a=a['text'],text_b=b['text'],intent_a=a['intent'],intent_b=b['intent'],jaccard=score))
    summary = [dict(dataset=dataset,scope=s,pairs=sum(c.values()),**{k:c[k] for k in ('low','moderate','high','very_high')}) for s,c in totals.items()]
    return summary, flagged


def association(rows, field):
    classes = sorted({r['intent'] for r in rows})
    values = sorted({r[field] for r in rows})
    counts = Counter((r['intent'],r[field]) for r in rows)
    a = Counter(r['intent'] for r in rows)
    b = Counter(r[field] for r in rows)
    n = len(rows)
    chi = sum((counts[i,j]-a[i]*b[j]/n)**2/(a[i]*b[j]/n) for i in classes for j in values)
    v = math.sqrt(chi/(n*min(len(classes)-1,len(values)-1))) if min(len(classes),len(values))>1 else 0
    print(f'{field}: Cramer V={v:.6f} (descriptive only)')
    return [dict(intent=i, **{j:counts[i,j] for j in values}) for i in classes]


def main():
    hashes = {p:digest(p) for p in PATHS.values()}
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        datasets = {k:load(p) for k,p in PATHS.items()}
        rows = datasets['small_talk']
        check(hashes[PATHS['small_talk']] == EXPECTED_HASH, 'Assembly SHA-256')
        check(len(rows)==600 and list(rows[0])==FIELDS, 'Small talk schema and count')
        check({r['id'] for r in rows}=={f'NST-{i:04d}' for i in range(1,601)}, 'ID coverage')
        check(all(r['source']==SOURCE and r['style'] in STYLES and r['scenario'] in SCENARIOS and r['context_required'] in ('True','False') for r in rows), 'Metadata values')
        for intent,quota in QUOTAS.items():
            group = [r for r in rows if r['intent']==intent]
            check(tuple(sum(length_bucket(r['text'])==i for r in group) for i in range(4))==quota, intent+' quotas')
        stats, pats, sims, flags = [],[],[],[]
        for name,data in datasets.items():
            check(len(data)==(1200 if name=='contrastive' else 600), name+' count')
            check(len({r['id'] for r in data})==len(data) and all(tokens(r['text']) for r in data), name+' unique IDs and nonempty text')
            check(set(r['intent'] for r in data)==set(QUOTAS), name+' classes')
            for scope in ['ALL',*QUOTAS]:
                subset = data if scope=='ALL' else [r for r in data if r['intent']==scope]
                stats.append(lexical(subset,name,scope))
                pats.extend(patterns(subset,name,scope))
            summary, pairs = similarities(data,name)
            sims.extend(summary); flags.extend(pairs)
        print('\n=== DATASET COMPARISON ===')
        for r in stats:
            if r['scope']=='ALL': print(r)
        print('\n=== PAIRWISE SIMILARITY ===')
        for r in sims: print(r)
        save(EXP+'_lexical_statistics',stats,list(stats[0]))
        save(EXP+'_repeated_patterns',pats,['dataset','scope','kind','n','pattern','records','share'])
        save(EXP+'_similarity_summary',sims,list(sims[0]))
        save(EXP+'_similarity_pairs',flags,['dataset','id_a','id_b','text_a','text_b','intent_a','intent_b','jaccard'])
        for field in ('style','scenario'):
            matrix=association(rows,field)
            save(EXP+'_intent_'+field,matrix,list(matrix[0]))
        context=[]
        for intent in QUOTAS:
            group=[r for r in rows if r['intent']==intent]
            context.append(dict(intent=intent,records=len(group),present=sum(bool(r['context'].strip()) for r in group),required=sum(r['context_required']=='True' for r in group),required_missing=sum(r['context_required']=='True' and not r['context'].strip() for r in group)))
        print('\n=== CONTEXT ===')
        for r in context: print(r)
        save(EXP+'_context_statistics',context,list(context[0]))
        missing=[r for r in rows if r['context_required']=='True' and not r['context'].strip()]
        save(EXP+'_missing_context',missing,FIELDS)
        groups=defaultdict(list)
        for r in rows: groups[' '.join(tokens(r['text']))].append(r)
        duplicates=[r for g in groups.values() if len(g)>1 for r in g]
        print('Duplicate groups:',sum(len(g)>1 for g in groups.values()))
        print('Conflicting groups:',sum(len({r['intent'] for r in g})>1 for g in groups.values()))
        save(EXP+'_duplicates',duplicates,FIELDS)
        # Original IDs encode class; replace them with shuffled opaque review IDs.
        shuffled=rows.copy(); random.Random(SEED).shuffle(shuffled)
        review=[]; key=[]
        for i,r in enumerate(shuffled,1):
            rid=f'REV-{i:04d}'
            review.append(dict(review_id=rid,text=r['text'],context=r['context'],reviewed_intent='',naturalness='',context_required='',review_status='',notes=''))
            key.append(dict(review_id=rid,**r))
        save('E09_07_07_04_semantic_review',review,list(review[0]))
        save('E09_07_07_04_semantic_review_answer_key',key,list(key[0]))
        print('Review records:',len(review))
        print('No records removed. No model trained. No evaluation dataset read.')
    finally:
        check(all(digest(p)==h for p,h in hashes.items()),'All three input files unchanged')
    print('VALIDATION: PASS (diagnostic flags require review)')


if __name__=='__main__':
    main()
