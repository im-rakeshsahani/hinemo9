import json, collections
from sklearn.metrics import f1_score

# blind human gold (majority of 3) vs the pre-labeler's prediction, on items
# no annotator had seen labelled
merged = json.load(open('data/blind_merged.json', encoding='utf-8'))
pred = {}
for t in json.load(open('data/blind_subset_withpred.json', encoding='utf-8')):
    cid = t['data']['id']
    for r in t['predictions'][0]['result']:
        if r.get('from_name') == 'primary_emotion':
            pred[cid] = (r['value']['choices'] or [None])[0]

agree = total = 0
per = collections.defaultdict(lambda: [0, 0])
gold_labels, pred_labels = [], []
for it in merged:
    labs = [a['primary_emotion'] for a in it.get('annotations', [])]
    if len(labs) != 3:
        continue
    ctr = collections.Counter(labs)
    top, n = ctr.most_common(1)[0]
    if n < 2:                      # no majority
        continue
    p = pred.get(it['id'])
    if p is None:
        continue
    total += 1
    gold_labels.append(top); pred_labels.append(p)
    per[top][1] += 1
    if p == top:
        agree += 1
        per[top][0] += 1

print(f'blind pre-labeler accuracy: {agree}/{total} = {100*agree/total:.1f}%')
print(f'(gold-subset figure was 78.4% — compare)')
print(f'blind macro-F1 of pre-labeler vs human: '
      f'{f1_score(gold_labels, pred_labels, average="macro", zero_division=0):.3f}')
print('\nper-emotion (blind):')
for e in ["Love","Joy","Anger","Sadness","Nostalgia","Devotion","Neutral"]:
    c, t = per[e]
    if t:
        print(f'  {e:10s}{c}/{t} = {100*c/t:.0f}%')