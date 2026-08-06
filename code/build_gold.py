import json, csv, collections

# 1) rater labels
merged = json.load(open('data/gold_merged.json', encoding='utf-8'))
# 2) adjudicated overrides
adj = {}
for row in csv.DictReader(open('data/adjudicated.csv', encoding='utf-8')):
    adj[row['id']] = row['adjudicated_label']
# 3) LLM predictions (from original import)
pred = {}
for t in json.load(open('data/final_annotation_set.json', encoding='utf-8')):
    cid = t['data'].get('id'); pr = t.get('predictions') or []
    if cid and pr:
        for r in pr[0].get('result', []):
            if r.get('from_name')=='primary_emotion':
                pred[cid] = (r.get('value',{}).get('choices') or [None])[0]

gold = []
source_counts = collections.Counter()
for item in merged:
    cid = item['id']
    labels = [a['primary_emotion'] for a in item.get('annotations', [])]
    ctr = collections.Counter(labels)
    top, n = ctr.most_common(1)[0]
    if cid in adj:
        gl, src = adj[cid], 'adjudicated'
    elif n >= 2:
        gl, src = top, ('unanimous' if n==3 else 'majority')
    else:
        gl, src = None, 'no_majority_unresolved'   # shouldn't happen; all 23 adjudicated
    source_counts[src] += 1
    if gl:
        gold.append({
            'id': cid,
            'masked_text': item.get('masked_text',''),
            'raw_text': item.get('raw_text',''),
            'language_label': item.get('language_label'),
            'genre': item.get('genre'),
            'gold_emotion': gl,
            'gold_source': src,
            'rater_labels': labels,
            'llm_prediction': pred.get(cid),
        })

json.dump(gold, open('data/gold_dataset.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)

print(f'GOLD SET: {len(gold)} items')
print('label source:', dict(source_counts))
print('\ngold emotion distribution:')
gd = collections.Counter(g['gold_emotion'] for g in gold)
for e,c in gd.most_common():
    print(f'  {e:12s}{c}')

# LLM validation: human gold vs LLM prediction
have_pred = [g for g in gold if g['llm_prediction']]
agree = sum(1 for g in have_pred if g['gold_emotion']==g['llm_prediction'])
print(f'\n=== LLM validation (human gold vs LLM prediction, n={len(have_pred)}) ===')
print(f'  agreement (accuracy): {agree}/{len(have_pred)} = {100*agree/len(have_pred):.1f}%')

# per-emotion LLM recall (of human-gold items of each emotion, how many LLM got right)
print('\n  per-emotion LLM accuracy (human gold as truth):')
by_emo = collections.defaultdict(lambda: [0,0])
for g in have_pred:
    by_emo[g['gold_emotion']][1]+=1
    if g['gold_emotion']==g['llm_prediction']:
        by_emo[g['gold_emotion']][0]+=1
for e in ["Love","Joy","Anger","Sadness","Fear","Surprise","Nostalgia","Devotion","Neutral"]:
    c,t = by_emo[e]
    if t: print(f'    {e:12s}{c}/{t} = {100*c/t:.0f}%')