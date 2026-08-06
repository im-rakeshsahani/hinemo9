#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_exports.py  —  combine 3 Label Studio (open-source) exports into the
single multi-rater file that agreement_analysis.py consumes.

Each input is one rater's full export (list of tasks). We pull each task's
human annotation (annotations[0].result), normalise the field values, stamp a
clean rater id (A01/A02/A03), and group by the comment id (data.id).

Usage:
    python merge_exports.py exports/r1.json.json exports/r2.json.json exports/r3.json.json -o data/pilot_merged.json
"""
import argparse, json
from collections import defaultdict

# alias -> canonical for emotion_carrying_language (raters store aliases)
ECL = {"English":"en","Hindi":"hi","Both":"both","Neither":"neither",
       "en":"en","hi":"hi","both":"both","neither":"neither"}

def extract(result):
    """Pull the schema fields out of one annotation's result[] list."""
    primary=None; others=[]; ecl=None; conf=None; difficult=False
    for r in result:
        fn=r.get("from_name"); v=r.get("value",{})
        if fn=="primary_emotion":
            primary=(v.get("choices") or [None])[0]
        elif fn=="other_emotions":
            others=list(v.get("choices") or [])
        elif fn=="emotion_carrying_language":
            raw=(v.get("choices") or [None])[0]
            ecl=ECL.get(raw, raw)
        elif fn=="confidence":
            conf=v.get("rating")
        elif fn=="difficult":
            difficult=bool(v.get("choices"))
    emotions=([primary] if primary else [])+[e for e in others if e!=primary]
    return primary, emotions, ecl, conf, difficult

def load_rater(path, rater_id):
    out={}
    for task in json.load(open(path,encoding="utf-8")):
        cid=task.get("data",{}).get("id")
        anns=task.get("annotations") or []
        if not cid or not anns:
            continue
        # take the first non-cancelled annotation
        ann=next((a for a in anns if not a.get("was_cancelled")), anns[0])
        primary,emotions,ecl,conf,difficult=extract(ann.get("result",[]))
        if primary is None:
            continue
        out[cid]={"annotator_id":rater_id,"primary_emotion":primary,
                  "emotions":emotions,"emotion_carrying_language":ecl,
                  "confidence":conf,"difficult":difficult,
                  "_text":task.get("data",{}).get("masked_text",""),
                  "_data":task.get("data",{})}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("files",nargs=3)
    ap.add_argument("-o","--out",default="data/pilot_merged.json")
    a=ap.parse_args()
    raters=[load_rater(f,f"A0{i+1}") for i,f in enumerate(a.files)]
    # union of all comment ids
    all_ids=set().union(*[set(r) for r in raters])
    merged=[]
    coverage=defaultdict(int)
    for cid in sorted(all_ids):
        anns=[r[cid] for r in raters if cid in r]
        coverage[len(anns)]+=1
        base=next(r[cid]["_data"] for r in raters if cid in r)
        merged.append({"id":cid,"masked_text":base.get("masked_text",""),
                       "raw_text":base.get("raw_text",""),
                       "language_label":base.get("language_label"),
                       "genre":base.get("genre"),
                       "annotations":[{k:v for k,v in an.items() if not k.startswith("_")} for an in anns]})
    json.dump(merged,open(a.out,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"merged {len(merged)} comments -> {a.out}")
    print("raters per comment:",dict(coverage))
    print("per-rater counts:",{f"A0{i+1}":len(r) for i,r in enumerate(raters)})

if __name__=="__main__":
    main()
