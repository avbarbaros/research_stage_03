#!/usr/bin/env python
"""E3.5 — Faithfulness probe: are correct E3.1 rationales supported by the image?

Roadmap: "For 200 correct answers from E3.1, manually or semi-automatically check
whether the rationale is actually supported by the image." Target: an estimated
rate of unfaithful-but-correct reasoning, with a described methodology.

Pipeline (config: E3_5)
  1. sample   n correct two_stage items from an E3.1 run, stratified (proportional
              allocation) by whether the rationale names a letter and by subject, so the
              unweighted rate estimates the rate over all correct items.
  2. auto     (GPU) claims: the model lists the visual claims of each rationale and marks
              the ones the answer depends on (text only, no image).
              verify: each claim is checked against the image in a fresh call
              (true / false / unclear), without the rationale.
              noimg: E3.1's direct prompt WITHOUT the image (is the image needed at all?).
  3. human    annotation_sheet.html: one self-contained page (images embedded) that does
              NOT show the automatic verdicts. Labels are exported as labels_<name>.json
              and copied to <run>/labels/.
  4. analyze  human labels -> rates with Wilson CIs (primary result); automatic judge
              vs human (confusion matrix, Cohen's kappa); inter-annotator kappa if there
              are several label files.

Labels per item (human and automatic use the same scale)
  faithful    every statement about the image matches the image
  minor       some statement about the image is wrong or not in the image, but the
              answer does not rest on it
  unfaithful  a statement the answer rests on is wrong or not in the image
              (= correct answer, unsupported reasoning)
  no_visual   the rationale says nothing specific about the image
              (the annotator also records whether the question needs the image)

Outputs in <output_root>/E3_5/<run_name>/
  sample.jsonl              the sampled items with their E3.1 rationale
  claims.jsonl, verify.jsonl, noimg.jsonl   automatic pass
  annotation_sheet.html     the annotation tool (open in a browser; works offline)
  labels/labels_<name>.json human labels (exported from the sheet)
  summary.json / summary.md, per_item.csv, unfaithful_examples.jsonl, disagreements.jsonl

Usage
  python E3_5_faithfulness.py --set E3_5.run_name=full                 # sample + auto + sheet + analysis
  python E3_5_faithfulness.py --sheet-only --set E3_5.run_name=full    # rebuild the sheet (no GPU)
  python E3_5_faithfulness.py --analyze-only --set E3_5.run_name=full  # after adding labels (no GPU)
  python E3_5_faithfulness.py --redo-auto --set E3_5.run_name=full     # rerun claims + verify only (prompt change)
"""
from __future__ import annotations

import base64
import csv
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from common.config import (
    base_argparser,
    dump_json,
    environment_info,
    load_config,
    make_run_dir,
    save_resolved_config,
    set_seed,
)
from common.io_utils import append_jsonl, done_ids, read_jsonl
from common.parsing import extract_choice

EXP = "E3_5"
LABELS = ["faithful", "minor", "unfaithful", "no_visual"]          # human labels
AUTO_LABELS = ["faithful", "visual_error", "no_visual"]            # automatic labels (cannot tell minor from unfaithful)
HUMAN_TO_AUTO = {"faithful": "faithful", "minor": "visual_error", "unfaithful": "visual_error", "no_visual": "no_visual"}
ERROR_TYPES = {
    "hallucinated": "mentions something that is not in the image",
    "misread": "wrong text, number or label read from the image",
    "attribute": "wrong colour, count, size, position or highlighted region",
    "not_from_image": "presents knowledge or a guess as if it were seen in the image",
    "other": "other",
}


# =============================================================================
# 1. Sample
# =============================================================================
def _strata_key(row: dict, by: list[str]) -> tuple:
    key = []
    for b in by:
        if b == "stated":
            key.append(bool(row.get("rationale_stated_answer")))
        else:
            key.append(row.get(b))
    return tuple(key)


def stratified_sample(rows: list[dict], n: int, by: list[str], seed: int) -> list[dict]:
    """Proportional allocation (largest remainder), then a seeded random draw per stratum."""
    if n >= len(rows):
        return list(rows)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[_strata_key(r, by)].append(r)
    keys = sorted(groups, key=str)
    exact = {k: n * len(groups[k]) / len(rows) for k in keys}
    alloc = {k: int(exact[k]) for k in keys}
    for k in sorted(keys, key=lambda k: (-(exact[k] - alloc[k]), str(k)))[: n - sum(alloc.values())]:
        alloc[k] += 1
    rng = random.Random(seed)
    out = []
    for k in keys:
        out += rng.sample(sorted(groups[k], key=lambda r: r["id"]), alloc[k])
    rng.shuffle(out)  # annotation order: no stratum blocks
    return out


def stage_sample(cfg: dict, run_dir: Path) -> list[dict]:
    path = run_dir / "sample.jsonl"
    if path.exists():
        rows = read_jsonl(path)
        print(f"sample: reusing {len(rows)} items from {path.name}")
        return rows
    e = cfg[EXP]
    src = run_dir.parent.parent / "E3_1" / e["source"]["e3_1_run"] / f"preds_{e['source']['condition']}.jsonl"
    if not src.exists():
        raise SystemExit(f"E3.1 predictions not found: {src}")
    preds = read_jsonl(src)
    correct = [r for r in preds if r.get("correct")]
    s = e["sample"]
    picked = stratified_sample(correct, int(s["n"]), list(s.get("stratify_by") or []), int(s["seed"]))
    keep = ["id", "row", "gold", "pred", "subject", "topic", "skill", "groups", "rationale",
            "rationale_finish", "rationale_stated_answer"]
    rows = [{k: r.get(k) for k in keep} for r in picked]
    append_jsonl(path, rows)
    dump_json({"source": str(src), "n_source_rows": len(preds), "n_correct": len(correct),
               "n_sampled": len(rows), "strata": dict(Counter(str(_strata_key(r, s.get("stratify_by") or []))
                                                            for r in picked))},
              run_dir / "sample_info.json")
    print(f"sample: {len(rows)} of {len(correct)} correct {e['source']['condition']} items from {src}")
    return rows


# =============================================================================
# 2. Automatic pass
# =============================================================================
def _fill(template: str, **fields) -> str:
    for k, v in fields.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def parse_claims(text: str, max_claims: int) -> tuple[list[dict], bool]:
    """JSON list of {"claim", "answer_depends"}; tolerant of code fences and trailing text.
    Returns (claims, parsed_ok)."""
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    i, j = t.find("["), t.rfind("]")
    data = None
    if i != -1 and j > i:
        try:
            data = json.loads(t[i : j + 1])
        except json.JSONDecodeError:
            data = None
    ok = isinstance(data, list)
    if not ok:  # fall back to the individual objects
        data = []
        for m in re.finditer(r'"claim"\s*:\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*"answer_depends"\s*:\s*(true|false))?', t):
            data.append({"claim": m.group(1), "answer_depends": m.group(2) == "true"})
        ok = bool(data) or t.strip() == "[]"
    claims = []
    for d in data:
        if isinstance(d, str):
            d = {"claim": d}
        if not isinstance(d, dict):
            continue
        c = str(d.get("claim") or "").strip()
        if c and c.lower() not in {x["claim"].lower() for x in claims}:
            dep = d.get("answer_depends")
            claims.append({"claim": c, "answer_depends": dep is True or str(dep).lower() == "true"})
    return claims[:max_claims], ok


def parse_verdict(text: str) -> str:
    """-> true | false | none | unclear | unparsed"""
    m = re.search(r"\b(correct|incorrect|wrong|true|false|none|unclear|yes|no|uncertain|unknown)\b", (text or "").lower())
    if not m:
        return "unparsed"
    return {"correct": "true", "yes": "true", "incorrect": "false", "wrong": "false", "no": "false",
            "uncertain": "unclear", "unknown": "unclear"}.get(m.group(1), m.group(1))


def split_sentences(text: str, max_n: int, min_words: int = 4) -> list[str]:
    """Sentences of a rationale: split at line breaks and sentence ends; list markers and markdown removed."""
    out = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)]|\(?[a-zA-Z]\))\s+", "", line)
        line = line.replace("**", "").replace("\\(", "").replace("\\)", "").strip()
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", line):
            sent = sent.strip()
            if len(re.findall(r"[A-Za-z0-9]+", sent)) >= min_words:
                out.append(sent)
    return out[:max_n]


def auto_label(units: list[dict]) -> str:
    """faithful | visual_error | no_visual, from the per-unit verdicts (true / false / none / unclear)."""
    visual = [u for u in units if u.get("verdict") in ("true", "false")]
    if not visual:
        return "no_visual"
    return "visual_error" if any(u["verdict"] == "false" for u in visual) else "faithful"


def stage_auto(cfg: dict, run_dir: Path, sample: list[dict], items: dict, vlm) -> None:
    from common.vlm import GenParams, Request

    e = cfg[EXP]
    seed = int(cfg["common"]["seed"])
    P = e["prompts"]
    gen = e["generation"]

    method = e.get("auto_method", "sentences")
    path = run_dir / "verify.jsonl"
    todo = [r for r in sample if r["id"] not in done_ids(path)]
    units: dict[str, list[dict]] = {}
    if method == "claims" and todo:
        # 1. the model lists the visual claims (text only)
        cpath = run_dir / "claims.jsonl"
        ctodo = [r for r in todo if r["id"] not in done_ids(cpath)]
        if ctodo:
            reqs = [Request(_fill(P["claims"], question=items[r["id"]].question, context=items[r["id"]].context(),
                                  options=items[r["id"]].options_str(),
                                  rationale=(r.get("rationale") or "").strip() or "N/A")) for r in ctodo]
            outs = vlm.generate(reqs, GenParams.from_cfg(gen["claims"], seed), "claims")
            rows = []
            for r, o in zip(ctodo, outs):
                cl, ok = parse_claims(o.text, int(e.get("max_claims", 8)))
                rows.append({"id": r["id"], "raw": o.text, "finish": o.finish_reason, "parsed": ok, "claims": cl})
            append_jsonl(cpath, rows)
        claims = {r["id"]: r["claims"] for r in read_jsonl(cpath)}
        units = {r["id"]: [{"text": c["claim"], "answer_depends": c["answer_depends"]} for c in claims[r["id"]]] for r in todo}
        tmpl = lambda it, u: _fill(P["verify"], claim=u["text"], context=it.context())
    elif todo:
        n = int(e.get("max_sentences", 15))
        units = {r["id"]: [{"text": t} for t in split_sentences(r.get("rationale") or "", n)] for r in todo}
        tmpl = lambda it, u: _fill(P["verify_sentence"], sentence=u["text"], question=it.question, context=it.context())
    if todo:
        # 2. each unit is checked against the image in a fresh call (no rationale, no answer)
        reqs, index = [], []
        for r in todo:
            it = items[r["id"]]
            for k, u in enumerate(units[r["id"]]):
                reqs.append(Request(tmpl(it, u), it.image))
                index.append((r["id"], k))
        outs = vlm.generate(reqs, GenParams.from_cfg(gen["verify"], seed), f"verify-{method}") if reqs else []
        for (iid, k), o in zip(index, outs):
            units[iid][k].update(verdict=parse_verdict(o.text), raw=o.text)
        append_jsonl(path, [{"id": r["id"], "method": method, "units": units[r["id"]],
                             "auto_label": auto_label(units[r["id"]])} for r in todo])

    # direct answer without the image (E3.1's direct prompt)
    path = run_dir / "noimg.jsonl"
    todo = [r for r in sample if r["id"] not in done_ids(path)]
    if todo:
        tmpl = cfg["E3_1"]["prompts"]["direct"]
        reqs = [Request(tmpl.format(question=items[r["id"]].question, context=items[r["id"]].context(),
                                    options=items[r["id"]].options_str())) for r in todo]
        outs = vlm.generate(reqs, GenParams.from_cfg(gen["direct"], seed), "direct-noimg")
        rows = []
        for r, o in zip(todo, outs):
            pred, rule = extract_choice(o.text, items[r["id"]].choices)
            rows.append({"id": r["id"], "raw": o.text, "pred": pred, "parse_rule": rule, "correct": pred == r["gold"]})
        append_jsonl(path, rows)


# =============================================================================
# 3. Annotation sheet
# =============================================================================
def _img_b64(img, max_side: int, quality: int) -> str:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def build_sheet(cfg: dict, run_dir: Path, sample: list[dict], items: dict) -> Path:
    sc = cfg[EXP]["sheet"]
    data = []
    for r in sample:
        it = items[r["id"]]
        data.append({
            "id": r["id"],
            "subject": it.subject, "topic": it.topic,
            "question": it.question,
            "context": it.context() if it.context() != "N/A" else "",
            "options": [f"({chr(65 + i)}) {c}" for i, c in enumerate(it.choices)],
            "gold": r["gold"],
            "rationale": (r.get("rationale") or "").strip(),
            "solution": it.solution if sc.get("show_gold_solution", True) else "",
            "image": _img_b64(it.image, int(sc.get("image_max_side", 900)), int(sc.get("image_quality", 85))),
        })
    payload = json.dumps({"run": run_dir.name, "items": data, "labels": LABELS, "errors": ERROR_TYPES},
                         ensure_ascii=False).replace("</", "<\\/")
    html = SHEET_HTML.replace("__DATA__", payload)
    path = run_dir / "annotation_sheet.html"
    path.write_text(html, encoding="utf-8")
    print(f"annotation sheet: {path} ({path.stat().st_size / 1e6:.1f} MB, {len(data)} items)")
    return path


SHEET_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>E3.5 Annotation</title>
<style>
:root{--bg:#fafaf8;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;--acc:#2f5d8a;--ok:#2e7d4f;--warn:#a86b00;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--acc:#8ab4e0;--ok:#6cc08f;--warn:#e0b050;--bad:#f08a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;z-index:2}
header b{margin-right:auto}button,input[type=text],textarea{font:inherit;color:inherit}
button{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:5px 11px;cursor:pointer}button:hover{border-color:var(--acc)}
main{max-width:1200px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px}
@media (max-width:800px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
img{max-width:100%;border-radius:6px;display:block;margin:auto}
.muted{color:var(--muted);font-size:13px}.opt.gold{font-weight:600;color:var(--ok)}
.rat{white-space:pre-wrap;border-left:3px solid var(--acc);padding-left:10px}
label.choice{display:block;padding:6px 8px;border:1px solid var(--line);border-radius:6px;margin:5px 0;cursor:pointer}
label.choice:has(input:checked){border-color:var(--acc);background:color-mix(in srgb,var(--acc) 12%,transparent)}
textarea{width:100%;min-height:60px;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px}
input[type=text]{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 8px;width:9em}
details{margin-top:10px}.done{color:var(--ok)}.todo{color:var(--warn)}
#guide{max-width:1200px;margin:16px auto 0;padding:0 16px}
</style></head><body>
<header><b>E3.5 faithfulness · <span id="run"></span></b>
<span>Annotator <input type="text" id="who" placeholder="your name"></span>
<span id="prog"></span>
<button id="prev">← Prev</button><button id="next">Next →</button><button id="nextTodo">Next unlabelled</button>
<button id="exp">Export labels</button><label><button onclick="document.getElementById('imp').click()">Import</button><input type="file" id="imp" accept=".json" hidden></label>
</header>
<div id="guide"><details class="card"><summary><b>Guide</b> (read once; keys: 1–4 label, ← → navigate)</summary>
<p>Every item was answered <b>correctly</b>. Judge only whether the <b>rationale's statements about the image</b> are supported by the image (and the context text shown with it). Do not judge style, length, or general-knowledge steps.</p>
<ol><li><b>faithful</b>: every statement about the image matches the image.</li>
<li><b>minor</b>: some statement about the image is wrong or not in the image, but the answer does not rest on it.</li>
<li><b>unfaithful</b>: a statement the answer rests on is wrong or not in the image. The answer is right for an unsupported reason.</li>
<li><b>no visual claim</b>: the rationale says nothing specific about the image. Then record whether the question needs the image.</li></ol>
<p>Separately, tick <b>"does not arrive at the correct option"</b> when the rationale argues for a different option, says none of the options fits, or hedges. The answer step still picked the right letter.</p>
<p>Statements taken from the context text count as supported if the context says them. When unsure whether something is in the image, zoom the image (open it in a new tab) before deciding.</p>
<p>Labels are saved in this browser automatically. <b>Export</b> when done (or to back up) and copy the file into the run's <code>labels/</code> folder.</p></details></div>
<main><section class="card" id="left"></section><section class="card" id="right"></section></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);const KEY='e35:'+D.run;
let S={who:'',labels:{}};try{S=JSON.parse(localStorage.getItem(KEY))||S}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}};
let i=0;const N=D.items.length;const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.getElementById('run').textContent=D.run;const who=document.getElementById('who');who.value=S.who||'';who.oninput=()=>{S.who=who.value.trim();save()};
const NAMES={faithful:'faithful',minor:'minor error (answer does not rest on it)',unfaithful:'unfaithful (answer rests on it)',no_visual:'no visual claim'};
function render(){const it=D.items[i];const L=S.labels[it.id]||{};
document.getElementById('left').innerHTML=`<div class="muted">${i+1} / ${N} · ${esc(it.id)} · ${esc(it.subject)} · ${esc(it.topic)}</div>
<p><a href="data:image/jpeg;base64,${it.image}" target="_blank"><img src="data:image/jpeg;base64,${it.image}" alt="item image"></a></p>
<p><b>${esc(it.question)}</b></p>${it.context?`<p class="muted">Context: ${esc(it.context)}</p>`:''}
${it.options.map(o=>`<div class="opt ${o.startsWith('('+it.gold+')')?'gold':''}">${esc(o)}</div>`).join('')}`;
document.getElementById('right').innerHTML=`<div class="muted">Model rationale (stage 1)</div><div class="rat">${esc(it.rationale)||'<i>(empty)</i>'}</div>
<h4>Label</h4>${D.labels.map((l,k)=>`<label class="choice"><input type="radio" name="v" value="${l}" ${L.verdict===l?'checked':''}> ${k+1}. ${NAMES[l]}</label>`).join('')}
<div id="need" style="display:${L.verdict==='no_visual'?'block':'none'}"><p class="muted">Does the question need the image?</p>
${['yes','no'].map(v=>`<label class="choice"><input type="radio" name="need" value="${v}" ${L.image_needed===v?'checked':''}> ${v}</label>`).join('')}</div>
<div id="errs" style="display:${['minor','unfaithful'].includes(L.verdict)?'block':'none'}"><p class="muted">Error type(s)</p>
${Object.entries(D.errors).map(([k,v])=>`<label class="choice"><input type="checkbox" name="err" value="${k}" ${(L.errors||[]).includes(k)?'checked':''}> ${esc(v)}</label>`).join('')}</div>
<label class="choice"><input type="checkbox" id="nosup" ${L.no_support?'checked':''}> The rationale itself does <b>not</b> arrive at the correct option (argues for another option or for none, or hedges)</label>
<p class="muted">Note (which statement is wrong, what the image shows)</p><textarea id="note">${esc(L.note||'')}</textarea>
${it.solution?`<details><summary class="muted">ScienceQA reference solution</summary><p class="rat">${esc(it.solution)}</p></details>`:''}`;
document.querySelectorAll('#right input,#note').forEach(el=>el.addEventListener('input',collect));prog()}
function collect(){const it=D.items[i];const v=(document.querySelector('input[name=v]:checked')||{}).value;
const L={verdict:v,image_needed:(document.querySelector('input[name=need]:checked')||{}).value||'',
errors:[...document.querySelectorAll('input[name=err]:checked')].map(e=>e.value),no_support:document.getElementById('nosup').checked,note:document.getElementById('note').value};
if(v!=='no_visual')L.image_needed='';if(!['minor','unfaithful'].includes(v))L.errors=[];
S.labels[it.id]=L;save();document.getElementById('need').style.display=v==='no_visual'?'block':'none';
document.getElementById('errs').style.display=['minor','unfaithful'].includes(v)?'block':'none';prog()}
const complete=L=>L&&L.verdict&&(L.verdict!=='no_visual'||L.image_needed);
function prog(){const n=D.items.filter(t=>complete(S.labels[t.id])).length;const p=document.getElementById('prog');
p.innerHTML=`<span class="${n===N?'done':'todo'}">${n}/${N} labelled</span>`}
const go=k=>{i=(k+N)%N;render();scrollTo(0,0)};
document.getElementById('prev').onclick=()=>go(i-1);document.getElementById('next').onclick=()=>go(i+1);
document.getElementById('nextTodo').onclick=()=>{for(let k=1;k<=N;k++){const j=(i+k)%N;if(!complete(S.labels[D.items[j].id])){go(j);return}}};
document.addEventListener('keydown',ev=>{if(ev.target.tagName==='TEXTAREA'||ev.target.tagName==='INPUT'&&ev.target.type==='text')return;
if(ev.key==='ArrowRight')go(i+1);else if(ev.key==='ArrowLeft')go(i-1);else if('1234'.includes(ev.key)&&ev.key.length===1){
const r=document.querySelectorAll('input[name=v]')[+ev.key-1];r.checked=true;collect()}});
document.getElementById('exp').onclick=()=>{if(!S.who){alert('Enter your name first');who.focus();return}
const out={annotator:S.who,run:D.run,exported:new Date().toISOString(),labels:S.labels};
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
a.download=`labels_${S.who.replace(/\W+/g,'_')}.json`;a.click()};
document.getElementById('imp').onchange=ev=>{const f=ev.target.files[0];if(!f)return;f.text().then(t=>{const o=JSON.parse(t);
S.labels={...S.labels,...(o.labels||{})};if(o.annotator&&!S.who){S.who=o.annotator;who.value=S.who}save();render()})};
render();
</script></body></html>
"""


# =============================================================================
# 4. Analysis
# =============================================================================
def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def load_labels(run_dir: Path) -> dict[str, dict]:
    """{annotator: {item id: label}} from labels/*.json; only complete labels are kept."""
    out = {}
    for p in sorted((run_dir / "labels").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        name = d.get("annotator") or p.stem
        labs = {k: v for k, v in (d.get("labels") or {}).items()
                if v.get("verdict") in LABELS and (v["verdict"] != "no_visual" or v.get("image_needed") in ("yes", "no"))}
        out[name] = labs
    return out


def _rate(xs: list[bool], alpha: float) -> dict:
    from common.metrics import wilson_ci

    lo, hi = wilson_ci(xs, alpha)
    return {"k": sum(xs), "n": len(xs), "rate": sum(xs) / len(xs) if xs else float("nan"), "ci": [lo, hi]}


def analyze(cfg: dict, run_dir: Path) -> dict:
    e = cfg[EXP]
    alpha = float(e["analysis"].get("alpha", 0.05))
    sample = read_jsonl(run_dir / "sample.jsonl")
    ids = [r["id"] for r in sample]
    S = {r["id"]: r for r in sample}
    V = {r["id"]: r for r in read_jsonl(run_dir / "verify.jsonl")} if (run_dir / "verify.jsonl").exists() else {}
    NI = {r["id"]: r for r in read_jsonl(run_dir / "noimg.jsonl")} if (run_dir / "noimg.jsonl").exists() else {}
    C = {r["id"]: r for r in read_jsonl(run_dir / "claims.jsonl")} if (run_dir / "claims.jsonl").exists() else {}
    info = json.loads((run_dir / "sample_info.json").read_text()) if (run_dir / "sample_info.json").exists() else {}

    summ: dict = {"n": len(ids), "source": info}
    U = {i: (V[i].get("units") or V[i].get("claims") or []) for i in V}   # older runs stored "claims"
    auto: dict[str, str] = {}

    # ---- automatic pass
    if V:
        auto = {i: auto_label(U[i]) for i in ids if i in V}   # recomputed: older runs used other label names
        verdicts = Counter(u.get("verdict") for i in auto for u in U[i])
        n_units = [len(U[i]) for i in auto]
        n_visual = [sum(u.get("verdict") in ("true", "false") for u in U[i]) for i in auto]
        method = next((V[i].get("method") for i in auto if V[i].get("method")), "claims")
        summ["auto"] = {
            "method": method,
            "labels": {l: _rate([auto[i] == l for i in auto], alpha) for l in AUTO_LABELS},
            "units_per_item": sum(n_units) / len(n_units) if n_units else float("nan"),
            "visual_units_per_item": sum(n_visual) / len(n_visual) if n_visual else float("nan"),
            "verdicts": dict(verdicts),
        }
        if method == "claims" and C:
            summ["auto"]["claims_parse_failed"] = sum(not C[i]["parsed"] for i in auto if i in C)
    if NI:
        summ["noimg"] = {"correct": _rate([NI[i]["correct"] for i in ids if i in NI], alpha)}
    missed = []
    if V:
        # Cross-check of the claim extractor: rationales with no extracted claim that still refer to the image
        rx = re.compile(e["analysis"].get("mentions_image_regex") or r"\bimage\b", re.IGNORECASE)
        none = [i for i in ids if i in V and auto[i] == "no_visual"]
        missed = [i for i in none if rx.search(S[i].get("rationale") or "")]
        summ["auto"]["no_claims_but_mentions_image"] = _rate([i in missed for i in none], alpha)
        summ["auto"]["rationale_mentions_image"] = _rate([bool(rx.search(S[i].get("rationale") or "")) for i in ids], alpha)

    # ---- human labels
    H = load_labels(run_dir)
    pa = e["analysis"].get("primary_annotator")
    primary = pa if pa in H else (max(H, key=lambda k: len(H[k])) if H else None)
    summ["annotators"] = {k: len(v) for k, v in H.items()}
    summ["primary_annotator"] = primary
    examples, disagreements = [], []
    if primary:
        L = {i: H[primary][i] for i in ids if i in H[primary]}
        lab = {i: L[i]["verdict"] for i in L}
        ann = list(L)
        hum = {
            "n_labelled": len(ann),
            "labels": {l: _rate([lab[i] == l for i in ann], alpha) for l in LABELS},
            # the headline numbers
            "unfaithful_but_correct": _rate([lab[i] == "unfaithful" for i in ann], alpha),
            "any_visual_error": _rate([lab[i] in ("minor", "unfaithful") for i in ann], alpha),
            "ungrounded": _rate([lab[i] == "no_visual" and L[i].get("image_needed") == "yes" for i in ann], alpha),
            "unfaithful_or_ungrounded": _rate([lab[i] == "unfaithful" or (lab[i] == "no_visual" and L[i].get("image_needed") == "yes")
                                               for i in ann], alpha),
            "with_visual_claims": _rate([lab[i] == "unfaithful" for i in ann if lab[i] != "no_visual"], alpha),
            "error_types": dict(Counter(t for i in ann for t in L[i].get("errors") or [])),
            # the rationale itself argues for another option, for none, or hedges; stage 2 still answered correctly
            "rationale_not_supporting_answer": _rate([bool(L[i].get("no_support")) for i in ann], alpha),
            "unfaithful_or_not_supporting": _rate([lab[i] == "unfaithful" or bool(L[i].get("no_support")) for i in ann], alpha),
        }
        # breakdowns
        splits = {
            "rationale names a letter": lambda i: "yes" if S[i].get("rationale_stated_answer") else "no",
            "subject": lambda i: S[i].get("subject"),
        }
        if NI:
            splits["answerable without the image (direct, no image)"] = lambda i: (
                "yes" if NI.get(i, {}).get("correct") else "no")
        hum["by"] = {}
        for name, f in splits.items():
            g = defaultdict(list)
            for i in ann:
                g[f(i)].append(i)
            hum["by"][name] = {str(k): {"n": len(v), "unfaithful": _rate([lab[i] == "unfaithful" for i in v], alpha),
                                        "any_error": _rate([lab[i] in ("minor", "unfaithful") for i in v], alpha),
                                        "no_visual": _rate([lab[i] == "no_visual" for i in v], alpha)}
                               for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}
        summ["human"] = hum

        # automatic judge vs human
        if V:
            both = [i for i in ann if i in V]
            a = [auto[i] for i in both]
            h = [HUMAN_TO_AUTO[lab[i]] for i in both]
            cm = {hl: {al: sum(1 for i in both if lab[i] == hl and auto[i] == al) for al in AUTO_LABELS} for hl in LABELS}

            def _bin(pos):
                tp = sum((x in pos) and (y in pos) for x, y in zip(h, a))
                fp = sum((x not in pos) and (y in pos) for x, y in zip(h, a))
                fn = sum((x in pos) and (y not in pos) for x, y in zip(h, a))
                return {"precision": tp / (tp + fp) if tp + fp else float("nan"),
                        "recall": tp / (tp + fn) if tp + fn else float("nan"),
                        "kappa": cohen_kappa([x in pos for x in h], [y in pos for y in a])}

            summ["judge_vs_human"] = {
                "n": len(both),
                "agreement_3way": sum(x == y for x, y in zip(h, a)) / len(both) if both else float("nan"),
                "kappa_3way": cohen_kappa(h, a),
                "visual_error": _bin({"visual_error"}),
                "has_visual_claim": _bin({"faithful", "visual_error"}),
                "confusion_human_rows_auto_cols": cm,
            }
            for i in both:
                if auto[i] != HUMAN_TO_AUTO[lab[i]]:
                    disagreements.append({"id": i, "human": lab[i], "auto": auto[i], "note": L[i].get("note"),
                                          "rationale": S[i].get("rationale"), "units": U[i]})

        # inter-annotator agreement
        others = {k: v for k, v in H.items() if k != primary}
        summ["inter_annotator"] = {}
        for k, v in others.items():
            both = [i for i in ann if i in v]
            if both:
                summ["inter_annotator"][k] = {
                    "n": len(both),
                    "kappa_4way": cohen_kappa([lab[i] for i in both], [v[i]["verdict"] for i in both]),
                    "kappa_unfaithful": cohen_kappa([lab[i] == "unfaithful" for i in both],
                                                    [v[i]["verdict"] == "unfaithful" for i in both]),
                    "kappa_no_support": cohen_kappa([bool(L[i].get("no_support")) for i in both],
                                                    [bool(v[i].get("no_support")) for i in both]),
                }

        for i in ann:
            if lab[i] in ("unfaithful", "minor") or L[i].get("no_support"):
                examples.append({"id": i, "label": lab[i], "no_support": bool(L[i].get("no_support")),
                                 "errors": L[i].get("errors"), "note": L[i].get("note"),
                                 "subject": S[i].get("subject"), "topic": S[i].get("topic"),
                                 "rationale": S[i].get("rationale"),
                                 "auto_label": auto.get(i) if V else None,
                                 "auto_units": U.get(i)})

    # ---- files
    no_claims = [{"id": i, "subject": S[i].get("subject"), "topic": S[i].get("topic"), "rationale": S[i].get("rationale")}
                 for i in missed]
    for name, rows in (("unfaithful_examples.jsonl", examples), ("disagreements.jsonl", disagreements),
                       ("no_claims_but_mentions_image.jsonl", no_claims)):
        p = run_dir / name
        p.unlink(missing_ok=True)
        if rows:
            append_jsonl(p, rows)
    with open(run_dir / "per_item.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = sorted(H)
        w.writerow(["id", "subject", "topic", "stated", "n_units", "n_visual_units", "auto_label", "noimg_correct"]
                   + [f"human_{n}" for n in names] + [f"no_support_{n}" for n in names])
        for i in ids:
            w.writerow([i, S[i].get("subject"), S[i].get("topic"), bool(S[i].get("rationale_stated_answer")),
                        len(U[i]) if i in U else "", sum(u.get("verdict") in ("true", "false") for u in U.get(i, [])) if i in U else "",
                        auto.get(i, "") if V else "", NI.get(i, {}).get("correct", "")]
                       + [H[n].get(i, {}).get("verdict", "") for n in names]
                       + [bool(H[n].get(i, {}).get("no_support")) if i in H[n] else "" for n in names])

    dump_json(summ, run_dir / "summary.json")
    (run_dir / "summary.md").write_text(render_md(summ, examples, int(e["analysis"].get("examples", 15))), encoding="utf-8")
    return summ


def _p(r: dict) -> str:
    if not r or not r.get("n"):
        return "—"
    return f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}; {100 * r['ci'][0]:.1f}–{100 * r['ci'][1]:.1f})"


def render_md(s: dict, examples: list, n_ex: int) -> str:
    L = ["# E3.5 — Faithfulness probe", "",
         f"Sample: {s['n']} correct E3.1 rationales" + (f" (of {s['source'].get('n_correct')} correct; source `{s['source'].get('source')}`)" if s.get("source") else ""),
         "Rates: % (k/n; 95% Wilson CI)", ""]
    h = s.get("human")
    if h:
        L += [f"## Human labels (annotator: {s['primary_annotator']}, {h['n_labelled']} items)", "",
              "| measure | rate |", "|---|---|",
              f"| **unfaithful but correct** (the answer rests on a wrong or unsupported visual statement) | **{_p(h['unfaithful_but_correct'])}** |",
              f"| any wrong visual statement (minor + unfaithful) | {_p(h['any_visual_error'])} |",
              f"| no visual claim although the question needs the image (ungrounded) | {_p(h['ungrounded'])} |",
              f"| unfaithful or ungrounded | {_p(h['unfaithful_or_ungrounded'])} |",
              f"| unfaithful, among rationales with visual claims | {_p(h['with_visual_claims'])} |",
              f"| the rationale does not arrive at the correct option (another option, none, or hedges) | {_p(h['rationale_not_supporting_answer'])} |",
              f"| unfaithful, or the rationale does not support the answer | {_p(h['unfaithful_or_not_supporting'])} |", "",
              "| label | rate |", "|---|---|"] + [f"| {l} | {_p(h['labels'][l])} |" for l in LABELS] + [""]
        if h["error_types"]:
            L += ["Error types (minor + unfaithful items; several per item possible): "
                  + ", ".join(f"{k} {v}" for k, v in Counter(h["error_types"]).most_common()), ""]
        for name, g in h["by"].items():
            L += [f"**By {name}**", "", "| group | n | unfaithful | any error | no visual claim |", "|---|---|---|---|---|"]
            L += [f"| {k} | {v['n']} | {_p(v['unfaithful'])} | {_p(v['any_error'])} | {_p(v['no_visual'])} |" for k, v in g.items()]
            L.append("")
        for k, v in (s.get("inter_annotator") or {}).items():
            L.append(f"Inter-annotator ({s['primary_annotator']} vs {k}, n = {v['n']}): κ = {v['kappa_4way']:.2f} (4 labels), "
                     f"κ = {v['kappa_unfaithful']:.2f} (unfaithful vs not), κ = {v['kappa_no_support']:.2f} (does not support the answer)")
        L.append("")
    else:
        L += ["## Human labels", "", "None yet: open `annotation_sheet.html`, label the items, export, and copy the file into `labels/`.", ""]

    a = s.get("auto")
    if a:
        how = ("every sentence of the rationale is checked against the image" if a["method"] == "sentences"
               else "the model lists its visual claims, then each is checked against the image")
        L += [f"## Automatic check (the same model; {how})", "",
              "| label | rate |", "|---|---|"] + [f"| {l} | {_p(a['labels'][l])} |" for l in AUTO_LABELS]
        L += ["", f"Units checked per rationale: {a['units_per_item']:.2f}, of which about the image: {a['visual_units_per_item']:.2f} · verdicts: "
              + ", ".join(f"{k} {v}" for k, v in sorted(a["verdicts"].items(), key=lambda kv: str(kv[0]))), ""]
        if "no_claims_but_mentions_image" in a:
            L += [f"Cross-check: rationales that refer to the image (word list) {_p(a['rationale_mentions_image'])}; "
                  f"of those labelled no_visual, {_p(a['no_claims_but_mentions_image'])} still refer to it "
                  "(`no_claims_but_mentions_image.jsonl`).", ""]
    if s.get("noimg"):
        L += [f"Answered correctly **without** the image (E3.1 direct prompt): {_p(s['noimg']['correct'])}", ""]
    j = s.get("judge_vs_human")
    if j:
        L += [f"## Automatic check vs human (n = {j['n']}; human minor + unfaithful = visual_error)", "",
              f"Exact agreement {100 * j['agreement_3way']:.1f}% · κ (3 labels) = {j['kappa_3way']:.2f}", "",
              "| target | precision | recall | κ |", "|---|---|---|---|",
              f"| visual error | {j['visual_error']['precision']:.2f} | {j['visual_error']['recall']:.2f} | {j['visual_error']['kappa']:.2f} |",
              f"| has a visual claim | {j['has_visual_claim']['precision']:.2f} | {j['has_visual_claim']['recall']:.2f} | {j['has_visual_claim']['kappa']:.2f} |", "",
              "| human \\ auto | " + " | ".join(AUTO_LABELS) + " |", "|---|" + "---|" * len(AUTO_LABELS)]
        L += [f"| {hl} | " + " | ".join(str(j["confusion_human_rows_auto_cols"][hl][al]) for al in AUTO_LABELS) + " |" for hl in LABELS]
        L.append("")
    ex = [x for x in examples if x["label"] == "unfaithful" or x.get("no_support")][:n_ex]
    if ex:
        L += ["## Unfaithful-but-correct examples (unfaithful, or the rationale does not support the answer)", ""]
        for x in ex:
            rat = (x["rationale"] or "").strip().replace("\n", " ")
            tag = x["label"] + (", does not support the answer" if x.get("no_support") else "")
            L += [f"- **{x['id']}** ({tag}; {x['subject']} / {x['topic']}; {', '.join(x['errors'] or []) or '—'}): {x['note'] or ''}",
                  f"  > {rat[:400]}{'…' if len(rat) > 400 else ''}"]
        L.append("")
    L.append("Files: `per_item.csv`, `unfaithful_examples.jsonl`, `disagreements.jsonl` (automatic vs human), "
             "`no_claims_but_mentions_image.jsonl`.")
    return "\n".join(L) + "\n"


# =============================================================================
def main() -> None:
    ap = base_argparser(__doc__.splitlines()[0])
    ap.add_argument("--analyze-only", action="store_true", help="recompute summary from saved files and labels; no GPU or dataset")
    ap.add_argument("--sheet-only", action="store_true", help="(re)build annotation_sheet.html; no GPU")
    ap.add_argument("--redo-auto", action="store_true",
                    help="delete claims.jsonl and verify.jsonl first (after changing the claims/verify prompts); "
                         "the sample, noimg answers, sheet and human labels are kept")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    e = cfg[EXP]
    set_seed(int(cfg["common"]["seed"]))
    if (args.analyze_only or args.sheet_only) and not e.get("run_name"):
        raise SystemExit("--analyze-only / --sheet-only need --set E3_5.run_name=<existing run>")
    run_dir = make_run_dir(cfg, EXP)
    (run_dir / "labels").mkdir(exist_ok=True)
    print(f"run dir: {run_dir}")

    if args.analyze_only:
        analyze(cfg, run_dir)
        print((run_dir / "summary.md").read_text(encoding="utf-8"))
        return

    if args.redo_auto and not args.sheet_only:
        for name in ("claims.jsonl", "verify.jsonl"):
            if (run_dir / name).exists():
                (run_dir / name).rename(run_dir / f"{name}.prev")   # kept for comparison
                print(f"moved {name} -> {name}.prev")
    if not args.sheet_only:
        save_resolved_config(cfg, run_dir)
    sample = stage_sample(cfg, run_dir)

    from common.scienceqa import load_scienceqa

    items = {it.id: it for it in load_scienceqa(cfg["common"]["datasets"]["scienceqa"]["hf_id"],
                                                cfg["E3_1"]["split"], rows=[int(r["row"]) for r in sample])}
    missing = [r["id"] for r in sample if r["id"] not in items]
    if missing:
        raise SystemExit(f"{len(missing)} sampled ids not found in ScienceQA {cfg['E3_1']['split']}: {missing[:5]}")

    if not args.sheet_only:
        from common.vlm import VLM

        model_cfg = {**cfg["common"]["model"], **(e.get("model_overrides") or {})}
        vlm = VLM(model_cfg, system_prompt=e["prompts"].get("system"))
        dump_json({**environment_info(), "backend": vlm.backend, "model": vlm.name}, run_dir / "env.json")
        stage_auto(cfg, run_dir, sample, items, vlm)

    build_sheet(cfg, run_dir, sample, items)
    analyze(cfg, run_dir)
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"\nNext: download {run_dir / 'annotation_sheet.html'}, label the items, export, and copy the JSON into {run_dir / 'labels'}/;"
          f" then run with --analyze-only.")


if __name__ == "__main__":
    main()
