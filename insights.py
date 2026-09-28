#!/usr/bin/env python3
"""Turn a hotel's guest reviews into a one-page report in Bulgarian.

Reads a CSV of reviews (any mix of Bulgarian, English, Greek and Romanian), asks a
local LLM (Ollama) which topics each review praises or criticises, keeps only the
answers whose quote really appears in the review, and writes:

    out/report.html   what guests praise, the five things to fix, reply drafts
    out/cache.json    the model's answers, so a rerun costs nothing

    python3 insights.py sample/reviews.csv --hotel "Хотел Пример" --sample
    python3 insights.py sample/reviews.csv --eval        # compare with gold_themes

Nothing leaves the machine: the model runs in Ollama on localhost.
CSV columns: id, date (YYYY-MM-DD), source, rating (0-10), text[, gold_themes]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

OLLAMA = "http://localhost:11434/api/chat"

# key: (Bulgarian label for the report, definition for the model)
THEMES = {
    "cleanliness": ("Чистота", "how clean the room, bathroom, towels, carpets and hotel are"),
    "room": ("Стая и легло", "size, bed, furniture, balcony, view, air conditioning, comfort of the room"),
    "food": ("Храна и ресторант", "breakfast, dinner, restaurant, bar, terrace, the food itself"),
    "spa": ("СПА и басейни", "pools, mineral water, sauna, jacuzzi, massage, the spa area"),
    "procedures": ("Процедури и лекар", "medical and balneo treatments, inhalations, physiotherapy, the doctor, the treatment schedule"),
    "staff": ("Персонал", "attitude and helpfulness of people: reception, waiters, therapists, doctor"),
    "communication": ("Комуникация и езици", "language barriers, slow or missing replies to messages, email, Viber or Facebook, bookings or information lost"),
    "noise": ("Шум", "noise from the street, bar, music, construction, neighbours"),
    "parking": ("Паркинг", "parking availability and convenience"),
    "wifi": ("Wi-Fi", "internet connection"),
    "price": ("Цена и стойност", "price, value for money, extra charges, refunds"),
    "location": ("Локация", "location, distance to the park, centre or attractions, quiet area"),
}
LANGS = {"bg": "български", "en": "английски", "el": "гръцки", "ro": "румънски"}
LANG_NAME_EN = {"bg": "Bulgarian", "en": "English", "el": "Greek", "ro": "Romanian"}

SYSTEM = (
    "You analyse hotel guest reviews. For the review given, list every topic it "
    "praises or criticises, using only these topic keys:\n"
    + "\n".join(f"- {k}: {d}" for k, (_, d) in THEMES.items())
    + "\nFor each topic give its sentiment (positive, negative or mixed) and a short "
    "quote copied exactly, word for word, from the review. Do not add topics the "
    "review does not mention. Also write summary_bg: one short sentence in Bulgarian "
    "saying what the guest liked or disliked."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string", "enum": list(THEMES)},
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "mixed"]},
                    "quote": {"type": "string"},
                },
                "required": ["theme", "sentiment", "quote"],
            },
        },
        "summary_bg": {"type": "string"},
    },
    "required": ["themes", "summary_bg"],
}

REPLY_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"}, "reply_bg": {"type": "string"}},
    "required": ["reply", "reply_bg"],
}


def detect_lang(text: str) -> str:
    """Script and diacritics decide the language; no model needed for this."""
    if re.search(r"[Ͱ-Ͽ]", text):
        return "el"
    if re.search(r"[а-яА-Я]", text):
        return "bg"
    if re.search(r"[ăâîșşțţĂÂÎȘŞȚŢ]", text):
        return "ro"
    return "en"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def ask(model: str, messages: list[dict], schema: dict) -> dict:
    body = json.dumps({
        "model": model, "messages": messages, "format": schema, "stream": False,
        "keep_alive": "15m", "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def analyse(reviews: list[dict], model: str, cache_file: Path) -> dict:
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    for i, r in enumerate(reviews, 1):
        key = hashlib.sha1(f"{model}|{r['text']}".encode()).hexdigest()[:16]
        if key in cache:
            r["ai"] = cache[key]
            continue
        t0 = time.time()
        ai = ask(model, [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Review:\n" + r["text"]}], SCHEMA)
        cache[key] = r["ai"] = ai
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {i}/{len(reviews)} {r['id']} {time.time() - t0:.0f}s", file=sys.stderr)
    return cache


def verify(r: dict) -> None:
    """Keep a topic only if its quote is really in the review: the model's
    output is checked, not trusted."""
    text = norm(r["text"])
    kept: dict[str, dict] = {}
    dropped = []
    for t in r["ai"].get("themes", []):
        q = norm(t.get("quote", ""))
        if t.get("theme") in THEMES and q and q in text:
            k = t["theme"]
            if k not in kept:
                kept[k] = dict(t)
            elif kept[k]["sentiment"] != t["sentiment"]:
                kept[k]["sentiment"] = "mixed"  # praised and criticised in one review
        else:
            dropped.append(t)
    r["themes"], r["dropped"] = list(kept.values()), dropped


def draft_replies(reviews: list[dict], hotel: str, model: str, n: int, cache_file: Path) -> list[dict]:
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    unhappy = [r for r in reviews if r["rating"] < 7 or any(t["sentiment"] == "negative" for t in r["themes"])]
    unhappy.sort(key=lambda r: r["date"], reverse=True)
    out = []
    for r in unhappy[:n]:
        key = "reply2|" + hashlib.sha1(f"{model}|{hotel}|{r['text']}".encode()).hexdigest()[:16]
        if key not in cache:
            lang = LANG_NAME_EN[r["lang"]]
            prompt = (
                f"You write the hotel's public reply to a guest review. The reply MUST be written in {lang}, "
                f"the language of the review, at most 70 words. Thank the guest, name the specific problem "
                f"they raised, say the team will look into it. No discounts, no promised dates, no excuses, "
                f"no emojis. Sign it with the hotel's name, {hotel}, written in {lang}."
                + ("" if r["lang"] == "bg" else " Then give reply_bg: the same reply translated into Bulgarian.")
                + f"\n\nReview:\n{r['text']}"
            )
            schema = REPLY_SCHEMA if r["lang"] != "bg" else {
                "type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]}
            cache[key] = ask(model, [{"role": "user", "content": prompt}], schema)
            cache[key].setdefault("reply_bg", cache[key]["reply"])
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  reply {r['id']}", file=sys.stderr)
        out.append({**r, "reply": cache[key]})
    return out


def evaluate(reviews: list[dict]) -> str:
    tp, fp, fn = Counter(), Counter(), Counter()
    for r in reviews:
        gold = set(filter(None, r.get("gold_themes", "").split(";")))
        pred = {t["theme"] for t in r["themes"]}
        for t in pred & gold:
            tp[t] += 1
        for t in pred - gold:
            fp[t] += 1
        for t in gold - pred:
            fn[t] += 1
    lines = ["| Тема | Точност (precision) | Пълнота (recall) | F1 |", "|---|---|---|---|"]
    for k, (label, _) in THEMES.items():
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0
        rc = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0
        f = 2 * p * rc / (p + rc) if p + rc else 0
        lines.append(f"| {label} | {p:.2f} | {rc:.2f} | {f:.2f} |")
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p, rc = T / (T + F), T / (T + N)
    lines.append(f"| **Общо (micro)** | **{p:.2f}** | **{rc:.2f}** | **{2 * p * rc / (p + rc):.2f}** |")
    return "\n".join(lines)


# ---------- report ----------

CSS = """
:root { --ink:#17252c; --ink2:#4b5d66; --ink3:#75878f; --line:#d6e0e3; --bg:#fff; --soft:#f3f7f8;
  --brand:#0050a7; --accent:#13b0a5; --good:#12805c; --bad:#cf3429; --warn:#b86e00; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 "Golos Text", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width:860px; margin:0 auto; padding:32px 20px 48px; display:grid; gap:28px; }
h1 { font-size:26px; margin:0; } h2 { font-size:18px; margin:0 0 10px; } h3 { font-size:15px; margin:0; }
.sub { color:var(--ink2); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.tile { background:var(--soft); border-radius:10px; padding:12px 14px; }
.tile b { display:block; font-size:24px; font-variant-numeric:tabular-nums; }
.tile span { color:var(--ink2); font-size:13px; }
ol.fix { margin:0; padding-left:22px; display:grid; gap:14px; }
ol.fix li::marker { font-weight:700; color:var(--bad); }
.ex { margin:4px 0 0; padding-left:12px; border-left:3px solid var(--line); color:var(--ink2); font-size:14px; }
.ex q { color:var(--ink); } .lang { font-size:11px; font-weight:700; color:var(--ink3); text-transform:uppercase; letter-spacing:.04em; }
.tag { font-size:12px; font-weight:600; padding:1px 7px; border-radius:999px; }
.tag.up { background:#fbe6e4; color:var(--bad); } .tag.flat { background:var(--soft); color:var(--ink2); }
table { width:100%; border-collapse:collapse; font-size:14px; }
td, th { padding:6px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }
th { font-size:12px; color:var(--ink3); font-weight:600; }
.bar { display:flex; height:10px; border-radius:5px; overflow:hidden; background:var(--soft); min-width:120px; }
.bar i { display:block; } .pos { background:var(--accent); } .neg { background:var(--bad); } .mix { background:#e0b04a; }
.num { font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }
.card { border:1px solid var(--line); border-radius:12px; padding:14px; display:grid; gap:8px; }
.draft { background:var(--soft); border-radius:8px; padding:10px 12px; }
.draft b { font-size:12px; color:var(--ink3); text-transform:uppercase; letter-spacing:.04em; display:block; margin-bottom:2px; }
.note { font-size:13px; color:var(--ink3); }
.strong li { margin-bottom:4px; }
svg text { font-size:10px; fill:var(--ink3); }
@media print { .wrap { padding:0; } .card { break-inside:avoid; } }
"""


def e(s) -> str:
    return html.escape(str(s))


def month_chart(reviews: list[dict]) -> str:
    by = defaultdict(list)
    for r in reviews:
        by[r["date"][:7]].append(r["rating"])
    months = sorted(by)
    w, h, pad = 36, 120, 22
    W = len(months) * w + 10
    bars = []
    for i, m in enumerate(months):
        avg = sum(by[m]) / len(by[m])
        bh = avg / 10 * (h - pad * 2)
        x = 5 + i * w
        y = h - pad - bh
        color = "var(--bad)" if avg < 7 else ("var(--warn)" if avg < 8.5 else "var(--accent)")
        bars.append(f'<rect x="{x + 6}" y="{y:.1f}" width="{w - 12}" height="{bh:.1f}" rx="3" fill="{color}"/>'
                    f'<text x="{x + w / 2}" y="{y - 4:.1f}" text-anchor="middle">{avg:.1f}</text>'
                    f'<text x="{x + w / 2}" y="{h - 6}" text-anchor="middle">{m[5:]}.{m[2:4]}</text>')
    return f'<svg viewBox="0 0 {W} {h}" width="100%" style="max-width:{W}px" role="img" aria-label="Средна оценка по месеци">{"".join(bars)}</svg>'


def render(reviews: list[dict], replies: list[dict], hotel: str, model: str, sample: bool, dropped: int) -> str:
    n = len(reviews)
    dates = sorted(r["date"] for r in reviews)
    avg = sum(r["rating"] for r in reviews) / n
    langs = Counter(r["lang"] for r in reviews)
    foreign = n - langs.get("bg", 0)
    complaining = sum(1 for r in reviews if any(t["sentiment"] != "positive" for t in r["themes"]))
    cut = sorted(dates)[len(dates) // 2]  # second half of the period

    pos, neg, mix = Counter(), Counter(), Counter()
    neg_recent, neg_old = Counter(), Counter()
    examples = defaultdict(list)
    for r in reviews:
        for t in r["themes"]:
            k, s = t["theme"], t["sentiment"]
            {"positive": pos, "negative": neg, "mixed": mix}[s][k] += 1
            if s != "positive":
                (neg_recent if r["date"] >= cut else neg_old)[k] += 1
                examples[k].append((r, t))

    fix_order = sorted((k for k in THEMES if neg[k] + mix[k]), key=lambda k: (-(neg[k] + mix[k] * 0.5 + neg_recent[k] * 0.5), k))[:5]
    fixes = []
    for k in fix_order:
        trend = '<span class="tag up">зачестява</span>' if neg_recent[k] > neg_old[k] else '<span class="tag flat">без промяна</span>'
        exs = sorted(examples[k], key=lambda x: x[0]["date"], reverse=True)[:2]
        ex_html = "".join(
            f'<div class="ex"><span class="lang">{e(LANGS[r["lang"]])}, {e(r["source"])}, {e(r["date"])}</span><br>'
            f'{e(r["ai"].get("summary_bg", ""))}<br><q>{e(t["quote"])}</q></div>' for r, t in exs)
        fixes.append(f'<li><h3>{e(THEMES[k][0])}: {neg[k] + mix[k]} оплаквания {trend}</h3>{ex_html}</li>')

    strengths = sorted((k for k in THEMES if pos[k]), key=lambda k: -pos[k])[:3]
    strong_html = "".join(f"<li><b>{e(THEMES[k][0])}</b>: {pos[k]} похвали</li>" for k in strengths)

    rows = []
    for k in sorted(THEMES, key=lambda k: -(pos[k] + neg[k] + mix[k])):
        tot = pos[k] + neg[k] + mix[k]
        if not tot:
            continue
        pw, nw, mw = (100 * x / tot for x in (pos[k], neg[k], mix[k]))
        rows.append(f'<tr><td>{e(THEMES[k][0])}</td><td class="num">{tot}</td><td><div class="bar">'
                    f'<i class="pos" style="width:{pw:.0f}%"></i><i class="mix" style="width:{mw:.0f}%"></i><i class="neg" style="width:{nw:.0f}%"></i>'
                    f'</div></td><td class="num">{pos[k]} / {mix[k]} / {neg[k]}</td></tr>')

    lang_rows = "".join(f'<tr><td>{e(LANGS[l])}</td><td class="num">{c}</td><td class="num">{100 * c / n:.0f}%</td></tr>'
                        for l, c in langs.most_common())

    cards = []
    for r in replies:
        rep = r["reply"]
        orig = "" if r["lang"] == "bg" else f'<div class="draft"><b>Чернова на {e(LANGS[r["lang"]])}</b>{e(rep["reply"])}</div>'
        cards.append(
            f'<div class="card"><span class="lang">{e(r["source"])}, {e(r["date"])}, оценка {r["rating"]:g}</span>'
            f'<div><q>{e(r["text"])}</q></div><div class="note">{e(r["ai"].get("summary_bg", ""))}</div>'
            f'{orig}<div class="draft"><b>На български</b>{e(rep["reply_bg"] if r["lang"] != "bg" else rep["reply"])}</div></div>')

    sample_note = "<p class=\"note\"><b>Примерни данни:</b> обектът и отзивите са измислени, за да се покаже как изглежда отчетът.</p>" if sample else ""
    return f"""<!doctype html><html lang="bg"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Какво казват гостите - {e(hotel)}</title><style>{CSS}</style></head><body><div class="wrap">
<header><h1>Какво казват гостите на {e(hotel)}</h1>
<p class="sub">{n} отзива от {e(dates[0])} до {e(dates[-1])} ({e(", ".join(sorted({r["source"] for r in reviews})))}), на {len(langs)} езика</p>{sample_note}</header>
<section class="tiles">
<div class="tile"><b>{avg:.1f}</b><span>средна оценка от 10</span></div>
<div class="tile"><b>{100 * foreign / n:.0f}%</b><span>от отзивите не са на български</span></div>
<div class="tile"><b>{100 * complaining / n:.0f}%</b><span>съдържат поне едно оплакване</span></div>
<div class="tile"><b>{len(replies)}</b><span>чернови на отговори за одобрение</span></div>
</section>
<section><h2>Петте неща за оправяне</h2><p class="note">Подредени по брой оплаквания, с повече тежест за последните месеци. Под всяко: какво казва гостът на български и цитат от отзива.</p><ol class="fix">{"".join(fixes)}</ol></section>
<section><h2>Какво хвалят</h2><ul class="strong">{strong_html}</ul></section>
<section><h2>Всички теми</h2><table><thead><tr><th>Тема</th><th class="num">Споменавания</th><th>Похвали / смесено / оплаквания</th><th class="num">Брой</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>
<section><h2>Средна оценка по месеци</h2>{month_chart(reviews)}</section>
<section><h2>На какъв език пишат гостите</h2><table><tbody>{lang_rows}</tbody></table>
<p class="note">Всеки отговор е на езика на госта. Ако сайтът и отговорите на запитвания са само на български, тези гости четат преведено от Google или отиват другаде.</p></section>
<section><h2>Чернови на отговори</h2><p class="note">Последните отзиви с оплакване. Прегледайте и редактирайте, преди да публикувате: черновата не обещава нищо конкретно.</p>{"".join(cards)}</section>
<footer class="note">Направено от <a href="https://sciscend.com/">SciScend</a> на {date.today().isoformat()}. Отзивите са обработени локално с модела {e(model)}: нищо не е изпратено към облачна услуга. Всяка тема е проверена: цитатът ѝ трябва да се намира в текста на отзива ({dropped} непотвърдени отговора на модела са отхвърлени).</footer>
</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--hotel", default="Хотел Пример")
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--replies", type=int, default=5, help="how many reply drafts (0 = none)")
    ap.add_argument("--sample", action="store_true", help="mark the report as made from invented data")
    ap.add_argument("--eval", action="store_true", help="score topics against the gold_themes column")
    args = ap.parse_args()

    reviews = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    for r in reviews:
        r["rating"] = float(r["rating"])
        r["lang"] = detect_lang(r["text"])
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "cache.json"
    print(f"{len(reviews)} reviews, model {args.model}", file=sys.stderr)
    analyse(reviews, args.model, cache)
    for r in reviews:
        verify(r)
    dropped = sum(len(r["dropped"]) for r in reviews)
    if args.eval:
        table = evaluate(reviews)
        (args.out / "eval.md").write_text(f"# Оценка на темите ({args.model})\n\n{table}\n\nОтхвърлени непотвърдени отговори: {dropped}\n", encoding="utf-8")
        print(table)
    replies = draft_replies(reviews, args.hotel, args.model, args.replies, cache) if args.replies else []
    (args.out / "report.html").write_text(render(reviews, replies, args.hotel, args.model, args.sample, dropped), encoding="utf-8")
    print(f"{args.out / 'report.html'} written ({dropped} unverified answers dropped)", file=sys.stderr)


if __name__ == "__main__":
    main()
