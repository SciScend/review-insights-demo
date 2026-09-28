# Review insights

Turns a hotel's guest reviews into a one-page report in Bulgarian: what guests
praise, the five things to fix, and reply drafts in each guest's language. Reviews
can be in any mix of Bulgarian, English, Greek and Romanian.

Everything runs on your own machine. A local model in [Ollama](https://ollama.com)
reads the reviews, so no guest text is sent to a cloud service.

## How it works

1. **Language** comes from the script and the diacritics (Greek letters, Cyrillic,
   Romanian ș/ț). No model is needed for that.
2. **Topics:** a local LLM (default `gemma3:12b`) returns, as JSON with a fixed
   schema, the topics each review praises or criticises. There are 12 topics:
   cleanliness, room, food, spa, treatments, staff, communication, noise,
   parking, Wi-Fi, price, location. Each topic comes with a quote.
3. **Verification:** a topic is kept only if its quote really appears in the
   review. The model's output is checked, not trusted. The report says how many
   answers were rejected.
4. **The report** ranks complaints, with more weight on recent months, and shows
   examples with a Bulgarian summary and the original quote. It also has praise,
   all topics, the rating by month and the languages.
5. **Reply drafts** for the latest unhappy reviews: in the guest's language, with
   a Bulgarian translation for the owner. They thank, name the problem and
   promise nothing.
6. **Evaluation:** `--eval` scores the topics against a hand-labelled
   `gold_themes` column (precision, recall, F1 per topic).

Results are cached in `out/cache.json`, so a rerun is instant.

## Run

```bash
ollama pull gemma3:12b
python3 insights.py sample/reviews.csv --hotel "Хотел Пример" --sample --eval
open out/report.html
```

No Python packages are needed beyond the standard library. On a CPU a review
takes about a minute with `gemma3:12b`. `--model` takes any Ollama model that
supports structured output.

**Input CSV columns:** `id, date (YYYY-MM-DD), source, rating (0-10), text`, and
optionally `gold_themes` (topics separated by `;`).

## The sample

**Output:** [docs/sample-report.html](docs/sample-report.html). **Scores:**
[docs/sample-eval.md](docs/sample-eval.md). With `gemma3:12b` on the sample, the topics
score micro F1 0.95: precision 0.93, recall 0.97. 2 of the model's answers were
rejected because their quote was not in the review. These are invented reviews,
written and labelled by the same author, so take the score as a smoke test, not a
benchmark.

`sample/reviews.csv` has 40 invented reviews of an invented hotel in Sandanski,
written in four languages for this demo. They are not from any real property.

---

Built with Claude Opus 5.5 via Claude Code. Published by [SciScend](https://sciscend.com/).
