import sys, json, textwrap, io
sys.path.insert(0, '.')

OUTPUT_FILE = "output.txt"

class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")
    def write(self, data):
        self.terminal.write(data)
        self.file.write(data)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.terminal

_tee = Tee(OUTPUT_FILE)
sys.stdout = _tee

SEP  = "=" * 70
DASH = "-" * 50

# ── 1. Load subset ───────────────────────────────────────────────────────────
print(SEP)
print("STAGE 1: Loading dataset")
print(SEP)

with open("Experiment2/data/problems_all.json") as f:
    full = json.load(f)
print(f"  Total problems in dataset: {len(full)}")


d8 = [p for p in full if p["depth"] == 8][:3]
subset =  d8

for p in subset:
    print(f"  [{p['depth']}] {p['problem_id']}  gt={p['ground_truth']}")
    print(f"       {p['question']}")

with open("/tmp/exp4_smoke.json", "w") as f:
    json.dump(subset, f)

# ── 2. Clause splitting preview ──────────────────────────────────────────────
print(f"\n{SEP}")
print("STAGE 2: Clause splitting preview (before model load)")
print(SEP)

from utils import split_into_clauses, chunk_clauses

for p in subset:
    clauses = split_into_clauses(p["question"])
    chunks  = chunk_clauses(clauses, 2)
    print(f"\n  [{p['depth']}] {p['problem_id']}")
    print(f"  Question : {p['question']}")
    print(f"  Clauses  : {clauses}")
    if len(clauses) < 2 and p["depth"] >= 4:
        print(f"  *** WARNING: only {len(clauses)} clause(s) for a depth-{p['depth']} "
              f"problem — sentence splitting may have failed ***")
    for ci, chunk in enumerate(chunks):
        print(f"    chunk[{ci}]: {chunk}")

# ── 3. Model load ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STAGE 3: Loading model")
print(SEP)

from llm_wrapper import init_model, llm as _llm

init_model("qwen25_math_1.5b", device="cpu")
print("  Model loaded OK")

# ── 4. Logging wrapper around llm ────────────────────────────────────────────
_call_counter = [0]

def logged_llm(prompt, system="", temperature=0.0):
    _call_counter[0] += 1
    n = _call_counter[0]
    print(f"\n  [LLM call #{n}]")
    print(f"  SYSTEM : {system}")
    print(f"  PROMPT :\n{textwrap.indent(prompt, '    | ')}")
    response = _llm(prompt, system=system, temperature=temperature)
    print(f"  RESPONSE:\n{textwrap.indent(response, '    > ')}")
    return response

# ── 5. Run experiment ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STAGE 4: Running exp4 pipeline")
print(SEP)

from exp4_chunked_solve import run_exp4

results = run_exp4(
    dataset_path  = "/tmp/exp4_smoke.json",
    output_path   = "/tmp/exp4_smoke_results.json",
    problem_field = "question",
    answer_field  = "ground_truth",
    depth_field   = "depth",
    steps_field   = None,
    chunk_size    = 2,
    llm_fn        = logged_llm,
)

# ── 6. Per-problem trace ─────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STAGE 5: Per-problem trace")
print(SEP)

for r in results:
    status = "PASS" if r["correct"] else "FAIL"
    print(f"\n{'#'*60}")
    print(f"  [{status}] depth={r['depth']}  gold={r['gold']}  predicted={r['predicted']}")
    print(f"  corrections={r['corrections_applied']}  "
          f"unverified_fallbacks={r['unverified_fallbacks']}  "
          f"extract_failures={r['extraction_failures']}")

    for ci, cl in enumerate(r["chunk_logs"]):
        print(f"\n  -- chunk[{ci}]: {cl['chunk_text']!r}")
        print(f"     context_before : {cl['context_before']}")
        print(f"     extraction     : {cl['extraction_method']}  "
              f"(retries={cl['num_retries']})")
        print(f"     expressions    : {cl['expressions_found']}")

        if cl["verification_log"]:
            for v in cl["verification_log"]:
                flag = ""
                if v["status"] == "corrected":
                    flag = f"  *** CORRECTED: {v['claimed']} -> {v['corrected_value']} ***"
                elif v["status"] == "error":
                    flag = "  *** EVAL ERROR ***"
                print(f"       {v['full_match']:<30} "
                      f"computed={v['computed']}  status={v['status']}{flag}")
        else:
            print(f"       (no arithmetic expressions found)")

        if cl["extraction_method"] == "last_number":
            print(f"       *** FALLBACK: used last number from response ***")
        if cl["extraction_method"] == "failed":
            print(f"       *** EXTRACTION FAILED: no answer recovered for this chunk ***")

        print(f"     context_after  : {cl['context_after']}")

# ── 7. Final summary ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

for r in results:
    status = "✓" if r["correct"] else "✗"
    print(f"  [depth {r['depth']}] {status}  gold={r['gold']}  pred={r['predicted']}  "
          f"corrections={r['corrections_applied']}  chunks={r['num_chunks']}")

total = len(results)
passed = sum(r["correct"] for r in results)
print(f"\n  {passed}/{total} correct")
print(f"  Total LLM calls: {_call_counter[0]}")

_tee.close()
print(f"Output written to {OUTPUT_FILE}")