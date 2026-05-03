#!python3
"""
Interactive human judge for benchmark calibration.

Samples (question, expected, llm_answer) tuples from existing run answers,
presents them one by one, asks the user for `recalled` (y/n) and
`accurate` (y/n) judgements, and saves them to disk. At the end, computes
agreement (Cohen's kappa) with the existing strict and lenient LLM-judge
verdicts on the same items.

The output JSON has the same shape as `judgments/SK_bs5*.json` so it can
be diffed against the LLM judges with `recompute_summaries.py` machinery.

Usage:
    python human_judge.py --run iterative_v6_R4_20260429_2249 --n 50
    python human_judge.py --resume human_judge_sample.json   # continue interrupted session
    python human_judge.py --report human_judge_sample.json   # just compute kappa, no UI
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from glob import glob
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent

CHECKPOINTS = ["checkpoint_500K", "checkpoint_1001K", "checkpoint_2001K",
               "checkpoint_3501K", "final_reeval"]
STRATS = ["S1", "S2", "S3", "S4"]


def load_meta_facts(density=200, seed=42):
    """Return {fact_id: (question, expected_answer, keywords)}."""
    matches = sorted(glob(str(ROOT / f"data/conversations/v6_R4/d{density}_*_seed{seed}_meta.json")))
    if not matches:
        print("ERROR: cannot find conversation meta", file=sys.stderr)
        sys.exit(1)
    with open(matches[0], encoding="utf-8") as f:
        meta = json.load(f)
    facts = {}
    for fm in meta["facts"]:
        facts[fm["fact_id"]] = (fm["question"], fm["answer"], fm["keywords"])
    return facts


def gather_items(run_dir, facts, batch_size=5):
    """For each (cp, sk), gather (cp, sk, fact_id, question, expected,
    llm_answer, llm_strict_recalled, llm_strict_accurate, llm_lenient_recalled,
    llm_lenient_accurate)."""
    items = []
    run_path = ROOT / run_dir
    if not run_path.exists():
        print(f"ERROR: run dir {run_dir} not found", file=sys.stderr)
        sys.exit(1)
    for cp in CHECKPOINTS:
        cp_dir = run_path / cp
        if not cp_dir.exists():
            continue
        for sk in STRATS:
            ans_file = cp_dir / "answers" / f"{sk}_bs{batch_size}.json"
            if not ans_file.exists():
                continue
            with open(ans_file, encoding="utf-8") as f:
                answers = json.load(f)
            # Load both judge files if present
            strict_file = cp_dir / "judgments" / f"{sk}_bs{batch_size}_strict.json"
            lenient_file = cp_dir / "judgments" / f"{sk}_bs{batch_size}.json"
            strict_v = {}
            lenient_v = {}
            if strict_file.exists():
                with open(strict_file, encoding="utf-8") as f:
                    sj = json.load(f)
                strict_v = {v["fact_id"]: v for v in sj.get("verdicts", [])}
            if lenient_file.exists():
                with open(lenient_file, encoding="utf-8") as f:
                    lj = json.load(f)
                lenient_v = {v["fact_id"]: v for v in lj.get("verdicts", [])}
            for fid, llm_ans in answers.items():
                if fid not in facts:
                    continue
                if llm_ans == "[no answer]":
                    continue
                q, expected, kw = facts[fid]
                items.append({
                    "cp": cp,
                    "strategy": sk,
                    "fact_id": fid,
                    "question": q,
                    "expected": expected,
                    "keywords": kw,
                    "llm_answer": llm_ans,
                    "strict_recalled": strict_v.get(fid, {}).get("recalled"),
                    "strict_accurate": strict_v.get(fid, {}).get("accurate"),
                    "lenient_recalled": lenient_v.get(fid, {}).get("recalled"),
                    "lenient_accurate": lenient_v.get(fid, {}).get("accurate"),
                })
    return items


def stratified_sample(items, n, rng):
    """Stratified sample on (cp, strategy) buckets."""
    buckets = {}
    for it in items:
        key = (it["cp"], it["strategy"])
        buckets.setdefault(key, []).append(it)
    keys = sorted(buckets)
    out = []
    # Round-robin until n
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if not buckets[k]:
                continue
            idx = rng.randrange(len(buckets[k]))
            out.append(buckets[k].pop(idx))
            if len(out) >= n:
                break
    rng.shuffle(out)
    return out


def cohens_kappa(a, b):
    """Cohen's kappa between two binary lists (skip None pairs)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    n = len(pairs)
    agree = sum(1 for x, y in pairs if x == y) / n
    pa = sum(1 for x, _ in pairs if x) / n
    pb = sum(1 for _, y in pairs if y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return 1.0, n
    return (agree - pe) / (1 - pe), n


def report(sample_path):
    with open(sample_path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", [])
    judged = [it for it in items if "human" in it]
    if not judged:
        print("No human judgements yet.")
        return
    print(f"\n=== Human judge calibration ({len(judged)} judged of {len(items)} sampled) ===")
    print(f"Run: {data.get('run', '?')}")
    print()
    h_recalled = [bool(it["human"]["recalled"]) for it in judged]
    h_accurate = [bool(it["human"]["accurate"]) for it in judged]
    s_recalled = [it.get("strict_recalled") for it in judged]
    s_accurate = [it.get("strict_accurate") for it in judged]
    l_recalled = [it.get("lenient_recalled") for it in judged]
    l_accurate = [it.get("lenient_accurate") for it in judged]

    # Summary fractions
    print(f"Human:    recalled={sum(h_recalled)}/{len(h_recalled)} "
          f"({100*sum(h_recalled)/len(h_recalled):.1f}%), "
          f"accurate={sum(h_accurate)}/{len(h_accurate)} "
          f"({100*sum(h_accurate)/len(h_accurate):.1f}%)")
    sr = [v for v in s_recalled if v is not None]
    if sr:
        print(f"LLM strict: recalled={sum(1 for v in sr if v)}/{len(sr)} "
              f"({100*sum(1 for v in sr if v)/len(sr):.1f}%)")
    lr = [v for v in l_recalled if v is not None]
    if lr:
        print(f"LLM lenient: recalled={sum(1 for v in lr if v)}/{len(lr)} "
              f"({100*sum(1 for v in lr if v)/len(lr):.1f}%)")

    # Cohen's kappa
    print()
    print("Cohen's kappa (recalled):")
    k, n = cohens_kappa(h_recalled, s_recalled)
    if k is not None:
        print(f"  Human vs Strict   : κ = {k:.3f}  (n={n})")
    k, n = cohens_kappa(h_recalled, l_recalled)
    if k is not None:
        print(f"  Human vs Lenient  : κ = {k:.3f}  (n={n})")

    print()
    print("Cohen's kappa (accurate):")
    k, n = cohens_kappa(h_accurate, s_accurate)
    if k is not None:
        print(f"  Human vs Strict   : κ = {k:.3f}  (n={n})")
    k, n = cohens_kappa(h_accurate, l_accurate)
    if k is not None:
        print(f"  Human vs Lenient  : κ = {k:.3f}  (n={n})")

    # Disagreement examples
    print()
    print("Disagreement examples (where human says recalled=False but a judge says True, or vice versa):")
    for it in judged:
        h = bool(it["human"]["recalled"])
        for jname, jval in (("strict", it.get("strict_recalled")), ("lenient", it.get("lenient_recalled"))):
            if jval is not None and h != bool(jval):
                print(f"  [{it['cp']}/{it['strategy']}/{it['fact_id']}] human={h} {jname}={bool(jval)}")
                print(f"    Q: {it['question'][:100]}")
                print(f"    expected: {it['expected'][:100]}")
                print(f"    answer: {it['llm_answer'][:120]}")
                break  # only one disagreement line per item


def _getch():
    """Read a single keypress without waiting for Enter. Cross-platform."""
    try:
        import msvcrt  # Windows
        ch = msvcrt.getch()
        # On Windows, msvcrt returns bytes
        try:
            return ch.decode("utf-8", errors="replace").lower()
        except Exception:
            return ""
    except ImportError:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.lower()


def yn(prompt, allow_v=False):
    extra = " / v" if allow_v else ""
    while True:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        ans = _getch()
        # Echo the key and a newline
        if ans in ("\x03",):  # Ctrl-C
            raise KeyboardInterrupt
        sys.stdout.write(ans + "\n")
        sys.stdout.flush()
        if ans in ("y", "o"): return True
        if ans in ("n",): return False
        if allow_v and ans == "v": return "v"
        if ans == "q": return None
        if ans in ("?", "/", "s"): return "?"
        print(f"  (y / n / ? to skip / q to save & quit{extra})")


def interactive(sample_path):
    with open(sample_path, encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"]
    todo = [i for i, it in enumerate(items) if "human" not in it]
    print(f"\n{len(items)} items in sample, {len(items) - len(todo)} already judged, {len(todo)} remaining.")
    print("For each item:")
    print("  recalled = does the LLM answer contain the expected fact?")
    print("    y = yes, fact is in the answer")
    print("    n = no, fact is missing  →  accurate auto-set to False, no extra prompt")
    print("    v = answer is 'I don't recall, but topic was mentioned'")
    print("        → recalled=False, accurate=False, notes auto-set")
    print("    ? = skip this item, q = save and quit")
    print("  accurate = is the recalled answer factually correct? (asked only if recalled=y)\n")

    for cnt, idx in enumerate(todo, 1):
        it = items[idx]
        print(f"--- Item {cnt}/{len(todo)} | [{it['cp']} / {it['strategy']} / {it['fact_id']}] ---")
        print(f"Question: {it['question']}")
        print(f"Expected: {it['expected']}")
        if it.get('keywords'):
            print(f"Keywords: {', '.join(it['keywords'])}")
        print(f"LLM answer: {it['llm_answer']}")
        print()

        rec = yn("  recalled? [y/n/v/?/q] ", allow_v=True)
        if rec is None:
            print("  Saving and quitting.")
            break
        if rec == "?":
            print("  Skipped.")
            continue

        if rec == "v":
            recalled = False
            accurate = False
            notes = "I don't recall but topic mentioned"
        elif rec is False:
            recalled = False
            accurate = False  # implies-False rule
            notes = ""
        else:
            recalled = True
            acc = yn("  accurate? [y/n/?/q] ")
            if acc is None:
                print("  Saving (without this item) and quitting.")
                break
            if acc == "?":
                accurate = False
            else:
                accurate = bool(acc)
            notes = ""

        items[idx]["human"] = {
            "recalled": recalled,
            "accurate": accurate,
            "notes": notes,
            "judged_at": datetime.now().isoformat(timespec="seconds"),
        }
        # Save after each judgement (crash-safe)
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print()

    # Final save (already saved on each judge but be safe)
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None,
                        help="Run dir to sample from (e.g. iterative_v6_R4_20260429_2249)")
    parser.add_argument("--n", type=int, default=50, help="Sample size")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument("--out", default="human_judge_sample.json", help="Output JSON path")
    parser.add_argument("--resume", default=None, help="Resume an existing sample file")
    parser.add_argument("--report", default=None, help="Just print report on a sample file")
    parser.add_argument("--positives-only", action="store_true",
                        help="Sample only items where lenient_recalled OR strict_recalled is True")
    args = parser.parse_args()

    if args.report:
        report(args.report)
        return

    if args.resume:
        sample_path = args.resume
        if not os.path.exists(sample_path):
            print(f"ERROR: {sample_path} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        if not args.run:
            print("ERROR: provide --run or --resume", file=sys.stderr)
            sys.exit(1)
        # Build a fresh sample
        facts = load_meta_facts()
        items = gather_items(args.run, facts)
        if args.positives_only:
            before = len(items)
            items = [it for it in items
                     if it.get("lenient_recalled") is True or it.get("strict_recalled") is True]
            print(f"  Filtered to positives-only: {len(items)}/{before} items")
        rng = random.Random(args.seed)
        sampled = stratified_sample(items, args.n, rng)
        data = {
            "run": args.run,
            "sample_seed": args.seed,
            "sample_size": len(sampled),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "items": sampled,
        }
        sample_path = args.out
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Sample created: {sample_path} ({len(sampled)} items)")

    interactive(sample_path)
    print()
    report(sample_path)


if __name__ == "__main__":
    main()
