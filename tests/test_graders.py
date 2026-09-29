"""CPU tests: answer extraction and grading."""
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from immopd.eval import graders as G
from immopd.eval import if_grader as IG


def wrap(ans: str) -> str:
    return f"<think>reasoning</think>\n\nAnswer: {ans}"


def test_extract():
    assert G.extract_letter(wrap("C"), 4) == "C"
    assert G.extract_letter("<think>x</think>The answer is (B).", 4) == "B"
    assert G.extract_letter("<think>x</think>I think D. Answer: A", 4) == "A"
    assert G.extract_number(wrap("14.1%")) == 14.1
    assert G.extract_number("<think>x</think>\\boxed{-3,200.5}") == -3200.5


def test_mcqa():
    for _ in range(100):
        gold = random.choice("ABCDE")
        wrong = random.choice([c for c in "ABCDE" if c != gold])
        assert G.grade_mcqa(wrap(gold), gold, 5)["correct"] and not G.grade_mcqa(wrap(wrong), gold, 5)["correct"]


def test_numeric():
    assert G.fin_numbers_equal(0.18, 0.18004) and G.fin_numbers_equal(18.004, 0.18004)
    assert not G.fin_numbers_equal(0.19, 0.18004)
    assert G.grade_fin_numeric(wrap("14.1%"), "14.1")["correct"] and G.grade_fin_numeric(wrap("yes"), "yes")["correct"]
    assert not G.grade_fin_numeric(wrap("13.0"), "14.1")["correct"]


def test_if():
    ids = ["punctuation:no_comma", "detectable_format:number_highlighted_sections", "length_constraints:number_words"]
    kwargs = [{}, {"num_highlights": 3}, {"relation": "at least", "num_words": 300}]
    body = "*sec one* " + "word " * 320 + "*sec two* *sec three*"
    e = IG.evaluate_example("ifeval", "p", ids, kwargs, "<think>ignore, commas</think>\n" + body)
    assert e["prompt_strict"]
    e = IG.evaluate_example("ifeval", "p", ids, kwargs, "Sure, here it is:\n" + body)
    assert not e["strict"][0] and e["loose"][0]
    assert len(IG.registry("ifeval")) == 25 and len(IG.registry("ifbench")) == 83 and len(IG.registry("ifevalg")) == 54


def test_validation_pool():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/validation/pool.jsonl"))]
    assert len(rows) == 1024 and all(sum(r["domain"] == d for r in rows) == 256 for d in ("med", "law", "fin", "if"))
    reg = IG.registry("ifevalg")
    assert all(i in reg for r in rows if r["domain"] == "if" for i in r["instruction_id_list"])
    assert len(json.load(open(os.path.join(ROOT, "data/validation/tau2_tasks.json")))["task_ids"]) == 128


if __name__ == "__main__":
    for f in (test_extract, test_mcqa, test_numeric, test_if, test_validation_pool):
        f()
        print(f.__name__, "ok")
