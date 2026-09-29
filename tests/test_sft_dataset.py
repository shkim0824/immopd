"""CPU test: SFT label masking and packing (set QWEN3_TOKENIZER to a Qwen3 tokenizer directory)."""
import json, os, sys, tempfile
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from immopd.common import chat as C
from immopd.training.sft_dataset import SFTDataset, SFTCollator, build_mixture, IGNORE_INDEX


def main():
    path = os.environ.get("QWEN3_TOKENIZER", "")
    if not path:
        print("skip (QWEN3_TOKENIZER unset)"); return
    from transformers import AutoTokenizer
    tok = C.prepare_tokenizer(AutoTokenizer.from_pretrained(path))
    rows = [{"messages": [{"role": "user", "content": "What is 2+2?"},
                          {"role": "assistant", "content": "<think>\n2+2 is 4.\n</think>\n\nThe answer is \\boxed{4}."}]},
            {"messages": [{"role": "user", "content": "x" * 50},
                          {"role": "assistant", "content": "<think>\n" + "long " * 3000 + "\n</think>\n\nans"}]}]
    ds = SFTDataset(rows, tok, max_len=512, thinking=True)
    assert ds.stats["dropped_long_completion"] == 1 and len(ds) == 1, ds.stats
    ex = ds[0]
    prompt_len = sum(1 for t in ex["labels"] if t == IGNORE_INDEX)
    prompt = tok.decode(ex["input_ids"][:prompt_len])
    assert prompt == C.render_prompt(tok, rows[0]["messages"][:-1]), prompt
    comp = tok.decode([t for t in ex["labels"] if t != IGNORE_INDEX])
    assert comp == rows[0]["messages"][1]["content"] + "<|im_end|>", comp
    assert ex["input_ids"][-1] == tok.convert_tokens_to_ids("<|im_end|>")
    full = tok.apply_chat_template(rows[0]["messages"], tokenize=False)
    assert tok.decode(ex["input_ids"]) + "\n" == full, (tok.decode(ex["input_ids"]), full)
    ds2 = SFTDataset(rows[:1], tok, max_len=512, thinking=False)
    e2 = ds2[0]
    assert tok.decode([t for t in e2["labels"] if t != IGNORE_INDEX]) == "The answer is \\boxed{4}.<|im_end|>"
    assert "<think>\n\n</think>" in tok.decode(e2["input_ids"])
    b = SFTCollator(tok.pad_token_id)([ex, ex])
    assert b["input_ids"].shape == b["labels"].shape == b["attention_mask"].shape
    with tempfile.TemporaryDirectory() as d:
        for name, n in (("a", 100), ("b", 50)):
            with open(os.path.join(d, name + ".jsonl"), "w") as f:
                for i in range(n):
                    f.write(json.dumps({"messages": [{"role": "user", "content": f"{name}{i}"}, {"role": "assistant", "content": "y"}], "domain": name}) + "\n")
        mix = build_mixture({"a": os.path.join(d, "a.jsonl"), "b": os.path.join(d, "b.jsonl")}, {"a": 0.5, "b": 0.5}, 80, 0)
        cnt = {}
        for r in mix:
            cnt[r["domain"]] = cnt.get(r["domain"], 0) + 1
        assert cnt == {"a": 40, "b": 40}, cnt
        mix2 = build_mixture({"a": os.path.join(d, "a.jsonl"), "b": os.path.join(d, "b.jsonl")}, {"a": 0.2, "b": 0.8}, 200, 0)
        cnt2 = {}
        for r in mix2:
            cnt2[r["domain"]] = cnt2.get(r["domain"], 0) + 1
        assert cnt2 == {"a": 40, "b": 50}, cnt2
    print("sft dataset ok")


if __name__ == "__main__":
    main()
