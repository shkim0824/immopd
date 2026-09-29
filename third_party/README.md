# Third-party code

| directory | source | use |
|---|---|---|
| `ifeval_google/` | google-research/google-research, `instruction_following_eval` | IFEval checkers |
| `ifbench/` | allenai/IFBench | IFBench checkers |
| `ifevalg/` | allenai/open-instruct, `open_instruct/IFEvalG` | checkers for the IF validation prompts |
| `verifiable_instructions/` | abukharin-nv/verifiable-instructions | reward of the IF teacher (GRPO) |

Changes to the upstream files: package imports point to `third_party.*`, and the sentence tokenizer falls
back to `nltk.tokenize.PunktTokenizer` when the pickled Punkt model cannot be loaded.
