# DSS Text Restoration

Automatic restoration of missing words (lacunae) in the Dead Sea Scrolls using Hebrew masked language models.

## Project Structure

```
├── data/
│   └── dss_split.py          # Compatibility entrypoint for the shared split code
├── training/
│   ├── finetune.py            # Scattered-mask finetuning
│   └── finetune_span.py       # Span-mask finetuning (our method)
├── eval/
│   ├── tf_lacuna_len.py       # Main evaluation: accuracy by gap length
│   ├── debug_lacuna_ft.py     # Content-word vs all-word breakdown
│   ├── eval_top20.py          # Top-1/10/20 comparison
│   ├── tf_test_lemma.py       # Lemmatization vs string-norm comparison
│   └── validate_leakage.py    # Integrity checks (split, masking, subtoken)
├── analysis/
│   ├── diagnose_errors.py     # Error categorization and ensemble tests
│   └── examples_gen.py        # Case study generation (1QS examples)
├── utils/
│   ├── morph_dss.py           # DictaBERT-lex lemmatization helper
│   ├── dss_split.py           # Shared split utilities
│   └── paths.py               # Repo-root path helpers
├── exploration/
│   ├── debug_lacuna.py        # Early exploration script
│   ├── tf_lacuna.py           # Initial TF-based evaluation
│   ├── tf_lacuna_len_cond.py  # Length-conditioned experiment
│   ├── tf_inspect_recs.py     # Inspect reconstruction markers
│   ├── tf_probe.py            # Initial probing
│   ├── tf_test_multiple.py    # Multi-word test
│   ├── tf_test_sentence.py    # Sentence-level test
│   └── eval_topk.py           # Early top-k eval
└── README.md
```

## Models

- **MsBERT** (`dicta-il/MsBERT`): Whole-word Hebrew BERT (baseline)
- **BEREL** (`dicta-il/BEREL`): Subword Hebrew BERT
- **MsBERT+span-ft** (`ft_msbert_span/`): Our span-mask finetuned model
- **BEREL+span-ft** (`ft_berel_span/`): BEREL with span-mask finetuning

## Key Results (Top-10, lemmatized normalization, non-biblical)

| Gap length | 1 | 2 | 3 | 4-5 | 6+ |
|---|---|---|---|---|---|
| MsBERT base | 37.1% | 29.6% | 30.2% | 26.6% | 25.4% |
| MsBERT+scattered-ft | 40.7% | 28.6% | 24.5% | 17.1% | 13.5% |
| **MsBERT+span-ft** | **41.4%** | **31.4%** | **30.5%** | **25.3%** | **26.7%** |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers text-fabric numpy
```

Run scripts from the repo root, for example `python eval/tf_lacuna_len.py`.

The large local artifacts stay out of git:
- Finetuned model weights live in `ft_*/`.
- The source dataset currently stays at `dss_chunks.csv` in the repo root.
