# Final method and system decision

Status: updated 25 July 2026.

The restoration protocol below remains the correct design, but restoration has
not produced a positive external-retrieval result. The strongest completed
paper method is now **quote-aware cross-corpus source recovery**, documented in
[`QUOTE_AWARE_SOURCE_CONNECTION_METHOD.md`](QUOTE_AWARE_SOURCE_CONNECTION_METHOD.md).
It recovers independently known Pesher source books after exact phrase
ablation and nested leave-one-manuscript-out feature selection. Unknown
historical connections remain exploratory.

Final multi-seed restoration training and the scholar study are not complete.

## The paper's method

The paper evaluates DSS restoration as three different empirical claims rather
than collapsing them into one accuracy number.

1. **Known-answer recovery:** hide contiguous, physically preserved DSS text
   after scroll- and composition-level splitting. This measures synthetic
   damage recovery under an unknown span length and unknown word boundaries.
2. **Real-lacuna literature agreement:** at genuinely damaged locations, rank
   all physically compatible, attributed scholarly proposals. These are
   reference readings, not the unknowable original text.
3. **Scholar assistance:** compare manuscript context, candidates, parallels,
   and candidates plus parallels in a blinded, counterbalanced study.

The primary automatic endpoint is exact complete-span Top-10 on synthetic
preserved DSS text. A prediction counts only when the whole missing sequence is
correct and in order. Empty output is a miss. Character error, boundary F1,
word-count error, calibration, abstention, and slot-level WordHit are secondary
diagnostics. Exact length and exact boundaries appear only as labelled oracle
ceilings.

This is methodologically stronger than Embible's evaluation because it adds
DSS-domain preserved-text targets, manuscript- and composition-disjoint splits,
near-duplicate controls, truly unknown multiword length, real-lacuna
literature-agreement evaluation, paired train-only retrieval ablations,
clustered uncertainty, and a scholar-assistance endpoint. It does not imply
that the current model is more accurate than Embible.

## Best implemented baseline

The retained baseline is the **preserved-only word span model with the
unknown-length 1–3-word decoder**:

- train only on physically preserved non-biblical DSS text;
- remove modern reconstructions before example creation;
- search one, two, and three missing-word hypotheses without gold word count,
  character count, or boundaries;
- tune the length penalty on development scrolls;
- rank complete sequences and evaluate every eligible held-out span.

On the expanded 300-span diagnostic it scores 15.0% exact Top-10. The dev-fitted
rank ensemble also scores 15.0%, so the simpler word-only system wins the
primary-metric simplicity tie-break. Fine-tuned TavBERT character completion
scores 6.0%, and the Embible-style overlap ensemble 4.7%; neither is promoted.
RAG likewise remains a paired ablation and an evidence-display feature because
it has not shown a reliable downstream gain.

The 300-span run is an expanded pilot, not the frozen paper test: it reuses the
same deterministic sampling stream as the earlier 30-span diagnostic and has
no clustered confidence interval. It is sufficient for rejecting unhelpful
components, not for a final claim.

## Required final architecture

The current word decoder assigns one tokenizer token to each missing word.
In the expanded diagnostic, 92.3% of target words and 86.3% of complete spans
are representable under that assumption; complete-span coverage falls to 77.0%
for three-word targets. In the original smaller diagnostic the corresponding
figures were 88.3% and 76.7%. Unrepresentable spans remain in the denominator,
as they must, but this makes the decoder unsuitable as the final architecture.

The final model must therefore generate an unknown-length sequence without
assuming one model token per word: a character-, byte-, or unrestricted
subword-level encoder-decoder with explicit stop and whitespace decisions. It
must still use only preserved-text labels. The word model, character model,
Embible overlap, rank fusion, and RAG variants remain matched baselines.

## Final experiment sequence

1. Freeze scroll-disjoint and composition-disjoint manifests, group
   near-duplicates, and publish hashes.
2. Train the tokenization-free span model with seeds 41, 42, and 43, selecting
   checkpoints and decoding parameters only on development scrolls.
3. Compare all systems on the same targets under unknown, approximate physical,
   and oracle information regimes.
4. Report natural-distribution micro results and gap-length macro results with
   paired 95% clustered-bootstrap intervals.
5. Run train-only retrieval as a paired ablation, including answer-string
   removal and help/harm analysis.
6. Run the fixed-decoder Bible transfer test to separate task failure from DSS
   domain failure.
7. Evaluate real lacunae against all compatible attributed proposals, then run
   the blinded scholar-assistance study.

The machine-readable contract is
[`eval/paper_protocol_v1.json`](../eval/paper_protocol_v1.json). A model may be
called stronger than the Embible-style baseline only after its paired 95%
interval is positive on the shared Bible protocol and on the primary DSS
benchmark, including a positive multiword exact-span gain without oracle
information.
