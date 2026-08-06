# Preserved-only RAG by lacuna length

## Held-out results

| Evaluation | Method | Single N | Top-1 | Top-10 | Multiword N | Top-1 | Top-10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Slot | BASELINE | 25 | 12.0% | 60.0% | 440 | 14.5% | 41.4% |
| Slot | RAG | 25 | 20.0% | 64.0% | 440 | 15.0% | 41.8% |
| Sequence | BASELINE | 25 | 12.0% | 60.0% | 100 | 4.0% | 7.0% |
| Sequence | RAG | 25 | 20.0% | 64.0% | 100 | 3.0% | 9.0% |

Single-word and multi-word scores are reported separately. Slot accuracy asks
whether each editorial word appears in Top-K. Sequence accuracy requires the
entire lacuna to match, in order, in one of the Top-K beams.

The Text-Fabric reconstructions are anonymous editorial evaluation labels, not
physical ground truth. They were excluded from model input, training, and
retrieval. RAG uses only preserved non-biblical training scrolls; alpha
`0.5` was selected only on dev scrolls.

No gold character lengths are used. The decoder knows only the number of word
slots in a lacuna.
