# Frozen paper-results snapshot

Status: **shareable_reproduced_snapshot_not_final_promotion**.

## Unknown-length exact complete-span recovery

| System | Top-1 | Top-5 | Top-10 | 95% scroll-cluster CI (Top-10) |
| :--- | ---: | ---: | ---: | :--- |
| Preserved-only word span | 4.3% | 12.7% | 15.0% | 8.0%--16.1% |
| Preserved-only TavBERT | 2.7% | 4.7% | 6.0% | 2.0%--8.3% |
| Embible-style overlap | 3.0% | 4.7% | 4.7% | 2.0%--5.3% |
| Dev-fitted rank fusion | 5.7% | 12.0% | 15.0% | 8.0%--16.7% |

## Qumran Digital literature agreement

- P0 Top-10: 63.5% (52.4%--74.4%).
- U0 Top-10: 9.5%.
- P0 + retrieval Top-10: 63.5%.
- Split audit: {'train': 0, 'dev': 0, 'heldout': 74} under the model-associated registry; {'train': 40, 'val': 22, 'test': 12} under the later registry.

## Promotion status

- train and evaluate three checkpoints with identical hyperparameters
- freeze one authoritative split registry before new training
- complete a formulaic and near-duplicate cross-split audit
