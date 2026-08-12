# Frozen paper-results snapshot

Status: **team_review_evidence_snapshot**.

## Unknown-length exact complete-span recovery

| System | Top-1 | Top-5 | Top-10 | 95% scroll-cluster CI (Top-10) |
| :--- | ---: | ---: | ---: | :--- |
| Preserved-only word span | 4.0% | 6.7% | 7.7% | 5.0%--10.8% |
| Preserved-only TavBERT | 1.0% | 2.3% | 2.7% | 0.9%--5.1% |
| Embible-style overlap | 2.3% | 2.3% | 2.3% | 0.6%--4.6% |
| Dev-fitted rank fusion | 4.3% | 7.0% | 7.7% | 5.1%--10.6% |

## Qumran Digital literature agreement

- Soft-trace Top-10: 55.6% (45.3%--65.6%).
- Context-only Top-10: 14.7%.
- Frequency + visible traces Top-10: 36.6%.
- Split audit: {'train': 0, 'dev': 0, 'heldout': 93}.

## Promotion status

- evaluate a representative composition-grouped population split
- audit base-model pretraining exposure if a complete corpus becomes available
