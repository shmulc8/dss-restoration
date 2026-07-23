# Context Degradation Ablation (Pythia-style stress test)

Randomly masks a fraction of the surrounding context words on 120 held-out non-biblical cases (≤30 per gap-length bucket), then measures restoration accuracy.

| Context Noise Level | Slot-Level Top-1 | Sequence-Level Top-1 |
| ------------------- | ---------------- | -------------------- |
| 0% noise | 12.2% | 7.5% |
| 10% noise | 11.9% | 8.3% |
| 25% noise | 8.7% | 5.8% |
| 40% noise | 7.4% | 2.5% |
