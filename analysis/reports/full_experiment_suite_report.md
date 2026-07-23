# Master Experiment Suite Report
*Executed on 2026-07-23 13:27:41 (Total Runtime: 21.8 minutes)*

## 📊 Summary of Executed Experiments

| Experiment Name | Status | Duration | Key Outcome |
| :--- | :---: | :---: | :--- |
| SQE & DSS Ecosystem Connector Test | SUCCESS | 0.1s | Editor Concordance Schema: {'source_edition': ['DJD (Discoveries in Judaean Dese |
| Cross-Epoch Historical Hebrew Benchmark | SUCCESS | 41.1s | ================================================== |
| Intact Text vs Real Lacunae Benchmark | SUCCESS | 85.7s | Real Lacunae (rec == 1)         36.0%    46.0%    26.7%    28.8%    21.4% |
| Strict Composition-Level Split Benchmark | SUCCESS | 64.3s | MsBERT span-ft      32.0%    30.0%    28.0%    29.6%    30.2% |
| TavBERT Character-Level Benchmark | SUCCESS | 155.7s | Top-20              32.0%    26.0%    24.0%    23.8%    30.8% |
| Ultimate RAG + Enhanced Decoding Pipeline | SUCCESS | 960.1s | Overall Top-10 Accuracy across 600 test spans: 36.8% |

## 🔍 Detailed Logs

### SQE & DSS Ecosystem Connector Test
```
SQE Connector Initialized!
SQE Schema Documentation Loaded (5357 bytes)
Editor Concordance Schema: {'source_edition': ['DJD (Discoveries in Judaean Desert)', 'Qimron QTD (BGU)', 'SQE (Qumranica)'], 'features': ['fragment_pixel_coordinates', 'physical_line_width_mm', 'multi_editor_reconstruction_consensus', 'morphological_alignment']}

```

### Cross-Epoch Historical Hebrew Benchmark
```
Loading historical Hebrew dataset from HF...

==================================================
=== CROSS-EPOCH HISTORICAL HEBREW EVALUATION ===
==================================================
Total Test Sentences: 100
Top-1  Slot Accuracy: 23.0%
Top-5  Slot Accuracy: 35.0%
Top-10 Slot Accuracy: 39.0%
==================================================

```

### Intact Text vs Real Lacunae Benchmark
```
Sampled 250 real lacuna spans (rec == 1)
Sampled 250 intact text spans (rec != 1)

==================================================
=== INTACT TEXT (SYNTHETIC) vs REAL LACUNAE TOP-10 ===
==================================================
Text Type                           1        2        3      4-5       6+
Intact Text (rec != 1)          34.0%    30.0%    27.3%    34.0%    29.7%
Real Lacunae (rec == 1)         36.0%    46.0%    26.7%    28.8%    21.4%


```

### Strict Composition-Level Split Benchmark
```
Total non-biblical compositions: 88
Held-out compositions (26): ['4QD', '4QM', '4QS', 'Apocr_Jer', 'Book_of_Jubilees', 'CD', 'Divrei Moshe', 'Harvesting', 'Hodayot', 'Hodayot-like']...
Gap-length buckets (composition-heldout spans found): {'1': 3047, '2': 1446, '3': 885, '4-5': 1209, '6+': 1222}
Total evaluated composition-heldout spans: 250

==================================================
=== COMPOSITION-LEVEL SPLIT: SLOT TOP-10 ACCURACY ===
==================================================
bucket                  1        2        3      4-5       6+
MsBERT span-ft      32.0%    30.0%    28.0%    29.6%    30.2%


```

### TavBERT Character-Level Benchmark
```
Evaluating TavBERT (tau/tavbert-he) on heldout split (heldout-scrolls)
Sample size: 250 test spans (PER_BUCKET=50)

Processed 25/250 test spans...
Processed 50/250 test spans...
Processed 75/250 test spans...
Processed 100/250 test spans...
Processed 125/250 test spans...
Processed 150/250 test spans...
Processed 175/250 test spans...
Processed 200/250 test spans...
Processed 225/250 test spans...
Processed 250/250 test spans...

==================================================
=== TavBERT (tau/tavbert-he) SLOT-LEVEL ACCURACY ===
==================================================
bucket                  1        2        3      4-5       6+
Top-1               22.0%    17.0%    15.3%    12.6%    17.4%

bucket                  1        2        3      4-5       6+
Top-5               28.0%    20.0%    22.7%    18.4%    23.9%

bucket                  1        2        3      4-5       6+
Top-10              30.0%    23.0%    23.3%    21.5%    26.3%

bucket                  1        2        3      4-5       6+
Top-20              32.0%    26.0%    24.0%    23.8%    30.8%


```

### Ultimate RAG + Enhanced Decoding Pipeline
```
Parallel Witness RAG DB built: 305889 n-gram contexts
Sample size: 300 test spans (PER_BUCKET=100)

==================================================
=== ULTIMATE RAG + ENHANCED DECODING (TOP-10) ===
==================================================
system                  1        2        3
RAG + Enhanced      48.0%    35.0%    34.3%

Overall Top-10 Accuracy across 600 test spans: 36.8%


```
