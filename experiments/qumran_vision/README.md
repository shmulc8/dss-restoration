# Modern Qumran image pilot

This exploratory track tests whether the released image data from Uzan,
Dershowitz, and Wolf (ICDAR 2017) contain useful recoverable ink signal. It is
not part of the paper's reported restoration results.

The original repository is MIT-licensed and uses TensorFlow 1.0, `tf.contrib`,
removed SciPy image APIs, legacy pickle caches, and up to eight GPUs. Its
`train.sh` actually uses processed dataset v1: 4,137 training letters plus 147
letters held out by fragment. The modern pilot keeps that released split,
creates an internal development subset only from the released training set,
and never deserializes the supplied pickle files.

```bash
.venv/bin/python experiments/qumran_vision/download_data.py
.venv/bin/python experiments/qumran_vision/modern_pilot.py --epochs 30
```

Outputs are written under the ignored `output/qumran_vision/` directory. The
binary PixelCNN is a feasibility baseline, not an exact numerical reproduction
of PixelCNN++. The released 19 real cases have masks and images but no
machine-readable identity labels, so their montage is qualitative and cannot
support an accuracy claim without new annotation.

See [ASSESSMENT.md](ASSESSMENT.md) for the integration decision and the DSS
research practices carried into the main paper.

Source: <https://github.com/ghostcow/pixel-cnn-qumran>

Data redistribution terms are not stated separately from the code license.
The downloader therefore verifies and uses the upstream archives locally; the
images are not committed or repackaged here.
