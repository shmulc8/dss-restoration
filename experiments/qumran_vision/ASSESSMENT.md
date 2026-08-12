# Assessment: Uzan et al. for DSS restoration

## Decision

The work is directly relevant to the missing visual half of our system, but the
released model cannot yet be fused into our paper results. Our current
"visible traces" are supplied by scholarly transcriptions; Uzan et al. model
actual infrared pixels. A future multimodal system should use an image model to
assign compatibility scores to candidate letters, then combine those scores
with contextual MLM ranks. Generated pixels must not be presented as recovered
historical ink.

## What we verified

- The public processed v1 archive is still downloadable and contains 4,137
  training letters, 147 fragment-held-out letters, and one mask per letter.
- The separate real set contains 19 broken-letter images and masks.
- The public code is MIT-licensed, but separate redistribution terms for the
  Google Drive image data are not stated.
- No trained checkpoints are released. The real set has no machine-readable
  letter identities or expert judgments, so its published accuracy cannot be
  recomputed automatically.
- The original stack requires TensorFlow 1, `tf.contrib`, removed SciPy image
  APIs, and MATLAB/MEX code for collecting new letters.
- A bounded PyTorch feasibility run on the released split reached 8.84 dB mean
  full-image PSNR over three sampling runs, versus the paper's 15.54 dB result.
  This is not a numerical reproduction: it uses a compact binary PixelCNN
  rather than PixelCNN++. Its poor real-case samples rule it out as a current
  image-evidence component.

## Research practices we should adopt

1. **Use infrared imagery.** IR suppresses parchment discoloration and makes
   ink more legible. A true physical-evidence claim requires image inputs.
2. **Model manuscript-specific writing.** Their letters come from the same
   scroll because scribal style and imaging conditions matter. We should test
   both within-scroll adaptation and leave-one-scroll-out transfer.
3. **Split by physical provenance.** Overlapping photographs of one fragment
   must never cross train and test. Our future unit should be fragment or plate,
   not cropped letter.
4. **Keep synthetic and real evaluations separate.** Intact letters with
   simulated edge damage provide observable ground truth; real damage supports
   agreement or expert-interpretation claims only.
5. **Return alternatives with uncertainty.** Scholars need ranked candidate
   letters or compatibility scores, not one cosmetically completed image.
6. **Record preprocessing.** Thresholds, wavelength, plate, fragment,
   recto/verso, crop, scale, and mask must remain attached to every example.

## Assumptions we should not inherit

- Damage is not always a single fragment edge or a one-sided half-plane; DSS
  lacunae also include holes, abrasion, internal losses, and multi-letter spans.
- One unusually well-preserved biblical scroll (11Q5) cannot establish
  transfer to non-biblical scrolls, other scripts, other scribes, or Aramaic.
- Resizing and binarizing every letter to 32x32 discards stroke, texture, scale,
  and multispectral information that may matter paleographically.
- PSNR measures pixel resemblance, not correct letter identity or useful word
  restoration.
- Scholarly agreement on 19 selected letters is not historical ground truth;
  one evaluator and manual image cleaning add further uncertainty.

## Minimum path to useful fusion

1. Obtain image coordinates for a small set of our QD targets and retain their
   IR plate metadata.
2. Have a qualified annotator record candidate letter identities and ambiguity,
   without asking for a new end-to-end human study.
3. Train or adapt a modern visual encoder to score observed ink against each
   candidate letter; do not optimize for attractive inpainting.
4. Evaluate image-only, text-only, and fused rankings on the same targets, with
   fragment/scroll-disjoint splits and all failures in the denominator.

Until those alignments and labels exist, the correct use of Uzan et al. is as a
methodological predecessor and a design constraint, not as a claimed component
of the current restoration system.

## Executed registration gate

We joined the frozen 93 QD word IDs through the public SQE 0.33.0 database's
word, sign-stream, position, and sign-ROI tables. Eighty-seven targets have an
SQE/Qumran-Digital word and text-section mapping, but zero has a sign-level
image ROI. SQE contains 162,754 image records, yet its 1,706 sign ROIs cover
only 551 Qumran-Digital words and none of our targets. Consequently, an
image-only/text-only/fused comparison is not currently estimable on the frozen
benchmark. This is a measured data-coverage failure, not a negative model
result.
