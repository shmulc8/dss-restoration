"""Probe etcbc/dss Text-Fabric: find the reconstruction marker so we can build a
REAL-lacuna eval (words editors restored in [ ]), not synthetic masking."""
from tf.app import use
A = use("etcbc/dss", silent="deep")
api = A.api
F, Fs, L, T = api.F, api.Fs, api.L, api.T

feats = sorted(api.Fall())
print("otypes:", F.otype.all)
print("\nfeatures:", feats)
recish = [f for f in feats if any(k in f.lower() for k in
          ("rec", "unc", "cor", "rem", "lac", "gap", "full", "glyph", "cons", "int"))]
print("\ncandidate reconstruction/int features:", recish)
for f in recish:
    try:
        fl = Fs(f).freqList()[:6]
        print(f"  {f}: {fl}")
    except Exception as e:
        print(f"  {f}: (n/a {e})")

# show a word that looks reconstructed vs preserved (sign-level scan)
print("\n--- sample signs (glyph | features) ---")
sign_type = "sign" if "sign" in F.otype.all else F.otype.all[0]
cnt = 0
for s in F.otype.s(sign_type):
    fd = {f: Fs(f).v(s) for f in recish if Fs(f).v(s) not in (None, "")}
    if any(k in fd for k in ("rec", "unc", "cor")):
        print(f"  sign {s}: {fd}")
        cnt += 1
        if cnt >= 8:
            break
