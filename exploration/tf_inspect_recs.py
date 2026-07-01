"""Inspect etcbc/dss features on reconstructed words to see if we can find alternative readings or notes on debates.
"""
from tf.app import use
A = use("etcbc/dss", silent="deep")
api = A.api
F, L = api.F, api.L

print("--- Inspecting features that might indicate debate or sources ---")

# Let's find some words with 'alt' feature
alt_words = []
for w in F.otype.s("word"):
    alt = F.alt.v(w)
    if alt:
        alt_words.append((w, alt))
print(f"Number of words with 'alt' feature: {len(alt_words)}")
for w, alt in alt_words[:10]:
    signs = L.d(w, "sign")
    glyph = "".join(F.glyph.v(s) or "" for s in signs)
    scroll = L.u(w, "scroll")[0]
    print(f"  Word {w} in {scroll}: '{glyph}' -> alt: '{alt}'")

# Let's inspect 'note_etcbc' feature
note_words = []
for w in F.otype.s("word"):
    note = F.note_etcbc.v(w)
    if note:
        note_words.append((w, note))
print(f"\nNumber of words with 'note_etcbc' feature: {len(note_words)}")
for w, note in note_words[:10]:
    signs = L.d(w, "sign")
    glyph = "".join(F.glyph.v(s) or "" for s in signs)
    scroll = L.u(w, "scroll")[0]
    print(f"  Word {w} in {scroll}: '{glyph}' -> note: '{note}'")

# Let's check 'cor' (correction) and 'rem' (removed) on sign level
print("\n--- Inspecting 'cor' (correction) and 'rem' (removed/erased) ---")
cor_signs = [s for s in F.otype.s("sign") if F.cor.v(s)]
rem_signs = [s for s in F.otype.s("sign") if F.rem.v(s)]
print(f"Signs with corrections: {len(cor_signs)}")
print(f"Signs removed/erased: {len(rem_signs)}")
