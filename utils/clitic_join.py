"""Conservative heuristics for joining likely Hebrew proclitics."""

HEB = set(chr(c) for c in range(0x05D0, 0x05EB))


def _is_hebrew_word(token: str) -> bool:
    return bool(token) and all(ch in HEB for ch in token)


def join_likely_clitics(text: str, prefixes: str = "ובכלמשה", min_next_len: int = 3) -> tuple[str, int]:
    """Join one-letter prefix tokens with the following multi-letter Hebrew token.

    This is intentionally conservative:
    - only one-letter prefix tokens are considered
    - the following token must be Hebrew and long enough
    - unchanged texts are easy to detect via the returned merge count
    """
    words = text.split()
    out = []
    i = 0
    merges = 0
    prefix_set = set(prefixes)
    while i < len(words):
        cur = words[i]
        if (
            i + 1 < len(words)
            and len(cur) == 1
            and cur in prefix_set
            and _is_hebrew_word(cur)
            and _is_hebrew_word(words[i + 1])
            and len(words[i + 1]) >= min_next_len
        ):
            out.append(cur + words[i + 1])
            merges += 1
            i += 2
            continue
        out.append(cur)
        i += 1
    return " ".join(out), merges
