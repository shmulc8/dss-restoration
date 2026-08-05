# שחזור טקסט ממגילות מדבר יהודה — תוכנית איחוד: נקודות החלטה וראיות

**מטרה.** לשמוליק ולאיתי יש שני codebases עצמאיים עבור אותה מטרת מחקר. מסמך זה מהווה את מראה המקום היחיד למיזוגם ל-repository אחד בעל פרוטוקול benchmark יחיד: הרקע (§0), שתים-עשרה ההחלטות שיש להגיע לגביהן להסכמה (§1–§12), הראיות האמפיריות שנאספו עד כה (§R), והמספרים שכל צד עומד מאחוריהם (נספח). כל הקביעות לגבי ה-codebases אושרו מול הקוד עצמו ב-31 ביולי 2026.

---

## 0. הרקע: המשימה ושני ה-pipelines

**הבעיה.** מגילות מדבר יהודה הן כתבי יד עבריים בני כ-2,000 שנה עם קטעים פיזיים נרחבים שנאבדו — חורים, שחיקות, שוליים קרועים (קטע חסר נקרא *lacuna* / לקונה). חוקרים מציעים שחזורים ידניים; שני הפרויקטים מאמנים מודלי שפה לעשות זאת מתוך ה-context השורד. שניהם עומדים בפני אותה מלכודת: מהדורות מודרניות מדפיסות את השחזורים של העורכים בתוך הטקסט, כך שמודל המאומן באופן נאיבי על מהדורה מודפסת לומד ניחושים מודרניים ולא שפה עתיקה — וה"תחזיות" שלו הופכות לציקליות (מעגליות). לכן, כל pipeline מתחיל בהחלטה אילו מילים מספיק אותנטיות כדי לאמן עליהן.

**המקור המשותף.** שניהם קוראים את אותו הקורפוס הקריא-מכונה: תעתוק ה-ETCBC של המגילות (בפורמט Text-Fabric), המתעד סימן-אחר-סימן אם כל תו שורד פיזית או שסופק על ידי עורך.

**ה-pipeline של איתי (`new_dead_sea_scrolls`):**
1. *Curation* — כל מילה מסווגת לאחת מ-7 קטגוריות שימור; מילים שמורות חלקית מושארות אם ≥ מחצית התווים שלהן אותנטיים; מילים משוחזרות מוסרות.
2. *Dataset* — משפטים בני ≥ 7 מילים (מגילות לא-מקראיות), קובץ xlsx אחד עם תגיות train/val/test; ה-split הוא scroll-disjoint באמצעות hash דטרמיניסטי של שם המגילה.
3. *Models* — מודלי שפה מסוג BERT (כלומר MLMs) שממלאים רווחים במקום: MsBERT (ברמת ה-subwords) ו-TavBERT (ברמת ה-character-level).
4. *Evaluation* — נזק סינתטי: הסתרת 30% מהמילים בכל משפט ב-test set, מילוי כל רווח ב-beam-search, חישוב מדדי hit@k / character-similarity / rank metrics עם bootstrap CIs. כל הרצה כותבת קובץ `predictions.jsonl` עמיד; HTML viewer משווה בין הרצות ומסמן פרוטוקולים שאינם בריי-השוואה ו-train/test contamination.

**ה-pipeline של שמוליק (`dss-restoration`):**
1. *Curation* — מחמירה יותר: מילה נפסלת אם *תו כלשהו* בה סופק על ידי עורך או מסומן כפגום; validator מוציא hard-fail אם תו עריכתי שורד. מגילות לא-מקראיות בלבד.
2. *Dataset* — רצפים רציפים (chunks) של טקסט משומר; split ברמת ה-manuscript (שהוא scroll-disjoint) מבוסס seed.
3. *Models* — בעיקר ByT5, מחולל sequence-to-sequence (seq2seq) על גבי בייטים גולמיים (ללא tokenizer) הכותב את הקטע החסר כטקסט חופשי; וריאנטים של MsBERT/TavBERT/BEREL שעברו fine-tuning כ-baselines.
4. *Evaluation* — שני מסלולים: (א) lacunae סינתטיות רציפות של 1–3 מילים עם 8 מילים context מכל צד; (ב) lacunae אמיתיות: הסכמה עם שחזורים מחקריים שפורסמו (יעדי Qumran-Digital), כולל lemma-level matching באמצעות למטיזר עברי.

**ארבעת ההבדלים המהותיים** (כל השאר הם הנדסה שמתמזגת בצורה נקייה):

| הבדל | איתי | שמוליק | הוכרע ב- |
|---|---|---|---|
| Clean training text | שמירת מילים חצי-אותנטיות | strict whole-word stripping | §1 |
| מבנה ה-test | scattered sentence masks | contiguous lacunae + real gaps | §7, §10 |
| ארכיטקטורה | MLM fill-in | seq2seq generation | §4 |
| מידע אורך הניתן למודל | gold token count (implicit) | gold length ±2 filter (explicit) | §5 |

---

## 12 נקודות ההחלטה

### 1. Data curation — אילו מילים נכנסות לקורפוס
- **איתי:** סכימה של 7 markers; מילים מסוג `PPP` (משומרות חלקית) נשמרות אם יחס התווים האותנטיים ≥ 0.5. Dataset: `ppp_nonbib`, 2,424 משפטים.
- **שמוליק:** stripping של מילה שלמה אם תו כלשהו מכיל `rec == 1` או `rem == 1` או `#` (תנאי OR על שלושה תנאים ברמת התו); `biblical == 0`; validator קשיח.
- **הצעה:** ה-test set יכיל טקסט strict-preserved בלבד, נקודה. ה-training ישתמש בקורפוס ה-`ppp` (≥ 0.5) — יותר data, ואין סכנת leakage לתוך test set מחמיר. control אחד המאומן ב-strict לכל משפחת מודלים; נבחן מחדש רק אם זה ישנה מסקנות.

### 2. יחידת הקורפוס — משפטים מול chunks
- **איתי:** משפטי ETCBC בני ≥ 7 מילים, עם metadata של POS ו-lemma. **שמוליק:** preserved chunks, ללא תלות ב-sentence segmentation.
- **הצעה:** קובץ ה-xlsx של המשפטים הוא יחידת ה-benchmark (תשתית ה-eval רצה עליו); chunks נשארים כאופציה ל-training data. נתעד את התלות ב-segmentation של ETCBC כ-limitation.

### 3. Train/val/test split
- **איתי:** `sha1(scroll) % 100`, cut-points 73/88 — מומש בפועל כ-~74/12/14 ב-`ppp_nonbib` (ה-cut-points מקרבים 70/15/15, הם לא מבטיחים זאת).
- **שמוליק:** RNG מבוסס seed ברמת הספר — מומש בפועל כ-≈ 61/14/25 לפי chunk; קיימים שני מימושי split שדורשים איחוד בכל מקרה.
- **הצעה:** ליצור פעם אחת בעזרת ה-`sha1` bucketing, **להתחייב להקצאת ה-scroll->split שהתקבלה כקובץ JSON קפוא**, ולגרום לכל נתיב קוד לטעון קובץ זה. בדיקה אחת תוודא scroll-disjointness ועקביות בין הקובץ לקורפוס.

### 4. משפחות מודלים — ואיך ByT5 נכנס ל-benchmark המשותף
- **איתי:** MLMs מאחורי מעטפת תאימות-tokenizer (MsBERT WordPiece, TavBERT char-level). **שמוליק:** ByT5 seq2seq פלוס וריאנטים של MLM.
- **הצעה:** headline models יישארו ByT5 + MsBERT + TavBERT (BEREL אופציונלי בנספח). ה-runner של איתי נשאר שכבת ה-scoring וה-artifacts; מתאם (adapter) מסוג `ByT5Predictor` פולט את אותה סכימת `predictions.jsonl` — נבנה והודגם ב-§R1. ה-decoding עשוי להשתנות בין מודלים; שכבת ה-metrics לא.

### 5. מידע אורך — סיכון ההשוואתיות הגדול ביותר
מה שהקוד עושה כיום: **הפרוטוקול של איתי מדליף gold length** — מספר הטוקנים מסוג `[MASK]` שווה ל-token count של מילת הזהב, ועבור TavBERT ברמת התו זהו *מספר התווים המדויק של התשובה* ועבור MsBERT מספר ה-WordPiece שלו: שני oracles שונים. **מסנן האורך של שמוליק נגזר גם הוא מ-gold** (מועמדים מחוץ לאורך הזהב ± 2 תווים מוסרים). אף אחד מהם אינו "unconstrained", ואין פרוטוקול נוכחי המשתמש בפערים שנמדדו פיזית.
- **הצעה:** להגדיר שלושה regimes, לתייג כל מספר מדווח, ולא לערבב לעולם בצורה שקטה:
  - **U0** — ללא מידע אורך (unconstrained);
  - **O-len** — אורך נגזר מ-gold (שני הפרוטוקולים הנוכחיים);
  - **P0** — תקציב אורך מהפער המתועד פיזית (זמין כיום — §6).
  טבלאות כותרת ידווחו U0 + P0; O-len יישאר כ-diagnostic. §R1 מכמת כמה זה משנה את התוצאות.

### 6. עיגון ה-benchmark בסטטיסטיקות לקונה אמיתיות
מתוך 12,971 הרצות מילים פגומות אמיתיות במגילות שנשמרו בצד (held-out data של שמוליק, חושבו ב-31-07-2026):

| אורך פער (מילים) | share | · | תווים חסרים לכל מילה פגומה | share |
|---|---|---|---|---|
| 1 | 77.6% | | 1 | 41.0% |
| 2 | 17.4% | | 2 | 36.6% |
| 3 | 3.7% | | 3 | 10.4% |
| 4 | 0.9% | | 4 | 6.7% |
| 5+ | 0.4% | | 5+ | 5.3% |

ברמת המילה: p90 = 2, p95 = 3, p99 = 4. ברמת התו (16,769 דפוסים מתועדים): median = 2, p95 = 5, p99 = 7. **82.5% מהמילים הפגומות שומרות על לפחות אות קריאה אחת** (דפוסים כמו `סר⬚⬚ך` עם מיקומי פער ידועים).
- **הצעה:** (א) אורכי ה-span ב-benchmark יידגמו מתוך התפלגות זו במקום uniform-K או fixed mask ratio; (ב) **P0 אמיתי** — תקציבי תווים מתוך ספירות ה-`⬚` המתועדות, לעולם לא מתשובת הזהב; (ג) **partial-letters regime** — המודל רואה גם את האותיות השורדות ומקומן. אף benchmark קיים לא מעריך את (ג); זוהי גם ההגדרה הנאמנה ביותר וגם novel contribution. §R2 מראה שאילוצים מסוג זה שווים +54 נקודות.

### 7. מדיניות ה-masking בהערכה (Eval masking policy)
- **איתי:** 30% ממילות המשפט ממוסכות, span concentration 0.5, seed דטרמיניסטי לכל משפט. **שמוליק:** lacunae רציפות של 1–3 מילים, context של 8 מילים.
- **הצעה:** שני מסלולים בשם אחד ב-runner יחיד — `scatter-30` (של איתי; בוחן global context) ו-`lacuna-real` (רצפים רציפים מהתפלגות §6; כולל את K ∈ {1,2,3} של שמוליק). אותו artifact format; שמירה על per-sentence seeding (זה הופך paired statistics לתקפים).

### 8. הוגנות הניקוד — החרגת unaligned words
ה-scorer של איתי משמיט מילים שלא ניתן לעשות להן word-alignment במקום להחשיב אותן כ-misses; MsBERT מאבד כ-38% מהמילים בדרך זו, TavBERT 0%. §R1 מראה שכלל יחיד זה מכריע את הזהות של המנצח ב-leaderboard.
- **הצעה:** unaligned words נחשבות כ-misses במדד הכותרת; המספר המבוסס על החרגה ידווח כמשני כ-`hit@k_aligned`.

### 9. מדדים, סטטיסטיקה ו-artifacts
- **אימוץ מאיתי:** artifact עמיד `predictions.jsonl`; הפרדה בין generation ל-scoring עם rescore; hit@{1,3,5,10}, char_sim, MRR; cluster bootstrap עם B=1000; McNemar pairing; gating לפי `protocol_id`/`decode_id`; train/test contamination check.
- **אימוץ משמוליק:** DictaBERT lemma-level matching; real-lacuna literature-agreement track; leakage/protocol validators; בדיקת forbidden-claims (סריקת docs עבור מספרים ששימשו בעבר כדי שלא יצוצו מחדש).
- **הצעה:** union של שניהם; הקפאת גרסת ה-artifact schema באיחוד.

### 10. הערכת לקונות אמיתיות (Literature agreement)
של שמוליק בלבד כיום; ההערכה של איתי היא סינתטית לחלוטין. לתמונת המצב של Qumran-Digital יש 1,811 שורות גולמיות ← **74 יעדים כשירים / 99 זוגות target-reading תואמים** מנוקדים כעת (מקורות כוללים קמרון 2013/2020, DJD XXIX).
- **הצעה:** לכלול כמסלול ה-benchmark השני — ההערכה היחידה עם טקסט אבוד באמת; טיעון ה-humanities relevance של המאמר נשען עליה. לדווח עם הסייג של n קטן; להגדיל את סט היעדים במשותף; לשלב עם §6(ג), מכיוון שללקונות אמיתיות יש character budgets ואותיות שורדות משלהן.

### 11. פרוטוקול אימון עבור מודלים שעברו fine-tuning
- **בחירת checkpoint:** probe-selected "best", תמיד; ללא epochs שנבחרו ידנית.
- **Seeds:** 3 לכל headline model עם mean ± sd, או seed 1 עם תיחום הטענה בהתאם — יוכרע לכל מודל לפני ההרצה.
- **היקף ה-tuning:** full fine-tune עבור המאמר; LoRA כ-ablation.
- **Recipe:** מתכון תערובת ה-masking של כל מודל הוא חלק מזהותו אך חייב להיות מוקלט ב-`training/tuning_config.json` בתוך כל checkpoint שמפורסם (המוסכמה של איתי — לשמור).

### 12. מכניקת ה-repository
- repo אחד: מחסנית ה-`tuning/` + `comparison/` של איתי כשלד ה-eval/reporting; `data/` של שמוליק (בניית קורפוס + validators), בנצ'מרקים של lemma ו-literature, וה-demo ממוזגים פנימה.
- מקור האמת הוא מודולי `.py`; מחברות (notebooks) מייבאות ממודולים (היפוך החוזה הנוכחי שבו המחברת היא המקור).
- אחסון תוצאות: HF-dataset sync של איתי, רק `manifest.json` + `metrics.json` ב-git.
- בדיקות: union של שתי הסוויטות (masking-regression + scoring; leakage validators + forbidden-claims guard).
- ניקוי בעת המיזוג: מחיקה או מימוש של placeholder code לפני שניתן יהיה לצטט אותו (לשמוליק יש stub benchmark runner שמתחזה לפגיעות); תיקון הפניות במסמכים (ה-docs של איתי מציגים שני קבצים שנעדרים מהארכיון המשותף).

---

## §R. ראיות שנאספו עבור מסמך זה (31-07-2026)

### R1. השוואת פרוטוקול מאוחד ראשונה — paired sample, כל משפחות המודלים

פרוטוקול: `eval_runner` של איתי, test split `ppp_nonbib`, mask_ratio 0.3 / span_concentration 0.5 / seed 42, beam 10×6. TavBERT ו-MsBERT עברו fine-tuning מחדש עם ה-`TuningConfig` המדויק של איתי (full FT, probe-best checkpoint; ראה R4). ByT5 הוא נקודת הביקורת המשומרת הקיימת של שמוליק שנכנסת דרך המתאם החדש (§4), **כאשר אינה מקבלת מידע אורך (U0)** בעוד ה-MLMs מקבלים במשתמע gold token counts (O-len) — נשמר בכוונה כדי לכמת את §5.

**Paired sample: אותם 100 משפטי בדיקה, 729 מילים ממוסכות, כל מודל.** הורץ על מחשב נייד; ה-test split המלא של 338 משפטים שמור למעבר ה-GPU.

Aligned-only scoring (המדד הנוכחי של איתי):

| מודל | regime | hit@1 | hit@10 | hit@10 CI | char_sim | MRR | unaligned |
|---|---|---|---|---|---|---|---|
| TavBERT base | O-len | 7.3% | 21.1% | [17.0, 25.4] | 0.176 | 0.117 | 0 / 729 |
| TavBERT fine-tuned | O-len | 8.0% | 20.6% | [17.0, 24.5] | 0.209 | 0.118 | 0 / 729 |
| MsBERT base | O-len | 6.5% | 16.7% | [11.7, 21.6] | 0.221 | 0.097 | 281 / 729 |
| MsBERT fine-tuned | O-len | 9.8% | 21.6% | [16.8, 26.6] | 0.242 | 0.134 | 279 / 729 |
| ByT5 preserved (adapter) | U0 | 0.5% | 3.3% | [2.1, 4.7] | 0.123 | 0.013 | 0 / 729 |
| ByT5 preserved (adapter) | O-len (±2) | 0.5% | 4.0% | [2.6, 5.5] | 0.127 | 0.015 | 0 / 729 |

Headline scoring לפי §8 (unaligned = miss), hit@10: TavBERT base **21.1%**, TavBERT FT 20.6%, MsBERT base 10.3%, MsBERT FT 13.3%, ByT5 3.3%.

**Contamination note:** 3 מתוך 6 המגילות במדגם (1Q16, 1Q25, 1QM) נמצאות ב-train split של ByT5. תמונת המצב ב-subset הנקייה בת 30 המשפטים (11Q17, 1Q27, 11Q5; 338 מילים) נותרה ללא שינוי: TavBERT base 25.2%, TavBERT FT 23.1%, MsBERT base 19.2% aligned-only (12.1% headline), MsBERT FT 24.3% aligned-only (15.4% headline), ByT5 2.4% (O-len 4.4%).

מה שמדגם זה מבוסס עליו:

1. **כלל הניקוד מכריע את המנצח (§8 אינה קוסמטית).** Aligned-only מכתיר את MsBERT-FT (21.6%); החשבת 38% המילים ה-unaligned שלו כ-misses מכתירה את TavBERT (21.1% מול 13.3%).
2. **יש לאמן את ByT5 מחדש על הנתונים המאוחדים לפני מסקנת seq2seq כלשהי.** מתן מידע האורך של ה-MLMs ל-ByT5 (O-len ±2) מעלה אותו ב-3.3% ← 4.0% בלבד, כך שדיאטת האורך אינה הפער המרכזי. ה-checkpoint מועבר בצורה גרועה מכיוון שהפרוטוקול רחוק מהתפלגות האימון שלו: הוא אומן ל-epoch אחד, על strict-preserved chunks, למילוי רצפים *רציפים* — כאן הוא עומד בפני משפטים מרובי-פערים מפוזרים על טקסט באוצרות שונה. ה-3–4% שלו מודדים distribution shift, לא את התקרה של הארכיטקטורה; fine-tuning של ByT5 על נתונים מאוחדים שייך למעבר ה-GPU לפני שקוראים להשוואת משפחות המודלים.
3. **Fine-tuning עוזר ל-MsBERT בצורה ברורה** (hit@1 6.5% ← 9.8%, MRR 0.097 ← 0.134) **אך בקושי מזיז את TavBERT** במתכון זה (נעצר מוקדם ב-epoch 4; char_sim אכן עלה 0.176 ← 0.209) — יש לבחון מחדש את תזמון מודל התווים לפני הרצת ה-GPU.
4. **דבר אינו מופרד סטטיסטית ב-n=100** — CIs חופפים; הרצת ה-GPU על ה-split המלא עם McNemar pairing מייצרת את הדירוג הניתן לציטוט.

Artifacts (המחשב של שמוליק, ניתן לשיתוף לפי דרישה): `scratch/external_finetune/merged_results/` — לכל מודל: `predictions.jsonl`, `metrics.json`, `word_scores.csv`, פלוס `summary_with_subsets.json`. כולם בפורמט run artifact של איתי, כך שהם נטענים ישירות ל-comparison viewer.

### R2. אותה שאלה על לקונות אמיתיות (ה-benchmark של שמוליק)

תכנון ההערכה הוא כשלעצמו החלטה פתוחה (§5–§8), לכן R1 הוא עדשה אחת בלבד. benchmark זה מעריך במיקומים פגומים באמת מול שחזורים מחקריים שפורסמו, תוך מתן הראיות המתועדות פיזית למודל (אותיות גלויות + אורך ±1 — משטרי real-P0 + partial-letters של §6):

| הגדרה (74 יעדי QD) | Top-1 | Top-10 |
|---|---|---|
| MLM מוגבל (אותיות גלויות + אורך ±1) | 40.5% | **63.5%** (CI 51.4–74.3) |
| אותו דבר, אורך מדויק (±0; 59 היעדים בוודאות אורך) | 52.5% | 69.5% |
| אותם יעדים, ללא אילוצים פיזיים | — | 9.5% |
| + train-only RAG (α מותאם על dev) | 40.5% | 63.5% (ללא שינוי) |
| בקרה: QD "initial reading" (נקודת ההתחלה של ה-human workflow) | 20.3% | 43.2% |

**איך זה עובד** (מפרט מלא: `eval/score_qd_researcher_benchmark.py` + `analysis/reports/QD_RESEARCHER_BENCHMARK.md` במאגר של שמוליק): המודל הוא MsBERT שעבר fine-tuning על קורפוס לא-מקראי strict-preserved (שחזורים עריכתיים הושמטו, כך שמוכח שהוא מעולם לא ראה ניחוש מודרני). בכל לקונה אמיתית היעד ממוסך; ה-MLM מציע מועמדי אוצר מילים; filter שומר רק מועמדים המכילים את האותיות השורדות באופן גלוי לפי הסדר (עם עוגנים משמאל/מימין) ומתאימים לאורך הפער המוערך; הניצולים מדורגים לפי ציון MLM. יעד נחשב כפגיעה (hit) כאשר שחזור המיוחס ביבליוגרפית ותואם פיזית מופיע ב-Top-K. יציב לכל אורך סובלנות ±0/±1/±2; אופליין לחלוטין מול cached QD snapshot.

**63.5% מול 9.5% על יעדים זהים הוא הממצא המרכזי עד כה: מה שנאמר למודל על הפער משפיע בהרבה מאשר איזה מודל ממלא אותו** (כל פער בין מודלים שנמדד ב-R1 הוא ≤ 8 נקודות). שים לב גם ל-69.5% תחת אורך מדויק — השיטה *משתפרת* ככל שהמדידה הפיזית משתפרת, וזהו המקרה המעשי עבור נתוני real-P0 (§6). מגבלות ידועות: לקונות של מילה בודדת (78% מהפערים האמיתיים, לפי §6); מועמדים מוגבלים למילות אוצר מילים של WordPiece בודד (תקרה ש-byte-level generator יכול להסיר); הסכמה עם חוקרים, לא אמת פיזית; n = 74.

### R3. רשם הראיות המקורי של שמוליק (טרום-איחוד)

פירוט מלא: `docs/RESULTS.md` (רשם ראיות, 25-07-2026); לכל שורה יש artifact בדוק; אף אחד מהם אינו תוצאת מאמר קפואה.

| # | Benchmark (יחידה, היקף) | מספרים מרכזיים | סיווג |
|---|---|---|---|
| A | שחזור מילים משומרות — 300 מילים שלמות, לא-מקראי held-out, טקסט עריכתי הושמט | MsBERT-preserved 13.7% Top-1, 30.7% Top-5, **36.3% Top-10**, 43.7% Top-20 | synthetic diagnostic |
| B | לקונות אמיתיות QD — 74 יעדים, אותיות גלויות + אורך ±1 | **40.5% / 63.5% / 67.6%** (Top-1/10/20); ללא אילוצים 9.5% Top-10 | literature-agreement pilot |
| C | Train-only RAG ablation | QD 63.5% ← 63.5%; TF מילה בודדת 60.0% ← 64.0% (25 spans); multiword slots 41.4% ← 41.8% | pilot |
| D | Embible-style synthetic spans — 30 held-out, K∈{1,2,3}, unknown length | מילה בלבד 16.7% Top-10; base TavBERT 6.7%; **כל המערכות 0% על שכבות של 2–3 מילים**; oracle-boundary ceiling 33.3% | synthetic diagnostic |
| E | Bible domain transfer — אותו מפענח על פסוקי Embible | מקרא 50.0% מול מגילות 16.7% Top-10 balanced (מילה 1: 80.0% מול 50.0%) ← הפער הוא התחום, לא המפענח | transfer diagnostic |
| F | Expanded model selection — 300 held-out spans | מילה בלבד 15.0% Top-10 (הטוב ביותר); TavBERT-preserved 6.0%; fusion/RAG לא קודמו | selection pilot |
| G | Quote-aware cross-corpus source recovery — 35 קטעי פשר עם מקור ידוע | 86.6% Top-1 / 99.1% Top-3 book recovery; שורד trigram ablation (52.4% Top-1, p=0.0008) | method validation (פונה למאמר) |

הרשם מפרט גם טענות שהמאגר **אינו** טוען במפורש (אין דיוק קצה-לקצה, אין רווח מ-RAG, אין SOTA, אין הצלחה בריבוי מילים באורך לא ידוע). שורות D–F מסמנות את הבעיה הפתוחה והכנה עבור ה-benchmark המאוחד: **שחזור של 2–3 מילים באורך לא ידוע עומד על כ-0% עבור כל המערכות הנוכחיות.**

### R4. Fine-tunes טריים מאחורי R1 (ההרצות של מסמך זה)

| מודל | Recipe | Stopped | Best checkpoint | probe_exact |
|---|---|---|---|---|
| TavBERT full-FT | TuningConfig של איתי, `ppp_nonbib` | epoch 4 | epoch 3 | 0.1425 ← 0.1471 |
| MsBERT full-FT | אותו דבר | epoch 14 | epoch 12 | 0.1048 ← 0.1572 |

---

## הצעות לצעדים הבאים

1. **פגישה:** מעבר על 12 נקודות ההחלטה (נספח א' הוא רשימת התצפיות); לרוב יש הצעה קונקרטית לקבל, לתקן או לדחות.
2. **הקפאת החלוקה** (§3): יצירת הקצאת ה-scrolls לפי sha1 פעם אחת, להתחייב ל-JSON, ושני בסיסי הקוד יטענו אותו.
3. **GPU pass** (מחליף את R1): test split מלא של 338 משפטים; TavBERT ו-MsBERT base + fine-tuned; ByT5 שאומן מחדש על הנתונים המאוחדים עם schedule מתאים; 3 seeds במידת האפשר; שתי גרסאות הניקוד (§8); McNemar pairing.
4. **בניית מסלול `lacuna-real`** (§6–§7): אורכי span מההתפלגות האמפירית, real-P0 character budgets ממיקומי פער מתועדים, ו-partial-letters regime — המועמד לתרומה חדשנית.
5. **הגדלת סט יעדי הלקונות האמיתיות** (§10) מעבר ל-74 היעדים הנוכחיים, במשותף.
6. **מיזוג repo-אים** לפי §12, כולל הבדיקות ו-cleanup items.

---

## נספח

### א. סיכום החלטות

| # | החלטה | הצעה |
|---|---|---|
| 1 | מילות PPP באימון | כן (יחס ≥ 0.5); test set ב-strict-preserved; control אימון strict אחד |
| 2 | יחידת benchmark | משפטים (קובץ xlsx של איתי); chunks = אופציית אימון |
| 3 | Split | sha1 buckets ← קובץ JSON קפוא של רשימת scrolls, נטען בכל מקום |
| 4 | Headline models | ByT5 + MsBERT + TavBERT; ByT5 דרך adapter (נבנה, §R1) |
| 5 | Length regimes | לתייג את כל המספרים כ-U0 / O-len / P0; headline = U0 + P0 |
| 6 | Masking statistics | דגימה מהתפלגות lacuna אמיתית; real character budgets; partial-letters regime |
| 7 | Masking tracks | `scatter-30` + `lacuna-real` |
| 8 | Unaligned words | miss ב-headline; aligned-only כמשני |
| 9 | Metrics & artifacts | union של שתי המחסניות; הקפאת schema |
| 10 | Real-lacuna track | לכלול (74 יעדים כעת; להגדיל במשותף) |
| 11 | Checkpoints & seeds | probe-best; 3 seeds במידת האפשר; full FT; recipe מתועד |
| 12 | Repo backbone | eval stack של איתי + data/benchmarks של שמוליק; modules מעל notebooks |

### ב. כבר הוסכם (אין צורך בהחלטה)
Scroll-disjoint splitting · קורפוס מרכזי לא-מקראי · zero editor-supplied letters ב-test set · durable per-run artifacts עם config fingerprints.

### ג. מספרים שאנו עומדים מאחוריהם
כלל: אף נתון לא ייכנס למאמר המשותף אלא אם כן הוא מוליך ל-artifact שנבדק ויצרנו או אימתנו בעצמו. כלומר: כל מה שב-§R1 (הרצות paired טריות, artifacts ב-`merged_results/`), §R2/R3 (רשם ראיות עם artifacts), §6 (התפלגויות lacuna מ-12,971 הרצות פער / 16,769 דפוסי תווים), ו-§R4. נתונים שהופצו בעבר ואינם קיימים במסמך זה הוצאו לגמלאות; בדיקת ה-forbidden-claims (§9) תאכוף את הוצאתם לגמלאות בעת האיחוד. ה-GPU pass המלא של 338 המשפטים יחליף את §R1 כשיגיע.
