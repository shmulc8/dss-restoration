const TRIAGE_KEY = "dss-restoration-demo-triage-v1";

const state = {
  failures: [],
  spans: [],
  unknowns: [],
  benchmark: null,
  guide: null,
  mode: "spans",
  triage: {},
  page: {
    spans: 1,
    unknown: 1,
    triage: 1,
  },
};

const PAGE_SIZE = 80;
const LOW_INFO_CANDIDATES = new Set([
  "אל", "כול", "כל", "את", "אשר", "על", "כי", "די", "ה", "ו", "לא", "לוא",
  "מן", "עם", "איש", "יום", "ים", "ישראל", "לב",
]);

const els = {
  heroStats: document.getElementById("hero-stats"),
  triageCards: document.getElementById("triage-cards"),
  spansCards: document.getElementById("spans-cards"),
  unknownCards: document.getElementById("unknown-cards"),
  benchmarkTop10: document.getElementById("benchmark-top10"),
  benchmarkRetrieval: document.getElementById("benchmark-retrieval"),
  spansCount: document.getElementById("spans-count"),
  spansPage: document.getElementById("spans-page"),
  spansPrev: document.getElementById("spans-prev"),
  spansNext: document.getElementById("spans-next"),
  unknownCount: document.getElementById("unknown-count"),
  unknownPage: document.getElementById("unknown-page"),
  unknownPrev: document.getElementById("unknown-prev"),
  unknownNext: document.getElementById("unknown-next"),
  triageCount: document.getElementById("triage-count"),
  triagePage: document.getElementById("triage-page"),
  triagePrev: document.getElementById("triage-prev"),
  triageNext: document.getElementById("triage-next"),
  searchInput: document.getElementById("search-input"),
  scrollFilter: document.getElementById("scroll-filter"),
  statusFilter: document.getElementById("status-filter"),
  issueFilter: document.getElementById("issue-filter"),
  rarityFilter: document.getElementById("rarity-filter"),
  shortFilter: document.getElementById("short-filter"),
  goldSimilarFilter: document.getElementById("gold-similar-filter"),
  top1SimilarFilter: document.getElementById("top1-similar-filter"),
  lengthConstraintFilter: document.getElementById("length-constraint-filter"),
  hideParticlesFilter: document.getElementById("hide-particles-filter"),
  sortSelect: document.getElementById("sort-select"),
  modeSwitch: document.getElementById("mode-switch"),
};

const triageTemplate = document.getElementById("triage-card-template");
const spanTemplate = document.getElementById("span-card-template");
const unknownTemplate = document.getElementById("unknown-card-template");

async function loadData() {
  const [failures, spans, unknowns, benchmark, guide] = await Promise.all([
    fetch("./data/failures.json").then((r) => r.json()),
    fetch("./data/spans.json").then((r) => r.json()),
    fetch("./data/unknowns.json").then((r) => r.json()),
    fetch("./data/benchmark.json").then((r) => r.json()),
    fetch("./data/guide.json").then((r) => r.json()),
  ]);
  state.failures = failures;
  state.spans = spans;
  state.unknowns = unknowns;
  state.benchmark = benchmark;
  state.guide = guide;
  state.triage = JSON.parse(localStorage.getItem(TRIAGE_KEY) || "{}");
}

function saveTriage() {
  localStorage.setItem(TRIAGE_KEY, JSON.stringify(state.triage));
}

function filteredFailures() {
  const search = els.searchInput.value.trim().toLowerCase();
  const scroll = els.scrollFilter.value;
  const status = els.statusFilter.value;
  const issue = els.issueFilter.value;
  const rarity = els.rarityFilter.value;
  const onlyShort = els.shortFilter.checked;
  const onlyGoldSimilar = els.goldSimilarFilter.checked;
  const onlyTop1Similar = els.top1SimilarFilter.checked;

  let rows = state.failures.filter((row) => {
    if (scroll && row.scroll !== scroll) return false;
    if (status && row.case_status !== status) return false;
    if (issue && row.likely_issue_label !== issue) return false;
    if (rarity && row.rarity_bucket !== rarity) return false;
    if (onlyShort && !row.has_short_top1) return false;
    if (onlyGoldSimilar && !row.gold_in_similar) return false;
    if (onlyTop1Similar && !row.top1_in_similar) return false;
    if (!search) return true;
    const haystack = [
      row.target_word,
      row.context_for_reading,
      row.all_top5,
      row.likely_issue_label,
      row.reader_note,
    ].join(" ").toLowerCase();
    return haystack.includes(search);
  });

  switch (els.sortSelect.value) {
    case "rarity-asc":
      rows = [...rows].sort((a, b) => (a.target_fit_frequency ?? 999999) - (b.target_fit_frequency ?? 999999));
      break;
    case "rarity-desc":
      rows = [...rows].sort((a, b) => (b.target_fit_frequency ?? -1) - (a.target_fit_frequency ?? -1));
      break;
    case "alpha":
      rows = [...rows].sort((a, b) => a.target_word.localeCompare(b.target_word, "he"));
      break;
    default:
      break;
  }
  return rows;
}

function filteredSpans() {
  const search = els.searchInput.value.trim().toLowerCase();
  const scroll = els.scrollFilter.value;
  const status = els.statusFilter.value;
  const issue = els.issueFilter.value;
  const rarity = els.rarityFilter.value;
  const onlyShort = els.shortFilter.checked;

  let rows = state.spans.filter((row) => {
    if (scroll && row.scroll !== scroll) return false;
    if (status && row.case_status !== status) return false;
    if (issue && row.likely_issue_label !== issue) return false;
    if (rarity) {
      const minFreq = Math.min(...row.target_fit_frequencies);
      const rowBucket = minFreq === 0 ? "unseen" : minFreq <= 3 ? "rare" : minFreq <= 20 ? "medium" : "common";
      if (rowBucket !== rarity) return false;
    }
    if (onlyShort) {
      const hasShort = row.slot_details.some((slot) => slot.top1 && slot.top1.length + 1 < slot.gold_word.length);
      if (!hasShort) return false;
    }
    if (!search) return true;
    const haystack = [
      row.target_phrase,
      row.context_for_reading,
      row.top1_phrase,
      row.reader_note,
    ].join(" ").toLowerCase();
    return haystack.includes(search);
  });

  switch (els.sortSelect.value) {
    case "rarity-asc":
      rows = [...rows].sort((a, b) => Math.min(...a.target_fit_frequencies) - Math.min(...b.target_fit_frequencies));
      break;
    case "rarity-desc":
      rows = [...rows].sort((a, b) => Math.max(...b.target_fit_frequencies) - Math.max(...a.target_fit_frequencies));
      break;
    case "alpha":
      rows = [...rows].sort((a, b) => a.target_phrase.localeCompare(b.target_phrase, "he"));
      break;
    default:
      break;
  }
  return rows;
}

function filteredUnknowns() {
  const search = els.searchInput.value.trim().toLowerCase();
  const scroll = els.scrollFilter.value;
  const status = els.statusFilter.value;
  const issue = els.issueFilter.value;

  let rows = state.unknowns.filter((row) => {
    if (scroll && row.scroll !== scroll) return false;
    if (status && row.case_status !== status) return false;
    if (issue && row.likely_issue_label !== issue) return false;
    if (!search) return true;
    const haystack = [
      row.scroll,
      row.context_for_display,
      row.raw_run_text,
      row.flags,
      row.category,
    ].join(" ").toLowerCase();
    return haystack.includes(search);
  });

  switch (els.sortSelect.value) {
    case "alpha":
      rows = [...rows].sort((a, b) => a.scroll.localeCompare(b.scroll, "he") || a.start_index - b.start_index);
      break;
    default:
      rows = [...rows].sort((a, b) => a.scroll.localeCompare(b.scroll, "he") || a.start_index - b.start_index);
      break;
  }
  return rows;
}

function pageRows(rows, mode) {
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  state.page[mode] = Math.min(Math.max(1, state.page[mode]), totalPages);
  const start = (state.page[mode] - 1) * PAGE_SIZE;
  return {
    rows: rows.slice(start, start + PAGE_SIZE),
    total: rows.length,
    start: rows.length ? start + 1 : 0,
    end: Math.min(start + PAGE_SIZE, rows.length),
    totalPages,
  };
}

function renderPager(mode, info) {
  const prefix = mode === "spans" ? "spans" : mode === "unknown" ? "unknown" : "triage";
  els[`${prefix}Count`].textContent = `Showing ${info.start}-${info.end} of ${info.total}`;
  els[`${prefix}Page`].textContent = `${state.page[mode]} / ${info.totalPages}`;
  els[`${prefix}Prev`].disabled = state.page[mode] <= 1;
  els[`${prefix}Next`].disabled = state.page[mode] >= info.totalPages;
}

function populateIssueFilter() {
  const issues = [...new Set([
    ...state.failures.map((row) => row.likely_issue_label),
    ...state.spans.map((row) => row.likely_issue_label),
    ...state.unknowns.map((row) => row.likely_issue_label),
  ])].sort();
  issues.forEach((issue) => {
    const option = document.createElement("option");
    option.value = issue;
    option.textContent = issue;
    els.issueFilter.appendChild(option);
  });
}

function populateScrollFilter() {
  const scrolls = [...new Set([
    ...state.failures.map((row) => row.scroll),
    ...state.spans.map((row) => row.scroll),
    ...state.unknowns.map((row) => row.scroll),
  ])].filter(Boolean).sort((a, b) => a.localeCompare(b, "he"));
  scrolls.forEach((scroll) => {
    const option = document.createElement("option");
    option.value = scroll;
    option.textContent = scroll;
    els.scrollFilter.appendChild(option);
  });
}

function renderHeroStats() {
  const summary = state.guide.summary;
  const entries = [
    ["Cases", summary.total_cases],
    ["Failures", summary.status.miss || 0],
    ["Hits", summary.status.hit || 0],
    ["Short top1", summary.short_top1_count],
    ["Gold in parallels", summary.gold_in_similar_count || 0],
  ];
  els.heroStats.innerHTML = "";
  entries.forEach(([label, value]) => {
    const card = document.createElement("div");
    card.className = "stat-card";
    card.innerHTML = `<div class="eyebrow">${label}</div><div class="value">${value}</div>`;
    els.heroStats.appendChild(card);
  });
}

function predictionPills(predictions) {
  const wrap = document.createDocumentFragment();
  predictions.filter(Boolean).forEach((pred, idx) => {
    const pill = document.createElement("span");
    pill.className = "prediction";
    pill.textContent = `${idx + 1}. ${pred}`;
    wrap.appendChild(pill);
  });
  return wrap;
}

function informativePredictions(predictions) {
  return predictions.filter((pred) => pred && !LOW_INFO_CANDIDATES.has(pred));
}

function getFilteredPredictions(predictions, goldWord, pattern) {
  let list = predictions || [];

  if (els.hideParticlesFilter && els.hideParticlesFilter.checked) {
    list = list.filter((pred) => pred && pred.length > 1);
  }

  if (els.lengthConstraintFilter && els.lengthConstraintFilter.checked) {
    if (goldWord) {
      list = list.filter((pred) => pred && pred.length === goldWord.length);
    } else if (pattern) {
      list = list.filter((pred) => pred && pred.length === pattern.length);
    }
  }

  return list;
}

function renderSimilarPassages(row, { revealOracle = false } = {}) {
  const wrap = document.createElement("div");
  wrap.className = "similar-box";
  const passages = row.similar_passages || [];
  const title = document.createElement("div");
  title.className = "similar-title";
  title.textContent = "Similar fit-corpus passages";
  wrap.appendChild(title);

  if (!passages.length) {
    const empty = document.createElement("p");
    empty.className = "small";
    empty.textContent = "No similar passage found.";
    wrap.appendChild(empty);
    return wrap;
  }

  passages.slice(0, 3).forEach((passage) => {
    const item = document.createElement("div");
    item.className = "similar-item";
    const flags = [];
    if (passage.same_composition) flags.push("same composition");
    if (revealOracle && passage.gold_present) flags.push("gold seen");
    if (passage.top1_present) flags.push("top1 seen");
    if (passage.candidate_hits?.length) flags.push(`candidate: ${passage.candidate_hits.join(", ")}`);
    item.innerHTML = `
      <div class="similar-meta">
        <strong>${passage.book || "unknown"}</strong>
        <span>${passage.sentence_path || ""}</span>
        <span>${passage.composition || "unlabeled composition"}</span>
        <span>score ${Number(passage.score).toFixed(3)}</span>
      </div>
      <p class="similar-text">${passage.text}</p>
      <div class="similar-flags">${flags.map((flag) => `<span>${flag}</span>`).join("")}</div>
    `;
    wrap.appendChild(item);
  });
  return wrap;
}



function getVisualLink(scroll) {
  const s = (scroll || "").trim().toLowerCase();
  if (s.startsWith("1qisa") && s.includes("a")) {
    return "http://dss.collections.imj.org.il/isaiah";
  }
  if (s === "1qs") {
    return "http://dss.collections.imj.org.il/community";
  }
  if (s === "1qphab" || s === "1qphabakkuk") {
    return "http://dss.collections.imj.org.il/habakkuk";
  }
  if (s === "1qm") {
    return "http://dss.collections.imj.org.il/war";
  }
  if (s === "1qha" || s === "1qh" || s.startsWith("1qhodayot")) {
    return "http://dss.collections.imj.org.il/thanksgiving";
  }
  if (s.startsWith("11qt")) {
    return "http://dss.collections.imj.org.il/temple";
  }
  return `https://www.deadseascrolls.org.il/explore-the-archive/search#q=${scroll}`;
}


function renderContextHtml(contextStr) {
  if (!contextStr) return "";
  const words = contextStr.split(" ");
  return words.map((w, idx) => `<span class="ctx-word" data-word-idx="${idx}">${w}</span>`).join(" ");
}

// Highlight context words by attention weight while hovering `trigger`.
// attentions[i] (0..1, normalized) maps to the word at [data-word-idx="i"] inside `card`.
function attachAttentionHover(trigger, card, attentions) {
  if (!attentions || !attentions.length) return;
  trigger.title = "Hover to highlight the context words the model attended to";
  const words = card.querySelectorAll(".ctx-word");
  trigger.addEventListener("mouseenter", () => {
    words.forEach((span) => {
      const idx = parseInt(span.dataset.wordIdx, 10);
      const score = idx < attentions.length ? attentions[idx] : 0;
      span.style.backgroundColor = `rgba(13, 148, 136, ${score * 0.65})`;
    });
  });
  trigger.addEventListener("mouseleave", () => {
    words.forEach((span) => {
      span.style.backgroundColor = "transparent";
    });
  });
}


function renderSpans() {
  const info = pageRows(filteredSpans(), "spans");
  const rows = info.rows;
  els.spansCards.innerHTML = "";
  renderPager("spans", info);
  rows.forEach((row) => {
    const node = spanTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".target").innerHTML = `
      ${row.scroll} · ${row.gap_length}-word gap
      <a href="${getVisualLink(row.scroll)}" class="image-link" target="_blank" title="View manuscript photos">📷 Explore Images</a>
    `;
    node.querySelector(".context").innerHTML = renderContextHtml(row.context_for_reading);
    node.querySelector(".span-meta").innerHTML = `
      <span class="prediction">Top-1 phrase: ${row.top1_phrase || "?"}</span>
      <span class="prediction">Top-5 slot hits: ${row.slot_top5_hits}/${row.gap_length}</span>
    `;
    const slots = node.querySelector(".span-slots");
    row.slot_details.forEach((slot) => {
      const box = document.createElement("div");
      box.className = "slot-box";
      attachAttentionHover(box, node, slot.attentions);

      const filteredCandidates = getFilteredPredictions(slot.top_candidates, slot.gold_word);
      const predictions = filteredCandidates.filter(Boolean).map((candidate, idx) => `<span class="prediction">${idx + 1}. ${candidate}</span>`).join("");
      box.innerHTML = `
        <div class="slot-head">
          <strong>Slot ${slot.slot_index}</strong>
          <span>${slot.hit_top5 ? "gold in top-5" : "gold missed"}</span>
        </div>
        <div class="predictions">${predictions}</div>
      `;
      slots.appendChild(box);
    });
    const oracle = node.querySelector(".oracle");
    const slotOracle = row.slot_details.map((slot) => {
      const top = (slot.top_candidates || []).join(", ");
      return `<div><strong>Slot ${slot.slot_index}:</strong> ${slot.gold_word} <span class="small">| top-5: ${top}</span></div>`;
    }).join("");
    oracle.innerHTML = `
      <strong>Oracle phrase:</strong> ${row.target_phrase}<br />
      <strong>Status:</strong> ${row.case_status}<br />
      <strong>Summary:</strong> ${row.reader_note}
      <div class="span-oracle-slots">${slotOracle}</div>
    `;
    oracle.style.display = "none";
    node.querySelector(".reveal-btn").addEventListener("click", (event) => {
      const shouldShow = oracle.style.display === "none";
      oracle.style.display = shouldShow ? "block" : "none";
      event.currentTarget.textContent = shouldShow ? "Hide oracle" : "Reveal oracle";
      event.currentTarget.setAttribute("aria-expanded", shouldShow ? "true" : "false");
      if (shouldShow) {
        oracle.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });
    els.spansCards.appendChild(node);
  });
}

function renderUnknowns() {
  const info = pageRows(filteredUnknowns(), "unknown");
  const rows = info.rows;
  els.unknownCards.innerHTML = "";
  renderPager("unknown", info);
  rows.forEach((row) => {
    const node = unknownTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".target").innerHTML = `
      ${row.scroll} · run ${row.row_id}
      <a href="${getVisualLink(row.scroll)}" class="image-link" target="_blank" title="View manuscript photos">📷 Explore Images</a>
    `;
    node.querySelector(".context").textContent = row.context_for_reading || row.context_for_display;
    node.querySelector(".span-meta").innerHTML = `
      <span class="prediction">Flags: ${row.flags || "?"}</span>
      <span class="prediction">Run length: ${row.run_length}</span>
      <span class="prediction">Raw center: ${row.raw_run_text}</span>
      ${row.top1_phrase ? `<span class="prediction" title="Context-only prediction">Top-1 raw: ${row.top1_phrase}</span>` : ""}
    `;

    const slots = node.querySelector(".span-slots");
    if (row.slot_details && row.slot_details.length) {
      row.slot_details.forEach((slot) => {
        const box = document.createElement("div");
        box.className = "slot-box";

        // Get filtered lists
        const rawFiltered = getFilteredPredictions(slot.raw_candidates, null, slot.pattern);
        const constrainedFiltered = getFilteredPredictions(slot.constrained_candidates, null, slot.pattern);

        // Combine and deduplicate, putting constrained (matching) ones first
        const combined = [];
        const seen = new Set();

        constrainedFiltered.forEach((cand) => {
          if (cand && !seen.has(cand)) {
            combined.push({ word: cand, matches: true });
            seen.add(cand);
          }
        });

        rawFiltered.forEach((cand) => {
          if (cand && !seen.has(cand)) {
            combined.push({ word: cand, matches: false });
            seen.add(cand);
          }
        });

        const displayList = combined.slice(0, 8);
        const predictionsHTML = displayList.map((item, idx) => {
          if (item.matches) {
            return `<span class="prediction matching-pill" title="Fits physical pattern">${idx + 1}. ${item.word} ✓</span>`;
          } else {
            return `<span class="prediction">${idx + 1}. ${item.word}</span>`;
          }
        }).join("");

        box.innerHTML = `
          <div class="slot-head">
            <strong>Slot ${slot.slot_index}</strong>
            ${slot.pattern ? `<span>Pattern: ${slot.pattern}</span>` : ""}
          </div>
          <div class="predictions">${predictionsHTML}</div>
        `;
        slots.appendChild(box);
      });
    }

    els.unknownCards.appendChild(node);
  });
}

function statusClass(status) {
  switch (status) {
    case "interesting":
      return "good";
    case "data-issue":
      return "issue";
    case "model-issue":
      return "warn";
    default:
      return "neutral";
  }
}

function renderTriage() {
  const info = pageRows(filteredFailures(), "triage");
  const rows = info.rows;
  els.triageCards.innerHTML = "";
  renderPager("triage", info);
  rows.forEach((row) => {
    const node = triageTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".target").innerHTML = `
      ${row.target_word} <span class="card-scroll-badge">(${row.scroll})</span>
      <a href="${getVisualLink(row.scroll)}" class="image-link" target="_blank" title="View manuscript photos">📷 Explore Images</a>
    `;
    node.querySelector(".context").innerHTML = renderContextHtml(row.context_for_reading);
    node.querySelector(".gold-line").innerHTML = `<strong>Gold:</strong> ${row.target_word}`;
    const filteredCandidates = getFilteredPredictions(row.top_candidates, row.target_word);
    const predictionsEl = node.querySelector(".predictions");
    predictionsEl.appendChild(predictionPills(filteredCandidates));
    attachAttentionHover(predictionsEl, node, row.attentions);
    node.querySelector(".similar-passages").appendChild(renderSimilarPassages(row, { revealOracle: true }));
    node.querySelector(".meta").innerHTML = `
      <strong>Status:</strong> ${row.case_status}<br />
      <strong>Likely issue:</strong> ${row.likely_issue_label}<br />
      <strong>Fit frequency:</strong> ${row.target_fit_frequency ?? "?"}<br />
      <strong>Raw category:</strong> ${row.likely_issue}
    `;

    const badge = node.querySelector(".issue-badge");
    badge.textContent = row.likely_issue_label;
    badge.className = `issue-badge ${statusClass("data-issue")}`;

    const statusSelect = node.querySelector(".triage-status");
    const noteArea = node.querySelector(".triage-note");
    const saved = state.triage[row.id] || {};
    statusSelect.value = saved.status || "";
    noteArea.value = saved.note || "";

    statusSelect.addEventListener("change", () => {
      state.triage[row.id] = { ...(state.triage[row.id] || {}), status: statusSelect.value, note: noteArea.value };
      saveTriage();
    });
    noteArea.addEventListener("input", () => {
      state.triage[row.id] = { ...(state.triage[row.id] || {}), status: statusSelect.value, note: noteArea.value };
      saveTriage();
    });
    els.triageCards.appendChild(node);
  });
}

function renderBenchmark() {
  // Render sequence accuracy comparison
  const seqComparisonDiv = document.getElementById("benchmark-sequence-comparison");
  if (seqComparisonDiv) {
    const seqTable = document.createElement("table");
    seqTable.className = "metric-table";
    seqTable.innerHTML = `
      <thead>
        <tr>
          <th>Metric Type (MsBERT + span-ft-refined)</th>
          <th>1 Word</th>
          <th>2 Words</th>
          <th>3 Words</th>
          <th>4-5 Words</th>
          <th>6+ Words</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Slot-Level Accuracy (Individual Words)</strong></td>
          <td>22.1%</td>
          <td>14.3%</td>
          <td>12.9%</td>
          <td>11.2%</td>
          <td>8.2%</td>
        </tr>
        <tr>
          <td><strong>Sequence Accuracy (Parallel MLM Baseline)</strong></td>
          <td>22.1%</td>
          <td>5.0%</td>
          <td>2.1%</td>
          <td>0.7%</td>
          <td>0.0%</td>
        </tr>
        <tr style="background: rgba(13, 148, 136, 0.04); font-weight: 600; color: var(--primary-hover);">
          <td><strong>Sequence Accuracy (Autoregressive Beam Search) 🏆</strong></td>
          <td><strong>22.1%</strong></td>
          <td><strong>7.9% <span style="color:#047857; font-size:11px;">(+58% rel)</span></strong></td>
          <td><strong>4.3% <span style="color:#047857; font-size:11px;">(+105% rel)</span></strong></td>
          <td><strong>2.1% <span style="color:#047857; font-size:11px;">(+200% rel)</span></strong></td>
          <td><strong>0.7% <span style="color:#047857; font-size:11px;">(recovered)</span></strong></td>
        </tr>
      </tbody>
    `;
    seqComparisonDiv.innerHTML = "";
    seqComparisonDiv.appendChild(seqTable);
  }

  const rows = state.benchmark.top_10;
  const table = document.createElement("table");
  table.className = "metric-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Model</th>
        <th>1</th>
        <th>2</th>
        <th>3</th>
        <th>4-5</th>
        <th>6+</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector("tbody");
  Object.entries(rows).forEach(([label, metrics]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${label}</td>
      <td>${metrics["1"].toFixed(1)}%</td>
      <td>${metrics["2"].toFixed(1)}%</td>
      <td>${metrics["3"].toFixed(1)}%</td>
      <td>${metrics["4-5"].toFixed(1)}%</td>
      <td>${metrics["6+"].toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });
  els.benchmarkTop10.innerHTML = "";
  els.benchmarkTop10.appendChild(table);

  // Render biblical contrast results
  const bibDiv = document.getElementById("benchmark-biblical");
  if (bibDiv) {
    const bibTable = document.createElement("table");
    bibTable.className = "metric-table";
    bibTable.innerHTML = `
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="2" style="text-align:center;">1 Word</th>
          <th colspan="2" style="text-align:center;">2 Words</th>
          <th colspan="2" style="text-align:center;">3 Words</th>
          <th colspan="2" style="text-align:center;">4-5 Words</th>
          <th colspan="2" style="text-align:center;">6+ Words</th>
        </tr>
        <tr>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-1</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-10</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-1</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-10</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-1</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-10</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-1</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-10</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-1</th>
          <th style="text-align:center; font-size:11px; font-weight:normal;">Top-10</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>MsBERT base (Baseline)</strong></td>
          <td style="text-align:center;">32.5%</td>
          <td style="text-align:center;">57.5%</td>
          <td style="text-align:center;">20.0%</td>
          <td style="text-align:center;">48.8%</td>
          <td style="text-align:center;">10.8%</td>
          <td style="text-align:center;">37.5%</td>
          <td style="text-align:center;">14.9%</td>
          <td style="text-align:center;">36.2%</td>
          <td style="text-align:center;">12.9%</td>
          <td style="text-align:center;">33.2%</td>
        </tr>
        <tr>
          <td><strong>MsBERT ft-SPAN-refined</strong></td>
          <td style="text-align:center;">30.0%</td>
          <td style="text-align:center;">52.5%</td>
          <td style="text-align:center;">22.5%</td>
          <td style="text-align:center;">45.0%</td>
          <td style="text-align:center;">8.3%</td>
          <td style="text-align:center;">35.0%</td>
          <td style="text-align:center;">13.2%</td>
          <td style="text-align:center;">34.5%</td>
          <td style="text-align:center;">11.2%</td>
          <td style="text-align:center;">30.5%</td>
        </tr>
        <tr>
          <td><strong>BEREL base</strong></td>
          <td style="text-align:center;">27.5%</td>
          <td style="text-align:center;">50.0%</td>
          <td style="text-align:center;">17.5%</td>
          <td style="text-align:center;">40.0%</td>
          <td style="text-align:center;">5.8%</td>
          <td style="text-align:center;">23.3%</td>
          <td style="text-align:center;">10.3%</td>
          <td style="text-align:center;">27.0%</td>
          <td style="text-align:center;">7.8%</td>
          <td style="text-align:center;">26.1%</td>
        </tr>
      </tbody>
    `;
    bibDiv.innerHTML = "";
    bibDiv.appendChild(bibTable);
  }

  const retrieval = state.benchmark.retrieval;
  const parallel = state.benchmark.parallel_lookup;
  els.benchmarkRetrieval.innerHTML = "";
  if (!retrieval && !parallel) {
    els.benchmarkRetrieval.textContent = "No retrieval benchmark loaded.";
    return;
  }
  const sections = [];
  sections.push(`
    <div class="benchmark-subsection">
      <h5>Validated Train-Only RAG Ablation</h5>
      <p class="small">Preserved non-biblical training text only; α=0.5 selected on dev. Held-out editorial labels are used only for scoring.</p>
      <table class="metric-table">
        <thead>
          <tr>
            <th>Held-out unit</th>
            <th>N</th>
            <th>MLM Top-10</th>
            <th>MLM + RAG Top-10</th>
            <th>Delta</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>QD single-word targets</td><td>74</td><td>63.5%</td><td>63.5%</td><td>0.0</td></tr>
          <tr><td>TF single-word spans</td><td>25</td><td>60.0%</td><td>64.0%</td><td>+4.0</td></tr>
          <tr><td>TF slots in multiword spans</td><td>440</td><td>41.4%</td><td>41.8%</td><td>+0.5</td></tr>
          <tr><td>TF exact multiword sequences</td><td>100</td><td>7.0%</td><td>9.0%</td><td>+2.0</td></tr>
        </tbody>
      </table>
      <p class="small">The exact-sequence score requires every word to match in order. These are modest ablation gains, not evidence that retrieval always helps.</p>
    </div>
  `);
  if (retrieval?.conditions) {
    const any = retrieval.conditions.fit_any_composition;
    const cross = retrieval.conditions.fit_cross_composition_only;
    sections.push(`
      <div class="benchmark-subsection">
        <h5>Legacy Exploratory Retrieval</h5>
        <p class="small">Retained for transparency. This older experiment is not the recommended method: its fixed reranker reduced Top-1, motivating the clean dev-tuned ablation above.</p>
        <table class="metric-table">
          <thead>
            <tr>
              <th>Condition</th>
              <th>Passages</th>
              <th>Gold In Retrieved</th>
              <th>Top-1 Before</th>
              <th>Top-1 After</th>
              <th>Delta</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Any composition</td>
              <td>${any.coverage.cases_with_passages_pct}%</td>
              <td>${any.coverage.gold_present_all_pct}%</td>
              <td>${any.rerank.baseline_top1_pct}%</td>
              <td>${any.rerank.reranked_top1_pct}%</td>
              <td>${any.rerank.delta_pts > 0 ? "+" : ""}${any.rerank.delta_pts}</td>
            </tr>
            <tr>
              <td>Cross composition only</td>
              <td>${cross.coverage.cases_with_passages_pct}%</td>
              <td>${cross.coverage.gold_present_all_pct}%</td>
              <td>${cross.rerank.baseline_top1_pct}%</td>
              <td>${cross.rerank.reranked_top1_pct}%</td>
              <td>${cross.rerank.delta_pts > 0 ? "+" : ""}${cross.rerank.delta_pts}</td>
            </tr>
          </tbody>
        </table>
      </div>
    `);
  }
  if (parallel) {
    const strictAny = parallel.strict_preserved_any_composition;
    const strictCross = parallel.strict_preserved_cross_composition;
    const relaxedAny = parallel.relaxed_preserved_target_any_composition;
    sections.push(`
      <div class="benchmark-subsection">
        <h5>Exact Parallel Lookup</h5>
        <p class="small">This is a researcher-assist signal, not a pure language-model benchmark. The strict setting uses only preserved witness text.</p>
        <table class="metric-table">
          <thead>
            <tr>
              <th>Condition</th>
              <th>Matched Cases</th>
              <th>Correct / All</th>
              <th>Correct / Matched</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Strict preserved, any composition</td>
              <td>${strictAny.matched_pct}%</td>
              <td>${strictAny.correct_pct_over_all}%</td>
              <td>${strictAny.correct_pct_over_matched}%</td>
            </tr>
            <tr>
              <td>Strict preserved, cross composition</td>
              <td>${strictCross.matched_pct}%</td>
              <td>${strictCross.correct_pct_over_all}%</td>
              <td>${strictCross.correct_pct_over_matched}%</td>
            </tr>
            <tr>
              <td>Relaxed target preserved, any composition</td>
              <td>${relaxedAny.matched_pct}%</td>
              <td>${relaxedAny.correct_pct_over_all}%</td>
              <td>${relaxedAny.correct_pct_over_matched}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    `);
  }
  els.benchmarkRetrieval.innerHTML = sections.join("");
}

function renderAll() {
  renderHeroStats();
  renderSpans();
  renderUnknowns();
  renderTriage();
  renderBenchmark();
}

function resetPages() {
  state.page.spans = 1;
  state.page.unknown = 1;
  state.page.triage = 1;
}

function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".content-mode").forEach((section) => {
    section.classList.toggle("active", section.id === `mode-${mode}`);
  });
  document.querySelectorAll("#mode-switch button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
}

function bindEvents() {
  [els.searchInput, els.scrollFilter, els.statusFilter, els.issueFilter, els.rarityFilter, els.shortFilter, els.goldSimilarFilter, els.top1SimilarFilter, els.lengthConstraintFilter, els.hideParticlesFilter, els.sortSelect].forEach((el) => {
    el.addEventListener("input", () => {
      resetPages();
      renderAll();
    });
    el.addEventListener("change", () => {
      resetPages();
      renderAll();
    });
  });
  els.spansPrev.addEventListener("click", () => {
    state.page.spans -= 1;
    renderAll();
  });
  els.spansNext.addEventListener("click", () => {
    state.page.spans += 1;
    renderAll();
  });
  els.unknownPrev.addEventListener("click", () => {
    state.page.unknown -= 1;
    renderAll();
  });
  els.unknownNext.addEventListener("click", () => {
    state.page.unknown += 1;
    renderAll();
  });
  els.triagePrev.addEventListener("click", () => {
    state.page.triage -= 1;
    renderAll();
  });
  els.triageNext.addEventListener("click", () => {
    state.page.triage += 1;
    renderAll();
  });
  els.modeSwitch.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-mode]");
    if (!button) return;
    switchMode(button.dataset.mode);
  });
}

async function main() {
  await loadData();
  populateIssueFilter();
  populateScrollFilter();
  bindEvents();
  renderAll();
  switchMode("spans");
}

main();
