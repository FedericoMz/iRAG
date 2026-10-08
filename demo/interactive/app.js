"use strict";

// A deterministic, in-browser concept. No provider, CRM, or payment calls.
const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
const seeds = [
  { id: "1842", title: "Annual plan refund after 22 days", answer: "A full refund is available within 30 days of purchase.", age: 700, similarity: .96, author: "Sam · Support", label: "Earlier policy", kind: "seed" },
  { id: "4921", title: "Refund requested after 18 days", answer: "Full refunds are available within 14 days of purchase. After that window, we can help you review alternative plan options.", age: 14, similarity: .89, author: "Maya · Support", label: "Updated policy", kind: "seed" },
];
const chapters = [
  { label: "01 / TIME-AWARE RETRIEVAL", headline: "Familiar question.<br><em>Different answer.</em>", intro: "The refund policy changed. Two past resolutions disagree.<br>Explore which precedent should guide the next decision.", id: "5943", name: "Alex Lewis", initials: "AL", title: "A refund after 20 days?", message: "Hi, I purchased an annual subscription 20 days ago, but it isn’t the right fit for our team. Can I still receive a full refund?", facts: ["Annual subscription", "Purchased 20 days ago"], lesson: "Similarity finds the familiar answer. Temporal weighting changes which answer comes first." },
  { label: "02 / INCREMENTAL MEMORY", headline: "Your decision.<br><em>The next precedent.</em>", intro: "A second customer has a similar question.<br>The resolution you just approved is now available to retrieve.", id: "6081", name: "Casey Morgan", initials: "CM", title: "Does the refund window still apply?", message: "We bought an annual subscription 20 days ago and would like to cancel. Could you clarify whether we qualify for a full refund?", facts: ["Annual subscription", "Purchased 20 days ago"], lesson: "The memory contains the final reply you saved, including your edits. Human supervision matters because errors can propagate too." },
  { label: "03 / GOVERNED AUTONOMY", headline: "Trust opens the door.<br><em>It also closes it.</em>", intro: "Explore a routine case after enough illustrated reviews.<br>Then introduce an exception and watch authority return to a human.", id: "6127", name: "Jordan Kim", initials: "JK", title: "Cancel my subscription after 7 days", message: "Hi, we subscribed last week and have decided to cancel. Can you confirm whether a purchase made 7 days ago qualifies for a full refund?", facts: ["Annual subscription", "Purchased 7 days ago"], lesson: "Autonomy is conditional. An explicit human intervention can return decision-making authority to the support team." },
  { label: "04 / EVIDENCE BOUNDARIES", headline: "A useful answer.<br><em>Sometimes, a handoff.</em>", intro: "This request falls outside the available resolution memory.<br>See how an evidence gap becomes a clear next action.", id: "6018", name: "Noor Patel", initials: "NP", title: "Restore a deleted integration?", message: "Our custom warehouse integration was deleted three years ago. Can you restore its original configuration and historical credentials?", facts: ["Custom integration", "Deleted three years ago"], lesson: "Abstention does not become a resolved precedent. A verified human resolution can be added after the investigation." },
];

function freshState() {
  return { chapter: 0, temporal: true, halfLife: 500, memory: seeds.map((item) => ({ ...item })), drafts: {}, saved: {}, events: [], numerator: 8.6, denominator: 10, observations: 18, autoResolved: false, rolledBack: false, escalated: false, activeTab: "ticket" };
}
let state = freshState();
let toastTimer;

function agreement() { return state.numerator / state.denominator; }
function hasTrust() { return state.observations >= 20 && agreement() > .8; }
function observe(agrees) {
  state.numerator = .9 * state.numerator + Number(agrees);
  state.denominator = .9 * state.denominator + 1;
  state.observations += 1;
}
function log(title, detail) {
  state.events.push({ title, detail, sequence: state.events.length + 1 });
}
function notify(message) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").classList.add("visible");
  toastTimer = setTimeout(() => $("#toast").classList.remove("visible"), 4500);
}
function rankedMemory() {
  return state.memory.map((record) => {
    const similarity = record.kind === "human" ? .98 : record.similarity;
    const weight = state.temporal ? Math.pow(.5, record.age / state.halfLife) : 1;
    return { ...record, similarity, weight, score: similarity * weight };
  }).sort((a, b) => b.score - a.score || a.age - b.age);
}
function draftFor(chapter) {
  const existing = state.drafts[chapter];
  if (existing?.dirty) return existing;
  const top = rankedMemory()[0];
  const text = top.id === "1842"
    ? "Your purchase was made 20 days ago, which falls within the 30-day refund window described in this precedent. You are eligible for a full refund."
    : top.kind === "human" ? top.answer
      : "Under our current policy, full refunds are available within 14 days of purchase. Your purchase was made 20 days ago, so it is outside the refund window. I can help you review alternative plan options.";
  state.drafts[chapter] = { text, dirty: false, agrees: true, sourceId: top.id };
  return state.drafts[chapter];
}

function render() {
  const chapter = chapters[state.chapter];
  $("#chapter-label").textContent = chapter.label;
  $("#headline").innerHTML = chapter.headline;
  $("#intro").innerHTML = chapter.intro.replaceAll("<br>", "<br> ");
  $("#ticket-id").textContent = `#${chapter.id}`;
  $("#customer-name").textContent = chapter.name;
  $("#customer-detail").textContent = state.chapter === 3 ? "Enterprise plan · Integrations" : "Professional plan · Billing & refunds";
  $("#avatar").textContent = chapter.initials;
  $("#ticket-title").textContent = chapter.title;
  $("#ticket-message").textContent = chapter.message;
  $("#ticket-facts").innerHTML = chapter.facts.map((fact) => `<span>${escapeHTML(fact)}</span>`).join("");
  $("#lesson-text").textContent = chapter.lesson;
  document.querySelectorAll("[data-chapter]").forEach((button) => {
    if (Number(button.dataset.chapter) === state.chapter) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  const autonomous = hasTrust() && !state.rolledBack;
  $("#authority-name").textContent = state.chapter === 3 ? "Human decides · evidence gap" : autonomous ? "Model may decide" : "Human decides";
  $("#state-assist").classList.toggle("selected", !autonomous);
  $("#state-auto").classList.toggle("selected", autonomous);
  $("#fea").textContent = `${Math.round(agreement() * 100)}%`;
  $("#observation-count").textContent = `${state.observations} illustrated reviews`;
  $("#memory-count").textContent = state.memory.length;
  const additions = state.memory.filter((item) => item.kind !== "seed").length;
  $("#memory-caption").textContent = additions ? `${additions} new resolution${additions === 1 ? "" : "s"} added in this session. Every final decision leaves a trace.` : "Two seeded precedents. Your next decision belongs here.";
  $("#audit-count").textContent = state.events.length;
  const status = state.saved[state.chapter] ? "Resolved by you" : state.chapter === 2 && state.autoResolved ? "Simulated resolution" : state.chapter === 3 && state.escalated ? "Escalated" : state.chapter === 3 ? "Needs specialist" : "Needs reply";
  $("#ticket-status").textContent = status;
  $("#ticket-status").classList.toggle("success", Boolean(state.saved[state.chapter]) || (state.chapter === 2 && state.autoResolved) || (state.chapter === 3 && state.escalated));
  renderDecision();
  renderEvidence();
  setTab(state.activeTab);
}

function renderDecision() {
  let content;
  if (state.chapter < 2) {
    const saved = state.saved[state.chapter];
    if (saved) {
      content = `<div class="result-box"><span class="eyebrow">✓ SAVED TO RESOLUTION MEMORY</span><h3>Your decision is now evidence.</h3><p>${escapeHTML(saved.answer)}</p></div><p class="explanation">Precedent #${saved.id} · Approved by you · ${state.memory.length} records in memory. ${saved.agrees ? "Agreement recorded." : "Correction recorded in the reliability history."}</p><button class="primary wide" data-action="next">${state.chapter === 0 ? "Retrieve it on the next ticket" : "Explore the autonomy gate"}<span>→</span></button>`;
    } else if (state.chapter === 1 && !state.saved[0]) {
      content = `<div class="result-box"><span class="eyebrow">THE LOOP STARTS WITH YOU</span><h3>Give memory something to learn.</h3><p>Resolve Alex’s ticket first. Your exact final answer will appear here as a new precedent.</p></div><button class="primary wide" data-action="first">Open the first ticket <span>→</span></button>`;
    } else {
      const draft = draftFor(state.chapter);
      content = `<div class="section-label"><span><span class="spark">✳</span> ${draft.dirty ? "Your edited reply" : "Evidence-grounded draft"}</span><button class="source-link" data-record="${draft.sourceId}">Source #${draft.sourceId} ↗</button></div><label class="sr-only" for="reply">Final reply to ${escapeHTML(chapters[state.chapter].name)}</label><textarea class="draft" id="reply" spellcheck="true">${escapeHTML(draft.text)}</textarea><label class="review-check"><input id="agrees" type="checkbox" ${draft.agrees ? "checked" : ""}><span>My final reply agrees with the suggestion.<br>Uncheck if you corrected its conclusion.</span></label><div class="decision-footer"><small>Review or edit, then save.<br> Only your final reply enters memory.</small><button class="primary" id="save-reply" data-action="save" ${!draft.text.trim() ? "disabled" : ""}>Resolve &amp; remember <span>↗</span></button></div>`;
    }
  } else if (state.chapter === 2) {
    if (state.rolledBack) {
      content = `<div class="result-box warning"><span class="eyebrow">↶ HUMAN CONTROL RESTORED</span><h3>The exception changes the next decision.</h3><p>A billing specialist has flagged annual-plan refunds for manual review during a payment migration. Future cases require human approval.</p></div><p class="explanation">The earlier simulated reply remains in the audit trail. The intervention changes future authority; it does not undo a previous action or silently rewrite memory.</p><button class="primary wide" data-action="next">Test an unfamiliar request <span>→</span></button>`;
    } else if (state.autoResolved) {
      content = `<div class="result-box"><span class="eyebrow">✓ SIMULATED AUTONOMOUS REPLY</span><h3>A routine case, resolved.</h3><p>Your purchase was made 7 days ago and falls within the 14-day refund window. You are eligible for a full refund.</p></div><p class="explanation">Saved as a model-finalized resolution with its evidence and authority state. No customer message or refund was sent.</p><button class="secondary wide" data-action="intervene">Introduce a policy exception <span>↶</span></button>`;
    } else {
      content = `<div class="section-label"><span><span class="spark">✳</span> Before the model can act</span><span class="muted">Illustrative gate</span></div><ul class="check-list"><li><span>Recent agreement above 80%</span><strong class="${agreement() > .8 ? "" : "failed"}">${Math.round(agreement() * 100)}% ${agreement() > .8 ? "✓" : "· Review needed"}</strong></li><li><span>At least 20 review observations</span><strong class="${state.observations >= 20 ? "" : "failed"}">${state.observations} / 20</strong></li><li><span>Current policy evidence selected</span><strong class="${state.temporal ? "" : "failed"}">${state.temporal ? "#4921 ✓" : "Enable temporal weighting"}</strong></li></ul><p class="explanation">This scenario uses an explicit 14-day policy fixture. Retrieval ranking alone does not validate a policy or authorize a real-world action.</p>${!hasTrust() ? '<button class="secondary wide" data-action="reviews">Replay 12 illustrated agreeing reviews <span>↗</span></button>' : ""}<button class="primary wide" data-action="auto" ${!hasTrust() || !state.temporal ? "disabled" : ""}>Simulate autonomous reply <span>→</span></button>`;
    }
  } else if (state.escalated) {
    content = `<div class="result-box"><span class="eyebrow">✓ HANDED TO INTEGRATIONS ENGINEERING</span><h3>Uncertainty, made actionable.</h3><p>The specialist receives the original request, the evidence gap, and a request to verify recoverability. No unsupported promise was sent.</p></div><p class="explanation">This session: ${state.memory.length - 2} resolutions added · ${state.events.length} recorded events · ${state.rolledBack ? "Human control restored" : "Evidence boundary respected"}.</p><button class="secondary wide" data-action="audit">Explore the decision trail <span>↗</span></button><button class="text-button restart-inline" data-action="reset">Start a fresh session ↺</button>`;
  } else {
    content = `<div class="result-box"><span class="eyebrow">NO SUPPORTED ANSWER</span><h3>Ask someone who can verify.</h3><p>The memory contains billing resolutions. None establishes whether this deleted integration or its credentials can be recovered.</p></div><p class="explanation">Recommended owner: Integrations Engineering.<br>Include the customer’s question and the missing evidence.</p><button class="primary wide" data-action="escalate">Escalate with context <span>↗</span></button>`;
  }
  $("#decision-area").innerHTML = `<div class="decision-section">${content}</div>`;
}

function evidenceCard(record, index) {
  return `<button class="precedent ${index === 0 ? "winner" : ""}" data-record="${record.id}" aria-label="Inspect precedent ${record.id}"><div class="precedent-top"><strong>${index === 0 ? "01 · FIRST RETRIEVED" : `0${index + 1} · ALTERNATIVE`}</strong><span>#${record.id} ↗</span></div><h3>${escapeHTML(record.title)}</h3><p>${escapeHTML(record.answer)}</p><div class="scores"><span>Match <b>${Math.round(record.similarity * 100)}%</b></span><span>×</span><span>Weight <b>${record.weight.toFixed(2)}</b></span><span>=</span><span>Score <b>${record.score.toFixed(2)}</b></span></div><div class="score-bar" aria-hidden="true"><span style="width:${record.score * 100}%"></span></div></button>`;
}
function renderEvidence() {
  if (state.chapter === 3) {
    $("#evidence-count").textContent = "0 eligible precedents";
    $("#evidence-content").innerHTML = `<div class="evidence-empty"><div class="empty-symbol">∅</div><span class="eyebrow">THE BOUNDARY OF MEMORY</span><h3>No relevant resolution.</h3><p>Related words are not enough.<br>A resolved billing ticket cannot support an answer about integration recovery.</p></div><div class="rejected-item"><b>Billing and refund precedents</b>Excluded from this scenario’s retrieval set: below the illustrated semantic relevance threshold.</div><div class="ranking-note"><span>↗</span><p>Abstention routes the decision to a human. It is an outcome, not an authority stage.</p></div><div class="evidence-footnote">This evidence gap is scripted. The demo does not compute embeddings or call a model.</div>`;
    return;
  }
  const records = state.chapter === 2
    ? rankedMemory().filter((record) => record.id === "4921")
    : rankedMemory().filter((record) => record.id !== chapters[state.chapter].id).slice(0, 3);
  $("#evidence-count").textContent = state.chapter === 2 ? "Explicit policy fixture" : `${records.length} precedents`;
  const first = records[0];
  const note = !state.temporal ? "Decay is off. In the first ticket, the older 30-day answer wins on similarity alone."
    : state.chapter === 1 && state.saved[0] ? "Your saved answer is now part of the retrieval set. Open it to inspect exactly what entered memory."
      : state.chapter === 2 ? "This case references the 14-day policy fixture. The simulated gate also requires enough review evidence."
        : "The newer resolution ranks first. Its wording is less similar, but its insertion age is much lower.";
  $("#evidence-content").innerHTML = `<div class="retrieval-controls"><div class="toggle-row"><div><strong>Temporal weighting</strong><p>Give recent decisions more influence.</p></div><button class="switch" id="temporal-toggle" role="switch" aria-checked="${state.temporal}" aria-label="Temporal weighting"><span></span></button></div><div class="slider-row"><label for="half-life">Decay half-life</label><output id="half-life-output" for="half-life">${state.halfLife} insertions</output><input id="half-life" type="range" min="100" max="1200" step="50" value="${state.halfLife}" ${!state.temporal ? "disabled" : ""}></div></div><div class="ranking-heading"><span>${state.chapter === 2 ? "SCENARIO EVIDENCE" : "RANKED RESOLUTIONS"}</span><span>Inspect any card ↗</span></div><div class="evidence-list">${records.map(evidenceCard).join("")}</div><div class="ranking-note ${!state.temporal ? "warning" : ""}"><span>${!state.temporal ? "!" : "↗"}</span><p>${note}</p></div><div class="evidence-footnote">Score = similarity × 0.5<sup>(insertion age / half-life)</sup>.<br>Scores illustrate retrieval priority, not correctness or confidence.${first?.kind === "human" ? " New resolutions retain your exact wording." : ""}</div>`;
}

function setTab(tab, focus = false) {
  state.activeTab = tab;
  $(".workspace").dataset.activeTab = tab;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    const selected = button.dataset.tab === tab;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
}
function navigate(chapter) {
  state.chapter = Math.min(3, Math.max(0, chapter));
  state.activeTab = "ticket";
  render();
  $("#headline").setAttribute("tabindex", "-1");
  $("#headline").focus({ preventScroll: true });
  if (window.matchMedia("(max-width: 760px)").matches) $(".page-heading").scrollIntoView({ block: "start" });
}
function showDialog(title, content, kicker = "UNDER THE SURFACE") {
  $("#dialog-title").textContent = title;
  $("#dialog-kicker").textContent = kicker;
  $("#dialog-body").innerHTML = content;
  $("#detail-dialog").showModal();
}
function openRecord(id) {
  const record = state.memory.find((item) => item.id === id);
  if (!record) return;
  const ranked = rankedMemory().find((item) => item.id === id);
  showDialog(`Resolution #${id}`, `<p class="eyebrow">${escapeHTML(record.label)} · ${escapeHTML(record.author)}</p><p><strong>${escapeHTML(record.title)}</strong></p><p>${escapeHTML(record.answer)}</p><div class="formula">Similarity ${ranked.similarity.toFixed(2)} × weight ${ranked.weight.toFixed(3)} = ${ranked.score.toFixed(3)}<br>Insertion age: ${record.age} · Half-life: ${state.temporal ? state.halfLife : "decay disabled"}</div><p>${record.kind === "seed" ? "A synthetic historical resolution. The policy label is supplied by the scenario, not inferred from recency." : "This is the exact final reply saved in this session. Provenance is retained; inclusion in memory does not certify correctness."}</p>`, "EVIDENCE & PROVENANCE");
}
function showAudit() {
  showDialog("Every decision leaves a trace.", state.events.length
    ? state.events.map((event) => `<div class="dialog-record"><small>EVENT ${String(event.sequence).padStart(2, "0")}</small><h3>${escapeHTML(event.title)}</h3><p>${escapeHTML(event.detail)}</p></div>`).join("")
    : "<p>No decisions yet. Change retrieval, save a reply, or escalate a case to start the trail.</p>", "SESSION ACTIVITY");
}
function insertResolution(record) {
  state.memory.forEach((item) => { item.age += 1; });
  state.memory.push({ ...record, age: 0 });
}
function saveReply() {
  if (state.chapter > 1 || state.saved[state.chapter] || (state.chapter === 1 && !state.saved[0])) return;
  const draft = state.drafts[state.chapter];
  if (!draft?.text.trim()) return;
  const chapter = chapters[state.chapter];
  const record = { id: chapter.id, title: chapter.title, answer: draft.text.trim(), author: "You · Human review", label: "Human-approved resolution", kind: "human", similarity: .98, agrees: draft.agrees };
  insertResolution(record);
  state.saved[state.chapter] = record;
  observe(draft.agrees);
  log(`Human finalized #${chapter.id}`, `Saved the final reply as a new precedent. Source #${draft.sourceId}. ${draft.agrees ? "Human reported agreement" : "Human reported a correction"}. Recent agreement: ${Math.round(agreement() * 100)}%.`);
  render();
  notify(`Resolution #${chapter.id} added to memory.`);
  $("[data-action=next]").focus({ preventScroll: true });
}
function reset() {
  state = freshState();
  clearTimeout(toastTimer);
  $("#toast").classList.remove("visible");
  if ($("#detail-dialog").open) $("#detail-dialog").close();
  navigate(0);
  notify("Fresh session. Two precedents, ready to explore.");
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || button.disabled) return;
  if (button.dataset.chapter !== undefined) return navigate(Number(button.dataset.chapter));
  if (button.dataset.tab) return setTab(button.dataset.tab);
  if (button.dataset.record) return openRecord(button.dataset.record);
  if (button.id === "temporal-toggle") {
    state.temporal = !state.temporal;
    log("Retrieval weighting changed", `Temporal decay ${state.temporal ? "enabled" : "disabled"}. Half-life: ${state.halfLife} insertions.`);
    render();
    $("#temporal-toggle").focus({ preventScroll: true });
    return;
  }
  switch (button.dataset.action) {
    case "save": saveReply(); break;
    case "first": navigate(0); break;
    case "next": navigate(state.chapter + 1); break;
    case "reviews":
      if (hasTrust()) break;
      for (let i = 0; i < 12; i += 1) observe(true);
      log("Replayed illustrated review history", "12 synthetic agreeing observations updated FEA. No new ticket resolutions were inserted. This is a demonstration shortcut, not measured performance.");
      render(); notify("12 illustrated reviews replayed. Inspect the gate again.");
      $("[data-action=auto]").focus({ preventScroll: true });
      break;
    case "auto":
      if (!hasTrust() || !state.temporal || state.rolledBack || state.autoResolved) break;
      state.autoResolved = true;
      insertResolution({ id: "6127", title: "Refund eligibility after 7 days", answer: "A purchase made 7 days ago falls within the 14-day refund window and is eligible for a full refund.", author: "iRAG · Simulated model decision", label: "Model-finalized resolution", kind: "model", similarity: .84 });
      log("Model finalized #6127 in simulation", "Evidence #4921. Illustrative review gate passed. Resolution appended with model provenance. No human agreement observation was added for the model’s own reply.");
      render(); notify("Simulated reply recorded. Now test a human intervention.");
      $("[data-action=intervene]").focus({ preventScroll: true });
      break;
    case "intervene":
      if (!state.autoResolved || state.rolledBack) break;
      state.rolledBack = true;
      log("Human intervention restored Assist", "A specialist flagged annual-plan refunds for manual review during a payment migration. Future autonomous replies paused. Prior decisions and FEA remain unchanged; no new human/model comparison occurred.");
      render(); notify("Authority returned to a human. History is preserved.");
      $("[data-action=next]").focus({ preventScroll: true });
      break;
    case "escalate":
      if (state.escalated) break;
      state.escalated = true;
      log("Escalated #6018 with context", "Owner: Integrations Engineering. Request: verify whether the deleted integration is recoverable. No supporting precedent. No answer inserted into memory and no FEA observation added.");
      render(); notify("Simulated handoff prepared for Integrations Engineering.");
      $("[data-action=audit]").focus({ preventScroll: true });
      break;
    case "audit": showAudit(); break;
    case "reset": reset(); break;
  }
});
document.addEventListener("input", (event) => {
  if (event.target.id === "reply") {
    const draft = state.drafts[state.chapter];
    draft.text = event.target.value;
    draft.dirty = true;
    $("#save-reply").disabled = !draft.text.trim();
  }
  if (event.target.id === "half-life") {
    state.halfLife = Number(event.target.value);
    $("#half-life-output").textContent = `${state.halfLife} insertions`;
  }
});
document.addEventListener("change", (event) => {
  if (event.target.id === "agrees") {
    state.drafts[state.chapter].agrees = event.target.checked;
    state.drafts[state.chapter].dirty = true;
  }
  if (event.target.id === "half-life") {
    log("Decay half-life changed", `${state.halfLife} insertions. Calendar age is not used in this demo’s retrieval formula.`);
    render();
    $("#half-life").focus({ preventScroll: true });
  }
});
$(".mobile-tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  event.preventDefault();
  setTab(event.key === "Home" ? "ticket" : event.key === "End" ? "evidence" : state.activeTab === "ticket" ? "evidence" : "ticket", true);
});
$("#dialog-close").addEventListener("click", () => $("#detail-dialog").close());
$("#detail-dialog").addEventListener("click", (event) => {
  if (event.target !== $("#detail-dialog")) return;
  const bounds = event.target.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) event.target.close();
});
$("#audit-open").addEventListener("click", showAudit);
$("#reset").addEventListener("click", reset);
$("#memory-open").addEventListener("click", () => showDialog("Resolved work becomes memory.", state.memory.map((record) => `<div class="dialog-record"><small>#${record.id} · ${escapeHTML(record.author)}</small><h3>${escapeHTML(record.title)}</h3><p>${escapeHTML(record.answer)}</p></div>`).join(""), `${state.memory.length} RESOLUTIONS IN THIS SESSION`));
$("#reliability-open").addEventListener("click", () => showDialog("Agreement, with a shorter memory.", `<p>Fading Empirical Accuracy (FEA) weights recent agreement more heavily. Here, each human review discounts earlier observations by 0.9. Agreement adds 1 to the numerator; a correction adds 0. Both add 1 to the denominator.</p><div class="formula">numerator = 0.9 × previous numerator + agreement<br>denominator = 0.9 × previous denominator + 1<br>FEA = numerator / denominator</div><p>This session starts with an illustrated weighted history: numerator 8.6, denominator 10, and 18 observations. The demo gate requires more than 80% agreement and at least 20 observations.</p><p>These are demo settings, not paper defaults. You report whether your final reply agrees with the draft; there is no semantic judge in this browser. Autonomous answers and abstentions do not count as fresh human agreement.</p>`, "ILLUSTRATIVE RELIABILITY HISTORY"));
$("#about").addEventListener("click", () => showDialog("A memory that learns from work.", "<p>Explore four connected moments: retrieve past resolutions, save your final decision, reuse it on the next ticket, and return authority to a human when circumstances change.</p><p><strong>Two independent forms of forgetting.</strong> Retrieval discounts records by insertion age. Reliability discounts earlier agreement observations. Neither measures correctness by itself.</p><p><strong>What is simulated?</strong> Customer data, semantic similarities, draft generation, review history, policy fixtures, and escalation. Ranking and reliability arithmetic run locally. No model, CRM, payment service, or backend is connected.</p><p>The autonomy gate is deliberately simplified. The research implementation includes SO/SC/DS transitions, minimum evidence, separate thresholds, and review policies. This concept is an explanation of the loop, not a reproduction of an experiment.</p><p>Everything stays in this tab. Reload or reset to start again.</p><button class='secondary wide' data-action='audit'>Open session activity ↗</button>", "A WORKING EXPLANATION"));

render();
