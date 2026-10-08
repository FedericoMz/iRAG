const tourSteps = [
  {
    eyebrow: "Start with the ticket",
    title: "A customer asks a familiar question.",
    description:
      "The wording looks routine, but the refund policy changed recently. Let’s see what iRAG remembers.",
    nextLabel: "Inspect memory",
    state: "Observing",
    stateClass: "",
    scenario: "refund",
    focus: "conversation",
    principles: [0],
  },
  {
    eyebrow: "A living knowledge base",
    title: "Two accepted resolutions are relevant.",
    description:
      "Unlike static document search, iRAG retrieves the outcomes of earlier human decisions.",
    nextLabel: "Compare evidence",
    state: "Observing",
    stateClass: "",
    scenario: "refund",
    focus: "assistant",
    principles: [1],
  },
  {
    eyebrow: "Recency changes the answer",
    title: "The closest match is not the safest precedent.",
    description:
      "Temporal weighting promotes the newer resolution and flags that the older answer may be stale.",
    nextLabel: "Review suggestion",
    state: "Assisting",
    stateClass: "assisting",
    scenario: "refund",
    focus: "assistant",
    principles: [2],
  },
  {
    eyebrow: "Human-in-the-loop",
    title: "iRAG drafts; the support agent decides.",
    description:
      "Use, edit, or dismiss the answer. The final human-approved resolution is what enters memory.",
    nextLabel: "Ask the harder question",
    state: "Assisting",
    stateClass: "assisting",
    scenario: "refund",
    focus: "assistant",
    principles: [3],
  },
  {
    eyebrow: "The reliability test",
    title: "An answer is only as good as its evidence.",
    description:
      "Now move beyond a familiar policy question and test what happens when organizational memory has no reliable precedent.",
    nextLabel: "Test uncertainty",
    state: "Observing",
    stateClass: "",
    scenario: "refund",
    focus: "assistant",
    principles: [4],
  },
  {
    eyebrow: "A deliberate non-answer",
    title: "When memory is insufficient, iRAG abstains.",
    description:
      "A novel integration request has no reliable precedent, so the system recommends escalation instead of guessing.",
    nextLabel: "See how trust grows",
    state: "Abstaining",
    stateClass: "abstaining",
    scenario: "integration",
    focus: "assistant",
    principles: [4],
  },
  {
    eyebrow: "Trust compounds",
    title: "Recent approvals become measurable confidence.",
    description:
      "Repeated human approval teaches iRAG where its evidence is stable enough to support greater autonomy.",
    nextLabel: "See guarded autonomy",
    state: "Learning",
    stateClass: "assisting",
    scenario: "summary",
    focus: "assistant",
    principles: [0, 1, 2, 3],
  },
  {
    eyebrow: "Progressive autonomy",
    title: "iRAG can resolve proven cases autonomously.",
    description:
      "Inside a validated, low-risk scope, iRAG answers directly while continuing to cite, monitor, and escalate.",
    nextLabel: "See it in product",
    state: "Autonomous",
    stateClass: "assisting",
    scenario: "summary",
    focus: "assistant",
    principles: [0, 1, 2, 3, 4],
  },
  {
    eyebrow: "Autonomy in product",
    title: "A proven case enters the queue.",
    description:
      "iRAG checks current evidence, policy stability, risk, and team-defined guardrails before taking action.",
    nextLabel: "Watch it resolve",
    state: "Evaluating",
    stateClass: "assisting",
    scenario: "autonomous",
    focus: "assistant",
    principles: [0, 2, 3],
  },
  {
    eyebrow: "Autonomous resolution",
    title: "iRAG answers, records, and keeps watching.",
    description:
      "The customer receives a response while the CRM retains the evidence trail, action record, and monitoring controls.",
    nextLabel: "See the outcome",
    state: "Autonomous",
    stateClass: "assisting",
    scenario: "autonomous",
    focus: "both",
    principles: [0, 1, 2, 3, 4],
  },
  {
    eyebrow: "The two engines",
    title: "Relevance in time. Autonomy through trust.",
    description:
      "iRAG combines temporal understanding with a progressive trust model that governs when the system can act.",
    nextLabel: "See the finale",
    state: "Autonomous",
    stateClass: "assisting",
    scenario: "summary",
    focus: "both",
    principles: [0, 1, 2, 3, 4],
  },
  {
    eyebrow: "The product thesis",
    title: "Resolved work becomes governed organizational memory.",
    description:
      "Time-aware retrieval, attributable evidence, and progressive trust form a feedback loop around the support team.",
    nextLabel: "Restart demo",
    state: "Assisting",
    stateClass: "assisting",
    scenario: "summary",
    focus: "assistant",
    principles: [0, 1, 2, 3, 4],
  },
];

const scenarios = {
  refund: {
    ticketId: "Ticket #5943",
    ticketStatus: "Open",
    customerAvatar: "AL",
    customerName: "Alex Lewis",
    customerEmail: "alex.lewis@example.com",
    customerPlan: "Professional",
    customerSince: "March 2022",
    previousTickets: "4",
    ticketTopic: "Billing & refunds",
    messageAvatar: "AL",
    messageSender: "Alex Lewis",
    messageTime: "10:42 AM",
    ticketSubject: "Refund request after 20 days",
    ticketMessage:
      "Hi, I purchased an annual subscription 20 days ago, but it isn't the right fit for our team. Can I still receive a full refund?",
    ticketSignoff: "Thanks,<br>Alex",
  },
  integration: {
    ticketId: "Ticket #6018",
    ticketStatus: "New",
    customerAvatar: "NP",
    customerName: "Noor Patel",
    customerEmail: "noor.patel@example.com",
    customerPlan: "Enterprise",
    customerSince: "November 2020",
    previousTickets: "12",
    ticketTopic: "Integrations",
    messageAvatar: "NP",
    messageSender: "Noor Patel",
    messageTime: "11:06 AM",
    ticketSubject: "Restore a deleted custom integration",
    ticketMessage:
      "Our custom warehouse integration was deleted three years ago. Can your team restore its original configuration and historical credentials?",
    ticketSignoff: "Best,<br>Noor",
  },
  autonomous: {
    ticketId: "Ticket #6127",
    ticketStatus: "New",
    customerAvatar: "JK",
    customerName: "Jordan Kim",
    customerEmail: "jordan.kim@example.com",
    customerPlan: "Professional",
    customerSince: "January 2024",
    previousTickets: "1",
    ticketTopic: "Billing & refunds",
    messageAvatar: "JK",
    messageSender: "Jordan Kim",
    messageTime: "11:23 AM",
    ticketSubject: "Refund request after 7 days",
    ticketMessage:
      "Hi, I purchased an annual subscription 7 days ago and would like to cancel. Can you issue a full refund?",
    ticketSignoff: "Thanks,<br>Jordan",
  },
};

let currentStep = 0;
let currentScenario = "refund";
let currentIntroPage = 0;
let decisionMade = false;

const screens = [...document.querySelectorAll(".panel-screen")];
const principles = [...document.querySelectorAll("#principleList li")];
const introOverlay = document.getElementById("introOverlay");
const introSlides = [...document.querySelectorAll(".intro-slide")];
const introProgress = document.querySelector(".intro-progress");
const introProgressDots = [...document.querySelectorAll(".intro-progress > span")];
const introProgressLabel = document.getElementById("introProgressLabel");
const introBack = document.getElementById("introBack");
const introNext = document.getElementById("introNext");
const introSkip = document.getElementById("introSkip");
const stateBadge = document.getElementById("stateBadge");
const tourStepLabel = document.getElementById("tourStepLabel");
const tourProgressBar = document.getElementById("tourProgressBar");
const tourEyebrow = document.getElementById("tourEyebrow");
const tourTitle = document.getElementById("tourTitle");
const tourDescription = document.getElementById("tourDescription");
const tourBack = document.getElementById("tourBack");
const tourNext = document.getElementById("tourNext");
const workspaceContent = document.querySelector(".workspace-content");
const questionInterstitial = document.getElementById("questionInterstitial");
const interstitialBack = document.getElementById("interstitialBack");
const interstitialNext = document.getElementById("interstitialNext");
const autonomyInterstitial = document.getElementById("autonomyInterstitial");
const autonomySlides = [...document.querySelectorAll(".autonomy-slide")];
const autonomyBack = document.getElementById("autonomyBack");
const autonomyNext = document.getElementById("autonomyNext");
const dualityInterstitial = document.getElementById("dualityInterstitial");
const dualityBack = document.getElementById("dualityBack");
const dualityNext = document.getElementById("dualityNext");
const outroInterstitial = document.getElementById("outroInterstitial");
const outroBack = document.getElementById("outroBack");
const outroRestart = document.getElementById("outroRestart");
const composerEditor = document.getElementById("composerEditor");
const submitReply = document.getElementById("submitReply");
const replyComposer = document.getElementById("replyComposer");
const autonomousReply = document.getElementById("autonomousReply");
const decisionActions = document.getElementById("decisionActions");
const decisionConfirmation = document.getElementById("decisionConfirmation");
const overviewDialog = document.getElementById("overviewDialog");

function setText(id, value, html = false) {
  const element = document.getElementById(id);
  if (!element) return;
  if (html) {
    element.innerHTML = value;
  } else {
    element.textContent = value;
  }
}

function setScenario(name) {
  if (name === "summary" || name === currentScenario) return;
  currentScenario = name;
  const scenario = scenarios[name];

  Object.entries(scenario).forEach(([id, value]) => {
    setText(id, value, id === "ticketSignoff");
  });

  composerEditor.textContent = "";
  composerEditor.setAttribute("contenteditable", "true");
  submitReply.disabled = true;
  submitReply.textContent = "Submit as solved";
  replyComposer.hidden = false;
  autonomousReply.hidden = true;
  decisionMade = false;
  decisionActions.hidden = false;
  decisionConfirmation.classList.remove("visible");
}

function setWorkspaceFocus(target) {
  workspaceContent.classList.remove(
    "focus-conversation",
    "focus-assistant",
    "focus-both",
  );
  if (target) workspaceContent.classList.add(`focus-${target}`);
}

function updateIntro(pageIndex) {
  currentIntroPage = Math.max(0, Math.min(introSlides.length - 1, pageIndex));

  introSlides.forEach((slide, index) => {
    const isActive = index === currentIntroPage;
    slide.classList.toggle("active", isActive);
    slide.setAttribute("aria-hidden", String(!isActive));
  });

  introProgressDots.forEach((dot, index) => {
    dot.classList.toggle("active", index === currentIntroPage);
  });

  introProgressLabel.textContent =
    `0${currentIntroPage + 1} / 0${introSlides.length}`;
  introProgress.setAttribute(
    "aria-label",
    `Intro screen ${currentIntroPage + 1} of ${introSlides.length}`,
  );
  introOverlay.setAttribute(
    "aria-labelledby",
    currentIntroPage === 0 ? "introTitle" : "introArchitectureTitle",
  );
  introBack.disabled = currentIntroPage === 0;
  introNext.firstChild.textContent =
    currentIntroPage === 0 ? "See how it fits " : "Enter the demo ";

  if (introOverlay.classList.contains("active")) {
    requestAnimationFrame(() => introNext.focus({ preventScroll: true }));
  }
}

function openIntro() {
  introOverlay.classList.add("active");
  introOverlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("intro-active");
  updateIntro(0);
}

function closeIntro() {
  introOverlay.classList.remove("active");
  introOverlay.setAttribute("aria-hidden", "true");
  document.body.classList.remove("intro-active");
  requestAnimationFrame(() => tourNext.focus({ preventScroll: true }));
}

function updateTour(stepIndex, options = {}) {
  const boundedStep = Math.max(0, Math.min(tourSteps.length - 1, stepIndex));
  currentStep = boundedStep;
  const step = tourSteps[currentStep];

  if (step.scenario !== "summary") {
    setScenario(step.scenario);
  }

  screens.forEach((screen, index) => {
    screen.classList.toggle("active", index === currentStep);
  });

  principles.forEach((principle, index) => {
    principle.classList.toggle("active", step.principles.includes(index));
  });

  stateBadge.className = `state-badge ${step.stateClass}`.trim();
  stateBadge.innerHTML = `<span></span>${step.state}`;

  tourStepLabel.textContent = `Step ${currentStep + 1} of ${tourSteps.length}`;
  tourProgressBar.style.width = `${((currentStep + 1) / tourSteps.length) * 100}%`;
  tourEyebrow.textContent = step.eyebrow;
  tourTitle.textContent = step.title;
  tourDescription.textContent = step.description;
  tourNext.firstChild.textContent = `${step.nextLabel} `;
  tourBack.disabled = currentStep === 0;
  setWorkspaceFocus(step.focus);

  const interstitialIsActive = currentStep === 4;
  const autonomyIsActive = currentStep === 6 || currentStep === 7;
  const autonomyPage = currentStep - 6;
  const dualityIsActive = currentStep === 10;
  const outroIsActive = currentStep === 11;
  questionInterstitial.classList.toggle("active", interstitialIsActive);
  autonomyInterstitial.classList.toggle("active", autonomyIsActive);
  dualityInterstitial.classList.toggle("active", dualityIsActive);
  outroInterstitial.classList.toggle("active", outroIsActive);
  questionInterstitial.setAttribute(
    "aria-hidden",
    String(!interstitialIsActive),
  );
  autonomyInterstitial.setAttribute("aria-hidden", String(!autonomyIsActive));
  dualityInterstitial.setAttribute("aria-hidden", String(!dualityIsActive));
  outroInterstitial.setAttribute("aria-hidden", String(!outroIsActive));

  autonomySlides.forEach((slide, index) => {
    const isActive = autonomyIsActive && index === autonomyPage;
    slide.classList.toggle("active", isActive);
    slide.setAttribute("aria-hidden", String(!isActive));
  });

  if (autonomyIsActive) {
    autonomyInterstitial.setAttribute(
      "aria-labelledby",
      autonomyPage === 0 ? "trustTitle" : "autonomyTitle",
    );
    autonomyNext.firstChild.textContent =
      autonomyPage === 0 ? "See guarded autonomy " : "See the outcome ";
  }

  document.body.classList.toggle(
    "interstitial-active",
    interstitialIsActive ||
      autonomyIsActive ||
      dualityIsActive ||
      outroIsActive,
  );

  if (interstitialIsActive) {
    requestAnimationFrame(() => interstitialNext.focus({ preventScroll: true }));
  } else if (autonomyIsActive) {
    requestAnimationFrame(() => autonomyNext.focus({ preventScroll: true }));
  } else if (dualityIsActive) {
    requestAnimationFrame(() => dualityNext.focus({ preventScroll: true }));
  } else if (outroIsActive) {
    requestAnimationFrame(() => outroRestart.focus({ preventScroll: true }));
  } else if (
    questionInterstitial.contains(document.activeElement) ||
    autonomyInterstitial.contains(document.activeElement) ||
    dualityInterstitial.contains(document.activeElement) ||
    outroInterstitial.contains(document.activeElement)
  ) {
    requestAnimationFrame(() => tourNext.focus({ preventScroll: true }));
  }

  if (currentStep !== 3) {
    composerEditor.blur();
  }

  if (currentStep === 8) {
    autonomousReply.hidden = true;
    replyComposer.hidden = false;
    setText("ticketStatus", "Evaluating");
  } else if (currentStep === 9) {
    autonomousReply.hidden = false;
    replyComposer.hidden = true;
    setText("ticketStatus", "Solved automatically");
  } else if (currentStep === 10 || currentStep === 11) {
    setText("ticketStatus", "Solved automatically");
  } else {
    autonomousReply.hidden = true;
    replyComposer.hidden = false;
  }

  if (options.focusPanel) {
    document.querySelector(".irag-panel").scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }
}

function addDraftToComposer(edit = false) {
  const draft = [
    "Hi Alex,",
    "",
    "Thanks for reaching out. Under our current policy, full refunds are available within 14 days of purchase. Because your purchase was made 20 days ago, it is outside the refund window.",
    "",
    "I can still help you review alternative plan options.",
  ].join("\n");

  composerEditor.textContent = draft;
  submitReply.disabled = false;
  decisionMade = true;
  decisionActions.hidden = true;
  decisionConfirmation.classList.add("visible");
  setWorkspaceFocus("conversation");

  if (edit) {
    composerEditor.focus();
    const range = document.createRange();
    range.selectNodeContents(composerEditor);
    range.collapse(false);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }
}

tourNext.addEventListener("click", () => {
  if (currentStep === tourSteps.length - 1) {
    restart();
    return;
  }
  updateTour(currentStep + 1);
});

tourBack.addEventListener("click", () => {
  updateTour(currentStep - 1);
});

interstitialBack.addEventListener("click", () => {
  updateTour(3);
});

interstitialNext.addEventListener("click", () => {
  updateTour(5);
});

autonomyBack.addEventListener("click", () => {
  updateTour(currentStep - 1);
});

autonomyNext.addEventListener("click", () => {
  updateTour(currentStep + 1);
});

dualityBack.addEventListener("click", () => {
  updateTour(9);
});

dualityNext.addEventListener("click", () => {
  updateTour(11);
});

outroBack.addEventListener("click", () => {
  updateTour(10);
});

outroRestart.addEventListener("click", restart);

introBack.addEventListener("click", () => {
  updateIntro(currentIntroPage - 1);
});

introNext.addEventListener("click", () => {
  if (currentIntroPage === introSlides.length - 1) {
    closeIntro();
  } else {
    updateIntro(currentIntroPage + 1);
  }
});

introSkip.addEventListener("click", closeIntro);

document.getElementById("decisionActions").addEventListener("click", (event) => {
  const button = event.target.closest("[data-decision]");
  if (!button) return;

  if (button.dataset.decision === "accept") {
    addDraftToComposer(false);
  } else if (button.dataset.decision === "edit") {
    addDraftToComposer(true);
  } else {
    decisionMade = true;
    decisionActions.hidden = true;
    decisionConfirmation.classList.add("visible");
    decisionConfirmation.querySelector("strong").textContent = "Suggestion dismissed";
    decisionConfirmation.querySelector("span").textContent =
      "Your feedback is recorded; no reply was changed.";
  }
});

composerEditor.addEventListener("input", () => {
  submitReply.disabled = composerEditor.textContent.trim().length === 0;
});

submitReply.addEventListener("click", () => {
  submitReply.textContent = "Solved ✓";
  submitReply.disabled = true;
  setText("ticketStatus", "Solved");
  setTimeout(() => {
    if (currentStep === 3) updateTour(4);
  }, 650);
});

function restart() {
  currentScenario = "";
  setScenario("refund");
  composerEditor.textContent = "";
  submitReply.textContent = "Submit as solved";
  submitReply.disabled = true;
  decisionMade = false;
  decisionActions.hidden = false;
  decisionConfirmation.classList.remove("visible");
  decisionConfirmation.querySelector("strong").textContent =
    "Draft added to the reply";
  decisionConfirmation.querySelector("span").textContent =
    "The final response remains yours.";
  updateTour(0);
  openIntro();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.getElementById("restartDemo").addEventListener("click", restart);

document.getElementById("openOverview").addEventListener("click", () => {
  if (typeof overviewDialog.showModal === "function") {
    overviewDialog.showModal();
  } else {
    overviewDialog.setAttribute("open", "");
  }
});

overviewDialog.addEventListener("click", (event) => {
  const rect = overviewDialog.getBoundingClientRect();
  const clickedBackdrop =
    event.clientX < rect.left ||
    event.clientX > rect.right ||
    event.clientY < rect.top ||
    event.clientY > rect.bottom;
  if (clickedBackdrop) overviewDialog.close();
});

document.addEventListener("keydown", (event) => {
  if (introOverlay.classList.contains("active")) {
    if (event.key === "ArrowRight") {
      introNext.click();
    } else if (event.key === "ArrowLeft" && currentIntroPage > 0) {
      introBack.click();
    }
    return;
  }

  if (overviewDialog.open) return;
  if (event.key === "ArrowRight") {
    tourNext.click();
  } else if (event.key === "ArrowLeft" && currentStep > 0) {
    tourBack.click();
  }
});

updateTour(0);
openIntro();
