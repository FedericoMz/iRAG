# iRAG — Memory in motion

A self-contained alternative to the parent demo. Open `index.html` directly in a
modern browser, or serve this directory from the repository root:

```sh
python3 -m http.server 4175 --directory demo/interactive
```

Visit http://localhost:4175. No build, dependencies, credentials, or backend are
required. Assets are local. Session state lives only in the current tab and is
discarded on reload or reset.

## The experience

1. Toggle temporal weighting to compare semantic-only and decayed retrieval.
   Adjust the half-life, inspect source records, and edit the suggested reply.
2. Resolve the ticket to insert the exact final reply into memory. The next
   customer retrieves it. Confirm whether each final reply agrees with the
   suggestion; reported corrections reduce recent agreement.
3. Inspect the autonomy gate, simulate a reply, then introduce a human-reported
   policy exception. Authority returns to Assist while earlier decisions stay
   in the audit trail.
4. Escalate an unsupported integration request with context. Abstention does
   not create a resolved memory record or a reliability observation.

Every chapter can be opened directly. The memory chapter explains its prerequisite
when no answer has been saved. The autonomy chapter offers an explicitly labeled
replay of 12 illustrated agreeing reviews if the review gate has not been met.

## Simulation contract

Customer records, semantic similarities, drafts, policy fixtures, review history,
and escalation are synthetic. No CRM messages or payments are sent. User edits
are stored verbatim; the browser does not assess their correctness. Saved human
replies use a fixed illustrative similarity of 0.98 for the next billing ticket.

Retrieval computes `similarity * 0.5 ** (insertion_age / half_life)`. The two
seeded records start at insertion ages 700 and 14; every new resolution ages
existing records by one insertion. Calendar age is not used. Ranking is evidence
priority, not a correctness score or an automatic policy-change detector.

Reliability uses separate decay of 0.9. It starts at weighted numerator 8.6,
denominator 10, and 18 historical observations. Each human-reported agreement or
correction updates that history. The simplified autonomy gate requires FEA above
0.8, at least 20 observations, and enabled temporal weighting for the explicit
14-day policy fixture. The last condition is a scenario guardrail, not a claim
that enabling temporal weighting validates arbitrary policy evidence.

The intervention explicitly returns control to a human without manufacturing an
accuracy observation. Model-finalized answers enter memory with model provenance;
they do not improve FEA by endorsing themselves. These settings and the simplified
gate are deliberate explanatory choices, not the research runner's full SO/SC/DS
controller or paper defaults.

## Files and checks

- `index.html`: workspace, navigation, and native detail dialog.
- `styles.css`: local design system, responsive panels, reduced-motion support.
- `app.js`: scenario fixtures, deterministic calculations, state, and rendering.
- `tests/browser.html`: browser checks covering ranking, the memory loop,
  editing, correction feedback, the authority gate, intervention, escalation,
  safe text rendering, and reset.

With the local server running, open http://localhost:4175/tests/browser.html.
The page reports each check and a final pass/fail result. It can also be run in
headless Chrome with `--dump-dom --virtual-time-budget=5000`; inspect the rendered
body’s `data-result` attribute for `passed` or `failed`.

The suite also checks overflow at 320, 390, 600, 768, 1024, and 1440 pixels and
keyboard navigation of mobile panels. For an exact-width visual preview, open
`tests/browser.html?preview=390` (or another width between 320 and 1920). This
renders the demo in an iframe without running the tests.

Desktop shows ticket and evidence together. Narrow viewports use keyboard-accessible
panel tabs. Native dialogs provide focus containment and Escape dismissal. There
are no global navigation shortcuts that interfere with editing replies.

The repository currently ignores `demo/`; this variant follows that existing
policy. No generated files or runtime outputs belong in this folder.
