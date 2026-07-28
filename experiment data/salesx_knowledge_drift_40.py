"""Authored change catalogue for the 40% SalesX drift benchmark.

The base article catalogue is shared with the 10% benchmark. This module
defines a separate release history: eight articles per category change in each
of Q2, Q3, and Q4. Every additional change is article-specific.
"""

from salesx_knowledge import (
    CHANGE_DIMENSIONS,
    DRIFT_KEYS,
    RULE_CHANGES,
)


DRIFT_KEYS_40 = {
    "Q2": {
        "billing": [
            "seat_billing",
            "refunds",
            "plans",
            "trial_conversion",
            "proration",
            "payment_methods",
            "failed_payments",
            "tax",
        ],
        "integrations": [
            "webhook_security",
            "conflict_resolution",
            "api_limits",
            "rate_handling",
            "bulk_api",
            "webhook_config",
            "webhook_retries",
            "connected_apps",
        ],
        "permissions": [
            "permission_sets",
            "explicit_denies",
            "role_model",
            "owner_admin",
            "custom_roles",
            "record_access",
            "field_access",
            "hierarchy",
        ],
        "reporting": [
            "exports",
            "subscriptions",
            "report_types",
            "builder_access",
            "filters",
            "formulas",
            "joined_reports",
            "dashboards",
        ],
        "onboarding": [
            "imports",
            "deduplication",
            "workspace_creation",
            "trial",
            "implementation_roles",
            "discovery",
            "invitations",
            "csv_preparation",
        ],
    },
    "Q3": {
        "billing": [
            "overages",
            "downgrades",
            "currency",
            "purchase_orders",
            "credits",
            "cancellation",
            "amendments",
            "billing_access",
        ],
        "integrations": [
            "oauth",
            "service_accounts",
            "sandboxes",
            "connector_sync",
            "field_mapping",
            "idempotency",
            "network",
            "logs",
        ],
        "permissions": [
            "scim",
            "delegated_admin",
            "teams",
            "mfa",
            "sso",
            "sessions",
            "approvals",
            "audit",
        ],
        "reporting": [
            "row_visibility",
            "currency",
            "recipients",
            "large_data",
            "snapshots",
            "history",
            "fiscal_calendar",
            "timezone",
        ],
        "onboarding": [
            "identity_setup",
            "domain_verification",
            "import_order",
            "external_ids",
            "validation_rules",
            "sandbox_rehearsal",
            "guided_sessions",
            "academy",
        ],
    },
    "Q4": {
        "billing": [
            "invoices",
            "renewals",
            "expiry_data",
            "disputes",
            "plans",
            "trial_conversion",
            "proration",
            "payment_methods",
        ],
        "integrations": [
            "api_versions",
            "event_streaming",
            "secret_rotation",
            "deprecation",
            "api_limits",
            "rate_handling",
            "bulk_api",
            "webhook_config",
        ],
        "permissions": [
            "guest_access",
            "access_reviews",
            "break_glass",
            "ownership_transfer",
            "role_model",
            "owner_admin",
            "custom_roles",
            "record_access",
        ],
        "reporting": [
            "dashboard_refresh",
            "schedules",
            "trash",
            "troubleshooting",
            "report_types",
            "builder_access",
            "filters",
            "formulas",
        ],
        "onboarding": [
            "assisted_migration",
            "go_live",
            "sample_data",
            "rollback",
            "workspace_creation",
            "trial",
            "implementation_roles",
            "discovery",
        ],
    },
}


ADDITIONAL_RULE_CHANGES = {
    "billing.plans": {
        "Q2": (
            "Feature access now resolves from a versioned entitlement manifest "
            "combining the workspace plan and active add-ons; Administrators may "
            "assign only permissions exposed by that manifest, and commercial "
            "entitlement still does not itself grant user access."
        ),
        "Q4": (
            "Feature access now resolves from an effective-dated entitlement "
            "manifest, and a plan or add-on change first produces an impact preview "
            "of features that will activate, become read-only, or require migration; "
            "user permissions remain a separate decision."
        ),
    },
    "billing.trial_conversion": {
        "Q2": (
            "Trial conversion now creates a pre-conversion inventory: customer "
            "records and approved configuration are retained, sample data is "
            "removed unless explicitly preserved, and paid usage begins only after "
            "the inventory and order activation both succeed."
        ),
        "Q4": (
            "Trial conversion now uses a signed conversion manifest that classifies "
            "retained configuration, customer records, sample data, and unsupported "
            "trial features; activation is atomic and rolls back to the trial if "
            "the paid entitlement check fails."
        ),
    },
    "billing.proration": {
        "Q2": (
            "Mid-term additions still co-terminate, but proration now starts at the "
            "later of requested activation and successful provisioning; a failed "
            "provisioning attempt cannot create a billable service interval."
        ),
        "Q4": (
            "Mid-term additions now use milestone proration: entitlement reservation "
            "is non-billable, successful provisioning starts the charge, and any "
            "approved staged rollout is priced from each stage's activation while "
            "all stages retain the existing term end."
        ),
    },
    "billing.payment_methods": {
        "Q2": (
            "Only a Billing Manager or Workspace Owner may register a payment "
            "method, and activation now requires step-up authentication plus "
            "ownership verification; an unverified method may be saved but cannot "
            "become the collection default."
        ),
        "Q4": (
            "Payment methods now have separate registrar and approver roles: a "
            "verified Billing Manager registers the instrument, another authorized "
            "billing actor approves default use, and emergency replacement requires "
            "a time-limited exception with automatic review."
        ),
    },
    "billing.failed_payments": {
        "Q2": (
            "Failed-payment recovery now classifies failures as retryable, "
            "authentication-required, or terminal; SalesX retries only retryable "
            "failures, routes authentication cases to verified billing contacts, "
            "and restricts service only after the applicable recovery path ends."
        ),
    },
    "billing.tax": {
        "Q2": (
            "Tax determination now snapshots the legal buyer, sold-to location, "
            "supply classification, and validated exemption evidence when an "
            "invoice is finalized; later profile edits apply prospectively and do "
            "not silently recalculate the closed document."
        ),
    },
    "billing.currency": {
        "Q3": (
            "Subscription currency remains fixed for an active term, but a renewal "
            "may adopt a supported new currency through a quoted currency migration "
            "that settles open documents and records the conversion basis before "
            "the next term activates."
        ),
    },
    "billing.purchase_orders": {
        "Q3": (
            "Purchase orders now pass a lifecycle check for buyer identity, amount, "
            "currency, service dates, and remaining authorization before invoice "
            "association; exhaustion or expiry places the invoice on billing review "
            "but does not alter the accepted SalesX order form."
        ),
    },
    "billing.credits": {
        "Q3": (
            "Service credits now enter a governed credit wallet with an eligible "
            "product scope and expiry; allocation follows oldest-expiring credit "
            "first, and a Billing Manager may preview but not redirect credits to "
            "another billing account or convert them to cash."
        ),
    },
    "billing.cancellation": {
        "Q3": (
            "Cancellation now records separate non-renewal, service-end, and data-"
            "handling instructions; access continues through the committed service "
            "end unless a documented early-termination right is approved, and the "
            "data instruction does not shorten mandatory retention."
        ),
    },
    "billing.amendments": {
        "Q3": (
            "Every mid-term commercial change now uses an effective-dated amendment "
            "preview showing entitlement, billing, and dependency impact; additions "
            "may co-terminate after acceptance, while reductions remain blocked "
            "until the amendment supplies an authorized effective date."
        ),
    },
    "billing.billing_access": {
        "Q3": (
            "Billing access now uses purpose-specific capabilities for documents, "
            "payment instruments, tax evidence, and commercial changes; Product "
            "Administrator status grants none of them, and sensitive combinations "
            "require separation-of-duties approval."
        ),
    },
    "billing.expiry_data": {
        "Q4": (
            "At service end the workspace enters a staged closure: ordinary access "
            "stops, authorized export remains available for a documented retrieval "
            "window, legal holds freeze eligible records, and deletion begins only "
            "after both the retention and hold checks clear."
        ),
    },
    "billing.disputes": {
        "Q4": (
            "Invoice disputes now separate disputed and undisputed ledger portions; "
            "the customer submits line-level reason and evidence, undisputed amounts "
            "continue through collection, and resolution creates a linked decision "
            "record and any required accounting document."
        ),
    },
    "integrations.api_limits": {
        "Q2": (
            "API capacity now uses endpoint-weighted workspace budgets with reserved "
            "capacity for approved critical integrations; retries consume budget, "
            "and borrowing unused reserved capacity requires an explicit burst "
            "policy rather than occurring automatically."
        ),
        "Q4": (
            "API capacity now uses adaptive endpoint-weighted budgets: critical "
            "reservations remain protected, each client receives a published cost "
            "forecast, and temporary burst leases expire automatically without "
            "raising the workspace's contracted sustained capacity."
        ),
    },
    "integrations.rate_handling": {
        "Q2": (
            "Rate-limited clients must now coordinate retries through a shared "
            "integration backoff state, honor Retry-After, add jitter, and cap both "
            "attempts and concurrency; independent workers may not reset the delay "
            "or create a synchronized retry wave."
        ),
        "Q4": (
            "Rate handling now uses server-issued retry epochs and client fairness "
            "tokens: workers coordinate one bounded queue per integration, preserve "
            "operation deadlines, and dead-letter work that cannot safely complete "
            "within its retry budget."
        ),
    },
    "integrations.bulk_api": {
        "Q2": (
            "Bulk jobs now have explicit Validate and Execute phases: validation "
            "freezes mappings and estimates row outcomes, while execution processes "
            "only the approved manifest and returns row-level results for "
            "reconciliation."
        ),
        "Q4": (
            "Bulk jobs now use partitioned execution manifests with independent "
            "checkpoints; failed partitions may resume from their last committed "
            "boundary, but changing mappings or source content creates a new job "
            "rather than mutating the approved run."
        ),
    },
    "integrations.webhook_config": {
        "Q2": (
            "Webhook subscriptions now bind event types, endpoint, owner, signing "
            "configuration, and data classification in one versioned subscription; "
            "changing a security-sensitive field creates a pending version that "
            "must pass a verification delivery before activation."
        ),
        "Q4": (
            "Webhook subscriptions now support atomic version promotion: the pending "
            "endpoint and signing policy must pass challenge, classification, and "
            "replay tests, after which new events switch together while the prior "
            "version remains available only for its bounded drain window."
        ),
    },
    "integrations.webhook_retries": {
        "Q2": (
            "Webhook retries now distinguish transport failure, receiver rejection, "
            "and invalid acknowledgement; only retryable classes use backoff, the "
            "delivery ID remains stable, and exhausted deliveries enter a replay-"
            "controlled dead-letter queue."
        ),
    },
    "integrations.connected_apps": {
        "Q2": (
            "Connected apps now move through Draft, Approved, Active, and Suspended "
            "states; scopes, assignments, network policy, and accountable owners are "
            "reviewed at approval, and changing a sensitive scope returns an active "
            "app to approval before the new scope can be used."
        ),
    },
    "integrations.sandboxes": {
        "Q3": (
            "Integration sandboxes now use explicit environment bindings: copied "
            "configuration is scrubbed of credentials and production endpoints, "
            "test identities cannot cross environments, and promotion recreates "
            "secrets from approved production references."
        ),
    },
    "integrations.connector_sync": {
        "Q3": (
            "Standard connectors now expose separate extraction, application, and "
            "reconciliation watermarks; a cycle is successful only when all three "
            "advance consistently, and a stalled stage pauses later cycles instead "
            "of reporting the last partial transfer as current."
        ),
    },
    "integrations.field_mapping": {
        "Q3": (
            "Connector mappings now use versioned field contracts that declare "
            "direction, type, null semantics, authority, and failure handling; an "
            "incompatible contract change requires sample validation and cannot be "
            "silently applied to queued records."
        ),
    },
    "integrations.idempotency": {
        "Q3": (
            "Idempotency keys now bind the caller, operation family, canonical "
            "payload digest, and retention window; replaying the same key and digest "
            "returns the recorded outcome, while a different digest is rejected "
            "rather than treated as a new request."
        ),
    },
    "integrations.network": {
        "Q3": (
            "Network controls now use signed, region-and-service-specific range "
            "feeds with activation windows; customers must stage additions before "
            "activation and remove retired ranges afterward, while identity and "
            "request authentication remain mandatory."
        ),
    },
    "integrations.logs": {
        "Q3": (
            "Integration diagnostics now use tiered disclosure: operators see "
            "correlation, timing, status, and redacted field paths, while access to "
            "approved payload fragments is case-bound, time-limited, and audited; "
            "stored secrets are never disclosed."
        ),
    },
    "integrations.secret_rotation": {
        "Q4": (
            "Secret rotation now uses declared Active, Next, and Retiring slots; "
            "traffic must prove use of Next before promotion, Retiring remains valid "
            "only for the bounded drain interval, and rollback cannot reactivate a "
            "credential already marked compromised."
        ),
    },
    "integrations.deprecation": {
        "Q4": (
            "Integration deprecation now has separate creation-close, support-end, "
            "and runtime-retirement milestones; each affected owner receives a "
            "compatibility report, and only an approved time-boxed bridge may operate "
            "between support end and final retirement."
        ),
    },
    "permissions.role_model": {
        "Q2": (
            "Effective access now evaluates base role, approved grants, sharing, "
            "field restrictions, and explicit denials as a traceable policy graph; "
            "the resulting explanation identifies every contributing path, and a "
            "denial remains authoritative unless its own policy permits an exception."
        ),
        "Q4": (
            "Effective access now uses a versioned policy graph with decision-time "
            "evidence; identity, classification, ownership, and denial changes "
            "invalidate cached decisions immediately, and every privileged outcome "
            "records the exact graph version that authorized it."
        ),
    },
    "permissions.owner_admin": {
        "Q2": (
            "Workspace Owner authority now separates constitutional actions such as "
            "ownership transfer and recovery governance from delegated product "
            "administration; Administrators receive named capabilities, and no "
            "delegation can reproduce the Owner's reserved authority."
        ),
        "Q4": (
            "Reserved Owner actions now require a verified succession and recovery "
            "policy, while delegated administration uses expiring capability "
            "charters with named sponsors; loss of the sponsor suspends the charter "
            "without transferring constitutional Owner authority."
        ),
    },
    "permissions.custom_roles": {
        "Q2": (
            "Custom roles now derive from a versioned job-function template and "
            "declare prohibited capability combinations; publishing a role requires "
            "an impact preview, and existing assignments retain the prior version "
            "until explicitly migrated."
        ),
        "Q4": (
            "Custom roles now use semantic versioning and assignment cohorts: safe "
            "reductions may migrate automatically, privilege increases require "
            "fresh approval, and rollback restores the prior role version without "
            "reviving grants that independently expired."
        ),
    },
    "permissions.record_access": {
        "Q2": (
            "Record access now resolves ownership, hierarchy, teams, rules, and "
            "explicit shares into purpose-labelled access paths; object and field "
            "controls constrain every path, and removing the last valid purpose "
            "revokes the record grant."
        ),
        "Q4": (
            "Record access paths now carry source, purpose, sensitivity ceiling, and "
            "expiry; classification or ownership changes trigger reevaluation, and "
            "a path that exceeds its ceiling is quarantined for review rather than "
            "continuing with stale visibility."
        ),
    },
    "permissions.field_access": {
        "Q2": (
            "Field security now distinguishes discover, view, export, and edit "
            "capabilities; record access supplies none of them implicitly, and "
            "masked display values cannot be used to infer or update the protected "
            "underlying field."
        ),
    },
    "permissions.hierarchy": {
        "Q2": (
            "Hierarchy access now requires an object-specific upward-visibility "
            "policy and stops at declared boundary roles; it cannot cross field "
            "restrictions, explicit denials, or a confidential-record boundary."
        ),
    },
    "permissions.teams": {
        "Q3": (
            "Record teams now use role-labelled, expiring memberships with a team "
            "sponsor; access is the intersection of the team role and each member's "
            "object and field permissions, and sponsor loss opens a recertification "
            "instead of transferring ownership."
        ),
    },
    "permissions.mfa": {
        "Q3": (
            "Authentication policy now assigns phishing-resistant MFA to privileged "
            "and high-risk sessions, permits only approved factors for ordinary "
            "users, and treats recovery as a temporary restricted session until a "
            "strong factor is re-enrolled."
        ),
    },
    "permissions.sso": {
        "Q3": (
            "Enforced SSO now uses routing policies by verified domain and user "
            "population; unmatched identities are blocked from password fallback, "
            "while controlled recovery accounts use a separately monitored path "
            "that cannot be assigned for routine work."
        ),
    },
    "permissions.sessions": {
        "Q3": (
            "Session policy now evaluates idle age, absolute age, device trust, "
            "authentication strength, and action risk; a high-risk action can force "
            "step-up or terminate only the affected session without changing the "
            "workspace-wide lifetime."
        ),
    },
    "permissions.approvals": {
        "Q3": (
            "Approval chains now bind the policy version, requester, resource, and "
            "risk at submission; later policy changes re-evaluate pending requests, "
            "and delegation cannot satisfy a step when it would violate separation "
            "of duties."
        ),
    },
    "permissions.audit": {
        "Q3": (
            "Governed audit events now form an integrity-linked sequence containing "
            "actor, action, target, source, policy decision, and before/after values; "
            "late enrichment appends a correction event rather than rewriting the "
            "original evidence."
        ),
    },
    "permissions.break_glass": {
        "Q4": (
            "Break-glass access now requires a declared incident, two-person release "
            "for destructive capabilities, a fixed privilege bundle, and automatic "
            "session expiry; every use opens a review, and extending access requires "
            "a new release rather than editing the active grant."
        ),
    },
    "permissions.ownership_transfer": {
        "Q4": (
            "Ownership transfer now uses a staged handover: the current Owner "
            "approves, the successor verifies recovery and accepts, and reserved "
            "authority moves atomically after a cooling-off check; emergency "
            "recovery follows a separate audited process."
        ),
    },
    "reporting.report_types": {
        "Q2": (
            "Report types now publish a versioned semantic contract covering primary "
            "grain, relationship cardinality, available fields, and null behavior; "
            "a breaking contract change creates a new version instead of silently "
            "altering saved reports."
        ),
        "Q4": (
            "Report-type contracts now include lineage and compatibility guarantees; "
            "saved reports pin a major version, may adopt backward-compatible minor "
            "fields automatically, and require an impact-reviewed migration for "
            "grain or relationship changes."
        ),
    },
    "reporting.builder_access": {
        "Q2": (
            "Report authoring now separates Draft, Validate, and Publish privileges; "
            "authors may use only fields visible to them, validation runs under the "
            "declared audience context, and publication requires folder authority "
            "independent of creation access."
        ),
        "Q4": (
            "Report publication now uses a release workflow: the definition is "
            "content-addressed, audience security is simulated against representative "
            "identities, and promotion atomically replaces the published version "
            "while preserving the prior version for rollback."
        ),
    },
    "reporting.filters": {
        "Q2": (
            "Report filters now store a typed expression tree with explicit timezone, "
            "locale, and null policy; invalid coercions fail validation, and relative "
            "dates are evaluated in the saved context rather than each viewer's "
            "unstated local settings."
        ),
        "Q4": (
            "Filter expressions now carry parameter provenance and evaluation time; "
            "scheduled and interactive runs display the resolved values, and a "
            "missing governed parameter blocks execution instead of falling back to "
            "an unbounded result."
        ),
    },
    "reporting.formulas": {
        "Q2": (
            "Report formulas now declare evaluation grain and aggregation stage; "
            "row, group, and grand-total formulas cannot be substituted for one "
            "another, and reusable business logic must be promoted to a governed "
            "calculated field."
        ),
        "Q4": (
            "Report formulas now use versioned dependencies and deterministic "
            "evaluation contexts; a field or rate-basis change marks affected "
            "formulas for revalidation, and published results identify the formula "
            "version used for every derived measure."
        ),
    },
    "reporting.joined_reports": {
        "Q2": (
            "Joined reports now require each block to declare its grain and a "
            "compatible grouping contract; blocks remain independently secured and "
            "aggregated, and unmatched groups are displayed explicitly rather than "
            "being interpreted as arbitrary row-level joins."
        ),
    },
    "reporting.dashboards": {
        "Q2": (
            "Dashboards now declare a security mode per component—viewer-bound or "
            "approved running identity—and display that mode with the component's "
            "refresh time; components with incompatible modes cannot share a filter "
            "that would expose hidden members."
        ),
    },
    "reporting.recipients": {
        "Q3": (
            "Report delivery now resolves recipients at send time against workspace "
            "policy, data classification, and authenticated identity; unresolved "
            "groups and newly ineligible recipients are withheld with a delivery "
            "audit rather than receiving a stale scheduled export."
        ),
    },
    "reporting.large_data": {
        "Q3": (
            "Large report workloads now use a governed query budget based on scanned "
            "rows, joins, and calculation cost; an over-budget interactive query "
            "returns a plan and must move to an asynchronous, aggregated, or "
            "partitioned execution path."
        ),
    },
    "reporting.snapshots": {
        "Q3": (
            "Snapshots now preserve the rendered result together with definition "
            "version, running identity, filters, freshness, timezone, currency "
            "basis, and lineage; they are immutable evidence and still cannot "
            "restore source CRM records."
        ),
    },
    "reporting.history": {
        "Q3": (
            "Historical reporting now distinguishes event-time and correction-time "
            "views over enabled tracked fields; late corrections append effective-"
            "dated events, and untracked fields remain unreconstructable rather than "
            "being inferred from current values."
        ),
    },
    "reporting.fiscal_calendar": {
        "Q3": (
            "Fiscal calendar changes now create an effective-dated calendar version; "
            "new report runs use the version applicable to their period, while saved "
            "historical snapshots retain their original labels and transaction "
            "timestamps are never rewritten."
        ),
    },
    "reporting.timezone": {
        "Q3": (
            "Report timezone is now an explicit execution parameter chosen from the "
            "definition, schedule, or approved viewer context; grouping records the "
            "resolved zone and daylight-saving rules while stored instants remain "
            "UTC."
        ),
    },
    "reporting.trash": {
        "Q4": (
            "Deleted reports now enter a dependency-aware Trash: definitions, folder "
            "references, schedules, and dashboard links are inventoried, restoration "
            "requires compatible destinations, and permanent deletion is blocked "
            "while a governed evidence hold applies."
        ),
    },
    "reporting.troubleshooting": {
        "Q4": (
            "Report discrepancy analysis now creates a reproducibility bundle with "
            "definition version, running identity, freshness watermarks, resolved "
            "filters, grain, currency, timezone, and sample record lineage; fixes "
            "must target the first diverging layer rather than broadening access."
        ),
    },
    "onboarding.workspace_creation": {
        "Q2": (
            "Production workspace creation now uses a signed provisioning manifest "
            "covering legal customer, region, initial Owner, plan, residency, and "
            "identifier; immutable attributes are validated before allocation, and "
            "a failed check leaves no partially usable workspace."
        ),
        "Q4": (
            "Production workspace creation now uses a two-party provisioning "
            "ceremony: customer and SalesX attest the manifest, residency and "
            "identity recovery are tested before activation, and configuration is "
            "released atomically from a quarantined build state."
        ),
    },
    "onboarding.trial": {
        "Q2": (
            "Trials now carry an Evaluation classification that blocks production "
            "integrations, regulated data, and broad external sharing until "
            "conversion; approved test configuration may be retained, but trial "
            "activity is not production acceptance evidence."
        ),
        "Q4": (
            "Trials now use policy-enforced evaluation zones with synthetic-data "
            "defaults, isolated connectors, and expiring collaborators; conversion "
            "requires a boundary scan, and prohibited data or dependencies must be "
            "removed before any configuration can enter production."
        ),
    },
    "onboarding.implementation_roles": {
        "Q2": (
            "Implementation roles now use a responsibility matrix assigning one "
            "accountable owner and named approvers for identity, data, integrations, "
            "reporting, training, cutover, and rollback; an unowned workstream cannot "
            "pass its readiness gate."
        ),
        "Q4": (
            "The implementation responsibility matrix now includes delegated "
            "authority limits, alternates, decision deadlines, and automatic "
            "escalation; loss of an accountable owner freezes that workstream's "
            "approvals without blocking unrelated completed gates."
        ),
    },
    "onboarding.discovery": {
        "Q2": (
            "Discovery now produces a traceable requirements baseline linking each "
            "business outcome to process owner, data source, security boundary, "
            "integration, report, compliance obligation, and measurable acceptance "
            "criterion."
        ),
        "Q4": (
            "Discovery now maintains an effective-dated decision ledger: requirement "
            "changes identify affected controls, mappings, tests, training, and "
            "cutover criteria, and a material change reopens only the dependent "
            "approvals rather than silently editing the accepted baseline."
        ),
    },
    "onboarding.invitations": {
        "Q2": (
            "Invitations now bind verified email, domain route, minimal role, seat, "
            "sponsor, and expiry in one claim; acceptance succeeds only for the "
            "matching authenticated identity, and unaccepted invitations grant no "
            "workspace access."
        ),
    },
    "onboarding.csv_preparation": {
        "Q2": (
            "CSV preparation now produces a signed source manifest containing "
            "encoding, schema, external-ID rules, reference dictionaries, row count, "
            "and checksum; import validation rejects a file whose bytes or declared "
            "mapping differ from that reviewed manifest."
        ),
    },
    "onboarding.import_order": {
        "Q3": (
            "Migration order now uses a dependency graph generated from mappings: "
            "independent stages may run in parallel, dependent stages wait for "
            "reconciled parent checkpoints, and cycles require an explicit deferred-"
            "link pass after both sides exist."
        ),
    },
    "onboarding.external_ids": {
        "Q3": (
            "External IDs now have a governed namespace, source owner, uniqueness "
            "rule, formatting contract, and non-reuse policy; crosswalk changes are "
            "versioned, and a collision is quarantined rather than matched by a "
            "mutable display name."
        ),
    },
    "onboarding.validation_rules": {
        "Q3": (
            "Migration bypasses now target named validation rules and record classes, "
            "require equivalent compensating checks, expire automatically, and "
            "produce a post-load exception report; workspace-wide disablement is no "
            "longer an approved migration path."
        ),
    },
    "onboarding.sandbox_rehearsal": {
        "Q3": (
            "Migration rehearsal now uses a statistically representative, classified "
            "dataset and records duration, throughput, automation effects, failures, "
            "reconciliation, and rollback evidence; production approval requires "
            "each metric to meet its declared tolerance."
        ),
    },
    "onboarding.guided_sessions": {
        "Q3": (
            "Guided sessions now follow an outcome charter with prepared decisions, "
            "customer executor, SalesX adviser, evidence to review, and completion "
            "criteria; configuration remains customer-controlled, and unresolved "
            "actions carry to a named owner rather than becoming outsourced work."
        ),
    },
    "onboarding.academy": {
        "Q3": (
            "Readiness training now maps production roles to required learning and "
            "scenario assessments; completion alone is insufficient when a learner "
            "fails the role assessment, and temporary access may be limited until "
            "the gap is remediated."
        ),
    },
    "onboarding.sample_data": {
        "Q4": (
            "Sample data now carries immutable provenance labels and is excluded from "
            "production automation, analytics, and outbound integration by default; "
            "before go-live it must be deleted or moved to an explicitly isolated "
            "training partition."
        ),
    },
    "onboarding.rollback": {
        "Q4": (
            "Cutover rollback now uses continuously evaluated trigger metrics and a "
            "signed decision checkpoint; invocation freezes writes, preserves "
            "evidence, restores the declared source authority, and requires bidirectional "
            "reconciliation before another cutover attempt."
        ),
    },
}


ADDITIONAL_CHANGE_DIMENSIONS = {
    "billing.plans": {
        "Q2": ["entitlement_representation", "permission_boundary"],
        "Q4": ["effective_dating", "impact_preview"],
    },
    "billing.trial_conversion": {
        "Q2": ["conversion_inventory", "activation_precondition"],
        "Q4": ["signed_manifest", "atomic_conversion"],
    },
    "billing.proration": {
        "Q2": ["billing_start_event", "failed_provisioning"],
        "Q4": ["milestone_proration", "staged_activation"],
    },
    "billing.payment_methods": {
        "Q2": ["ownership_verification", "default_activation"],
        "Q4": ["separation_of_duties", "emergency_exception"],
    },
    "billing.failed_payments": {
        "Q2": ["failure_classification", "recovery_path"],
    },
    "billing.tax": {
        "Q2": ["tax_snapshot", "prospective_correction"],
    },
    "billing.currency": {
        "Q3": ["renewal_currency_migration", "conversion_basis"],
    },
    "billing.purchase_orders": {
        "Q3": ["purchase_order_lifecycle", "authorization_balance"],
    },
    "billing.credits": {
        "Q3": ["credit_wallet", "allocation_order"],
    },
    "billing.cancellation": {
        "Q3": ["closure_instructions", "early_termination_boundary"],
    },
    "billing.amendments": {
        "Q3": ["effective_dated_preview", "dependency_impact"],
    },
    "billing.billing_access": {
        "Q3": ["purpose_capabilities", "separation_of_duties"],
    },
    "billing.expiry_data": {
        "Q4": ["staged_closure", "legal_hold"],
    },
    "billing.disputes": {
        "Q4": ["ledger_partition", "linked_resolution"],
    },
    "integrations.api_limits": {
        "Q2": ["reserved_capacity", "endpoint_weighting"],
        "Q4": ["adaptive_budget", "burst_lease"],
    },
    "integrations.rate_handling": {
        "Q2": ["coordinated_backoff", "concurrency_cap"],
        "Q4": ["retry_epoch", "fairness_token"],
    },
    "integrations.bulk_api": {
        "Q2": ["validate_execute_split", "approved_manifest"],
        "Q4": ["partition_checkpoint", "immutable_job_definition"],
    },
    "integrations.webhook_config": {
        "Q2": ["versioned_subscription", "verification_delivery"],
        "Q4": ["atomic_promotion", "drain_window"],
    },
    "integrations.webhook_retries": {
        "Q2": ["failure_classification", "controlled_replay"],
    },
    "integrations.connected_apps": {
        "Q2": ["application_lifecycle", "scope_reapproval"],
    },
    "integrations.sandboxes": {
        "Q3": ["environment_binding", "secret_recreation"],
    },
    "integrations.connector_sync": {
        "Q3": ["stage_watermarks", "cycle_consistency"],
    },
    "integrations.field_mapping": {
        "Q3": ["field_contract", "queued_record_safety"],
    },
    "integrations.idempotency": {
        "Q3": ["canonical_payload_digest", "replay_outcome"],
    },
    "integrations.network": {
        "Q3": ["signed_range_feed", "activation_window"],
    },
    "integrations.logs": {
        "Q3": ["tiered_disclosure", "case_bound_diagnostics"],
    },
    "integrations.secret_rotation": {
        "Q4": ["credential_slots", "compromise_boundary"],
    },
    "integrations.deprecation": {
        "Q4": ["retirement_milestones", "time_boxed_bridge"],
    },
    "permissions.role_model": {
        "Q2": ["policy_graph", "decision_explanation"],
        "Q4": ["decision_evidence", "cache_invalidation"],
    },
    "permissions.owner_admin": {
        "Q2": ["reserved_authority", "capability_delegation"],
        "Q4": ["succession_policy", "delegation_charter"],
    },
    "permissions.custom_roles": {
        "Q2": ["role_versioning", "impact_preview"],
        "Q4": ["assignment_cohort", "privilege_migration"],
    },
    "permissions.record_access": {
        "Q2": ["purpose_label", "access_path"],
        "Q4": ["sensitivity_ceiling", "event_reevaluation"],
    },
    "permissions.field_access": {
        "Q2": ["field_capability_split", "masked_value_boundary"],
    },
    "permissions.hierarchy": {
        "Q2": ["hierarchy_boundary", "object_policy"],
    },
    "permissions.teams": {
        "Q3": ["expiring_membership", "sponsor_recertification"],
    },
    "permissions.mfa": {
        "Q3": ["factor_assurance", "restricted_recovery"],
    },
    "permissions.sso": {
        "Q3": ["population_routing", "recovery_separation"],
    },
    "permissions.sessions": {
        "Q3": ["risk_based_session", "action_step_up"],
    },
    "permissions.approvals": {
        "Q3": ["policy_binding", "pending_reevaluation"],
    },
    "permissions.audit": {
        "Q3": ["integrity_link", "append_only_correction"],
    },
    "permissions.break_glass": {
        "Q4": ["two_person_release", "fixed_privilege_bundle"],
    },
    "permissions.ownership_transfer": {
        "Q4": ["staged_handover", "atomic_authority_move"],
    },
    "reporting.report_types": {
        "Q2": ["semantic_contract", "breaking_version"],
        "Q4": ["compatibility_guarantee", "pinned_major_version"],
    },
    "reporting.builder_access": {
        "Q2": ["authoring_lifecycle", "audience_validation"],
        "Q4": ["release_workflow", "security_simulation"],
    },
    "reporting.filters": {
        "Q2": ["typed_expression", "saved_context"],
        "Q4": ["parameter_provenance", "resolved_value_disclosure"],
    },
    "reporting.formulas": {
        "Q2": ["evaluation_grain", "aggregation_stage"],
        "Q4": ["dependency_versioning", "formula_revalidation"],
    },
    "reporting.joined_reports": {
        "Q2": ["grouping_contract", "independent_security"],
    },
    "reporting.dashboards": {
        "Q2": ["component_security_mode", "filter_compatibility"],
    },
    "reporting.recipients": {
        "Q3": ["send_time_resolution", "ineligible_withholding"],
    },
    "reporting.large_data": {
        "Q3": ["query_budget", "execution_path"],
    },
    "reporting.snapshots": {
        "Q3": ["reproducibility_context", "immutable_evidence"],
    },
    "reporting.history": {
        "Q3": ["event_time", "late_correction"],
    },
    "reporting.fiscal_calendar": {
        "Q3": ["calendar_version", "prospective_evaluation"],
    },
    "reporting.timezone": {
        "Q3": ["explicit_execution_zone", "resolved_context"],
    },
    "reporting.trash": {
        "Q4": ["dependency_inventory", "evidence_hold"],
    },
    "reporting.troubleshooting": {
        "Q4": ["reproducibility_bundle", "first_diverging_layer"],
    },
    "onboarding.workspace_creation": {
        "Q2": ["provisioning_manifest", "atomic_allocation"],
        "Q4": ["two_party_attestation", "quarantined_build"],
    },
    "onboarding.trial": {
        "Q2": ["evaluation_classification", "production_boundary"],
        "Q4": ["evaluation_zone", "conversion_boundary_scan"],
    },
    "onboarding.implementation_roles": {
        "Q2": ["responsibility_matrix", "readiness_ownership"],
        "Q4": ["delegated_authority", "owner_loss_behavior"],
    },
    "onboarding.discovery": {
        "Q2": ["requirements_traceability", "measurable_acceptance"],
        "Q4": ["decision_ledger", "dependent_reapproval"],
    },
    "onboarding.invitations": {
        "Q2": ["identity_bound_claim", "preacceptance_access"],
    },
    "onboarding.csv_preparation": {
        "Q2": ["source_manifest", "content_integrity"],
    },
    "onboarding.import_order": {
        "Q3": ["dependency_graph", "deferred_link"],
    },
    "onboarding.external_ids": {
        "Q3": ["governed_namespace", "collision_quarantine"],
    },
    "onboarding.validation_rules": {
        "Q3": ["targeted_bypass", "compensating_check"],
    },
    "onboarding.sandbox_rehearsal": {
        "Q3": ["representative_rehearsal", "metric_tolerance"],
    },
    "onboarding.guided_sessions": {
        "Q3": ["outcome_charter", "customer_execution"],
    },
    "onboarding.academy": {
        "Q3": ["role_assessment", "access_readiness"],
    },
    "onboarding.sample_data": {
        "Q4": ["provenance_label", "production_exclusion"],
    },
    "onboarding.rollback": {
        "Q4": ["trigger_metric", "bidirectional_reconciliation"],
    },
}


RULE_CHANGES_40 = {
    key: dict(changes) for key, changes in RULE_CHANGES.items()
}
for key, changes in ADDITIONAL_RULE_CHANGES.items():
    RULE_CHANGES_40.setdefault(key, {}).update(changes)


CHANGE_DIMENSIONS_40 = {quarter: {} for quarter in DRIFT_KEYS_40}
for quarter, categories in DRIFT_KEYS_40.items():
    for category, article_keys in categories.items():
        for article_key in article_keys:
            key = f"{category}.{article_key}"
            if article_key in DRIFT_KEYS[quarter][category]:
                CHANGE_DIMENSIONS_40[quarter][key] = CHANGE_DIMENSIONS[key]
            else:
                CHANGE_DIMENSIONS_40[quarter][key] = (
                    ADDITIONAL_CHANGE_DIMENSIONS[key][quarter]
                )
