# Vyapaar — Verified Agentic Commerce

**Razorpay AI Buildathon, Track 01 (AI Growth & Agentic Commerce)**

Vyapaar is a two-sided agentic commerce system built around one rule:

> **The LLM proposes. It never authorizes.**

A buyer's AI agent interprets natural-language shopping requests and negotiates with a merchant agent. Every decision that moves money passes through a pure, deterministic, exhaustively-tested policy engine that no LLM output can bypass. The web-facing demo is called **AgentCart** — a chat-style interface where you select a merchant, describe what you want to buy, and watch both agents negotiate in real time.

---

## What Vyapaar Is

Vyapaar has two sides:

- **Merchant side** (`merchant_agent/`, `merchant_kit/`): A merchant AI-readiness toolkit that turns a real storefront page into a signed, machine-readable catalog. A merchant agent then matches buyer intents against that catalog, proposes carts, and can offer premium substitutes or quantity upsells.
- **Buyer side** (`buyer/`, `mandates/`, `identity/`): A buyer agent that interprets shopping requests, negotiates with the merchant, and submits every proposed cart to a deterministic policy engine before any payment is authorized.

The **AgentCart web demo** (`web_demo/`) ties both sides together in a chat UI suitable for live demos and judges.

---

## Architecture

Vyapaar has two independent halves. **Merchant onboarding** happens first — a real storefront page is turned into a signed, machine-readable catalog. **Buyer negotiation** happens at purchase time — the buyer agent reads that signed catalog, negotiates, and the policy engine decides.

```
MERCHANT ONBOARDING (one-time per merchant)
============================================

  Real product page URL
         |
         v
  merchant_kit/
    robots.py     -- enforce robots.txt before any fetch (fail-closed on 5xx)
    fetch.py      -- HTTP GET + snapshot cache (--offline replays from disk)
    extract/
      structured  -- JSON-LD / microdata / RDFa  (no LLM, preferred path)
      llm.py      -- Gemini fallback for JS-heavy SPAs (BigBasket)
    normalize.py  -- provenance-tagged CatalogItems
    sign.py       -- Ed25519 sign the catalog
    emit.py       -- write .well-known/agent-catalog.json
         |
         v
  out/<merchant>/
    agent-catalog.json   (signed)
    agent-policy.json
    agent-keys.json      (public keys only)

Run:  python -m merchant_kit.cli --url <product-page> --out-dir ./out


BUYER PURCHASE FLOW (per transaction)
======================================

  BUYER (AgentCart Chat UI)               MERCHANT
         |                                     |
         |  Natural-language request            |  Reads signed catalog from out/
         v                                     v
  +--------------------+         +----------------------------+
  | web_demo/server.py |-------->| merchant_agent/driver.py   |
  | (Demo controller)  |<--------| (Cart driver + upsell)     |
  +--------+-----------+         +----------------------------+
           |
           v
  buyer/
    interpret.py  --> Gemini proposes IntentMandate (never authorizes)
    negotiate.py  --> bounded loop, MAX_TURNS=6
                      (merchant pitch_text ignored by control flow)
    policy_engine --> 20 gates, PURE, deterministic, fail-closed
    execute.py    --> only ever called on APPROVED
           |
           v
  +--------------------+   +--------------------------------+
  | mandates/          |   | razorpay_client/               |
  | Ed25519 sign/verify|   | create_order, capture          |
  | JCS hash chain     |   | Tier 1 recon (capture-time)    |
  | Intent->Cart->Pay  |   | Tier 2 recon (HMAC webhook)    |
  +--------------------+   +--------------------------------+
```

**Tier 1 reconciliation** — runs immediately after payment: verifies the Razorpay-captured amount matches the merchant's signed cart total and the mandate hash binding written into the order's `notes`.

**Tier 2 reconciliation** — Razorpay fires an HMAC-SHA256 signed webhook when a payment is captured. This is an independent, out-of-band confirmation: even if Tier 1 had a bug, the webhook arrives separately and re-verifies against the same signed cart. Deduped by `X-Razorpay-Event-Id`. (Requires a public URL to receive live events — not configured in this environment, but tested against realistic payload shapes.)

### Key Modules

| Module | Role |
|---|---|
| `web_demo/server.py` | Central demo controller: interprets intents, routes to merchant agent, evaluates carts, executes payments |
| `web_demo/index.html` | AgentCart chat UI — merchant selector, chat interface, dual-cart view |
| `mandates/` | Intent -> Cart -> Payment schema chain, Ed25519 over RFC 8785 JCS, hash-linked |
| `buyer/policy_engine.py` | The pure, deterministic authority — 20 gates, one entry point, fail-closed |
| `buyer/interpret.py` | Turns free text into a typed IntentMandate draft via Gemini |
| `buyer/negotiate.py` | Bounded negotiation loop against the merchant driver |
| `buyer/execute.py` | Razorpay payment execution — only ever called after APPROVED |
| `merchant_agent/driver.py` | Catalog matching, cart mandate generation |
| `merchant_agent/upsell.py` | Deterministic upsell logic (premium substitute -> quantity bump) |
| `merchant_kit/` | Storefront -> signed catalog pipeline: robots -> fetch -> extract -> normalize -> sign -> emit |
| `identity/` | RFC 9421 HTTP Message Signatures — signing, verification, browse-only downgrade |
| `audit/` | Append-only, hash-chained, tamper-evident action log |
| `redteam/` | Attack variants A0-A6 with machine-checkable goal predicates |
| `razorpay_client/` | Live test-mode HTTP client, capture-time reconciliation |
| `eval/` | Ablation matrix (deterministic + live-AI) and report generator |

---

## The AgentCart Web Demo

Start the server:

```bash
# Kill any existing instance first if port is in use:
kill $(lsof -ti:8000)

python -m web_demo.server --port 8000
# Open: http://localhost:8000
```

### How It Works

1. **Select a merchant** from the sidebar (BigBasket, Fabindia, Chumbak, or a demo merchant).
2. **Type your request** in the chat box (e.g. "get me toor dal, urad dal, and rice").
3. The buyer agent interprets your request via Gemini, then negotiates with the selected merchant agent.
4. The **primary cart** is shown with line items, quantities, and total.
5. If the merchant offers a **smart alternative** (a premium substitute or different product), it appears in a second cart panel — never mixing SKUs with the primary cart.
6. If a requested item is **unavailable**, it appears in a dimmed section below the cart with a "Not Available" badge.
7. Click **Confirm & Pay** to execute the Razorpay payment. The button is disabled if the cart total is Rs 0.

### Chat Prompts That Demonstrate Features

| What to type | What you will see |
|---|---|
| `get me toor dal, urad dal, and rice` | Multi-item basket; unavailable items shown dimmed |
| `buy 1kg basmati rice` | Single item with upsell offer if available |
| `get me organic turmeric powder` | Smart alternative if a premium variant exists |
| `buy something over my budget` | Policy engine REJECTS; no payment button |

### The Two-Cart View

- **Primary cart**: The merchant's best matching cart for your request.
- **Smart alternative**: A second option the merchant proposes — a premium substitute or comparable product. Will never contain the same SKUs as the primary cart.
- **Unavailable items**: Products you requested but the merchant could not fulfill. Displayed dimmed with "Not Available" badge.

---

## The Policy Engine — 20 Gates

`buyer/policy_engine.py::evaluate()` is the sole authorization authority. It is:

- **Pure** — no I/O, no network, no LLM, no datetime.now() (clock is injected)
- **Deterministic** — identical inputs always produce identical outputs
- **Fail-closed** — any uncaught exception resolves to REJECTED / ENGINE_ERROR
- **Typed at the boundary** — LLM free text cannot reach it; only a provenance-tagged, signed, typed CartMandate can

Every gate always runs and is recorded, even when the outcome is already determined by an earlier gate.

| Gate | Code | What it checks | Severity |
|---|---|---|---|
| G0.1 | MANDATE_EXPIRED | Mandate expires_at has passed | REJECT |
| G0.2 | SIGNATURE_INVALID | Ed25519 verify fails on intent or cart | REJECT |
| G0.3 | MANDATE_CHAIN_BROKEN | Hash linkage between intent and cart is broken | REJECT |
| G0.4 | MERCHANT_NOT_AUTHORIZED | Merchant not in buyer's allowed list | REJECT |
| G0.5 | AGENT_IDENTITY_UNVERIFIED | RFC 9421 agent identity not verified | REJECT |
| G1.1 | PROVENANCE_INSUFFICIENT | Any price sourced from LLM inference or unverified | REJECT |
| G2.1 | CART_ARITHMETIC_MISMATCH | Line totals do not reconcile to cart total | REJECT |
| G3.1 | HARD_CONSTRAINT_VIOLATED | Required attribute missing or mismatched | REJECT |
| G3.2 | CATEGORY_NOT_ALLOWED | Line category in buyer's excluded list | REJECT |
| G4.1 | PER_ITEM_CAP_EXCEEDED | Unit price exceeds per-item budget | REJECT |
| G4.2 | QUANTITY_CAP_EXCEEDED | Quantity exceeds max-quantity limit | REJECT |
| G4.3 | BUDGET_EXCEEDED | Total exceeds total budget | REJECT |
| G5.1 | SUBSTITUTION_OUT_OF_TOLERANCE | Substitute price delta % exceeds tolerance | REJECT |
| G6.1 | MERCHANT_POLICY_CONFLICT | Merchant's published policy forbids it | REJECT |
| G7.1 | ESCALATION_THRESHOLD | Total exceeds buyer's own tripwire | ESCALATE |
| G7.2 | AMBIGUOUS_SUBSTITUTION | Substitute within tolerance but non-identical product | ESCALATE |
| G7.3 | LOW_CONFIDENCE_INTERPRETATION | Interpretation flagged load-bearing ambiguity | ESCALATE |
| G8.1 | CATALOG_PRICE_MISMATCH | Cart price above merchant's own signed catalog price | REJECT |
| G8.2 | CATALOG_ITEM_UNKNOWN | SKU or category not in published catalog | REJECT |
| G8.3 | CART_SHAPE_INVALID | Duplicate SKUs, too many lines, or >1 upsell line | REJECT |
| — | ENGINE_ERROR | Any uncaught exception | REJECT (fail closed) |

---

## Mandate Chain

Every transaction flows through a cryptographically linked chain:

```
IntentMandate  (buyer signs with Ed25519)
      |
      | intent_hash
      v
CartMandate  (merchant signs with Ed25519)
      |
      | cart_hash + intent_hash
      v
PaymentMandate  (recorded after APPROVED + execute())
```

- **IntentMandate**: The buyer's signed shopping request (budget, constraints, allowed merchants)
- **CartMandate**: The merchant's signed cart offer, cryptographically bound to the intent it responds to
- **PaymentMandate**: Final record, bound to both cart and intent hashes, written into the Razorpay order's `notes` at creation time

Any mutation after signing breaks the hash chain — caught at G0.3.

---

## Merchant Toolkit

`merchant_kit/` turns an ordinary storefront page into the signed catalog the buyer agent negotiates against:

1. **`robots.py`** — checks and enforces robots.txt before any page is fetched (fail-closed on 5xx or redirect errors)
2. **`fetch.py`** — plain HTTP GET with snapshot cache; `--offline` replays from disk with zero network calls
3. **`extract/structured.py`** — parses JSON-LD, microdata, and RDFa (no LLM)
4. **`extract/llm.py`** — LLM fallback only when structured extraction finds zero products
5. **`normalize.py` / `sign.py` / `emit.py`** — unify into provenance-tagged CatalogItems, Ed25519-sign, write `.well-known/agent-catalog.json`
6. **`report.py`** — readiness report per field: parsed, inferred, rejected, or derived

Live-tested against three real merchant sites:

| Site | Structure | Path used |
|---|---|---|
| Chumbak | Rich JSON-LD (Product, Offer, AggregateRating) | extract/structured.py |
| Fabindia | Minimal JSON-LD (sku/name/image/offers only) | extract/structured.py |
| BigBasket | Next.js SPA, zero static structured data | extract/llm.py |

Run the toolkit:

```bash
python -m merchant_kit.cli --url <product-page-url> [--url <another>] \
    [--offline] [--out-dir ./out] \
    [--key-file ./out/merchant-key.json] [--policy policy.toml]
```

---

## Security Design

### Anti-Injection Is Structural

Two properties hold regardless of what an LLM is told or shown:

- **Free text never reaches the policy engine.** A merchant's `pitch_text` is carried for audit and display only — `negotiate.py`'s control flow never reads it.
- **Signatures cannot be forged.** Every mandate schema is a FrozenModel — once verified, no field can be mutated before evaluation.

### Money Safety

- **Money is int paise everywhere.** No floats, no bare Decimal. Field names carry the unit (total_paise, unit_price_paise).
- **Field-level provenance.** CartLine.field_provenance is keyed by field name — G1.1 catches an LLM-inferred price on a schema.org-titled item.
- **Mandate chain is a real cryptographic binding.** Verified, not assumed, and confirmed to round-trip through the live Razorpay order's notes field.

### Upsell Rules

- **Merchant agent cannot change prices** — any price above the signed catalog price fails G8.1
- **Premium substitutes replace the primary line** — not appended (which would fail G8.3 CART_SHAPE_INVALID)

---

## Evaluation Results

### Deterministic Matrix (337 tests)

30 scenarios, 1,395 instances across 3 ablation configs and 6 attacks:

| Config | Benign Utility | False Approval | Attack Success Rate |
|---|---|---|---|
| C1 (undefended) | ~52% | ~48% | 87-93% |
| C2/C3 (policy engine) | 100% | 0.0% | 0% |
| C4 (+ identity) | 100% | 0.0% | 0% |

Revenue under C4: AOV lift +40.9% on accepted upsells, Rs 14,12,040 prevented across 165 attack instances.

### Live AI Evaluation

| Metric | Result |
|---|---|
| budget_accuracy | 96.7% |
| category_accuracy | 96.7% |
| attributes_accuracy | 93.3% |
| pass_at_k | 93.3% |
| E2E benign_utility | 80.0% |
| E2E false_approval_rate | 0.0% |

---

## Test Suite

**337 tests, 337 passing.**

```bash
pytest                    # full suite; live tests auto-skip without keys in .env
pytest -k "not live"      # fully offline
```

| File | Tests |
|---|---|
| test_policy_engine.py | 65 — every gate, cap boundaries (+-1), purity, fail-closed |
| test_merchant_kit.py | 35 — robots, multi-URL catalogs, per-merchant policy |
| test_buyer_loop.py | 32 — interpret, negotiate, approval re-signing, execute guard |
| test_mandates.py | 31 — tamper, chain breakage, expiry, immutability, JCS |
| test_eval.py | 28 — corpus, harness, metrics, ablation, report |
| test_live_llm_eval.py | 23 — fidelity/ambiguity scoring, controlled-vocabulary, simulated-review |
| test_redteam.py | 15 — A1-A6 at each attack's own layer |
| test_recon.py | 15 — Tier 1 capture, Tier 2 webhook, replay/tamper rejection |
| test_merchant_agent.py | 15 — upsell selection, budget asymmetry, evaluate() integration |
| test_identity.py | 17 — RFC 9421 round-trip, cross-verification, staleness |
| test_audit.py | 10 — chain integrity, tamper/delete/reorder detection |
| test_review.py | 8 — pre-signature review, clarification rounds |
| test_demo.py | 8 — five demo beats + cancel/recon coda |
| test_budget.py | 8 — spend guard, abort-before-exceed |
| test_shop.py | 7 — multi-merchant comparison, ranking |
| test_cancel.py | 7 — chain-bound refund, forged-cancellation refusal |
| test_razorpay_client.py | 6 — cassette record/replay, live order round-trip |
| test_llm_provider.py | 6 — banned models, timeout, structured output |
| test_human_key.py | 3 — human signing-key persistence |

---

## Setup

Requires **Python 3.11+**.

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in your API keys
pytest
```

`.env` keys:
- `GEMINI_API_KEY` — for intent interpretation in the demo
- `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` — for payment execution
- `RAZORPAY_WEBHOOK_SECRET` — for Tier 2 webhook reconciliation
- `LLM_MODEL_EVAL` / `LLM_MODEL_DEMO` — model overrides (defaults: gemini-3.1-flash-lite / gemini-3.5-flash)

---

## Step 1 — Merchant Onboarding

Before any purchase can happen, a merchant's storefront must be onboarded into a signed catalog. This is already done for the three demo merchants — their signed catalogs are checked in under `out/`. To onboard a new merchant or re-onboard an existing one:

```bash
# Onboard a single product page
python -m merchant_kit.cli --url https://example.com/product-page     --out-dir ./out --key-file ./out/merchant-key.json

# Onboard multiple pages into one catalog
python -m merchant_kit.cli     --url https://example.com/product-1     --url https://example.com/product-2     --out-dir ./out --key-file ./out/merchant-key.json

# Replay from cached snapshot (no network calls)
python -m merchant_kit.cli --url <url> --offline --out-dir ./out
```

The toolkit:
1. Checks `robots.txt` — stops immediately if crawling is disallowed
2. Fetches the page and caches a snapshot to disk
3. Extracts product data from JSON-LD/microdata (or Gemini fallback for JS-heavy pages)
4. Signs the catalog with Ed25519 and writes `out/<merchant>/agent-catalog.json`

The three real demo merchants already onboarded:

| Merchant | Products | Notes |
|---|---|---|
| BigBasket | 7 items (dals, rice, atta, snacks, drinks) | Extracted via Gemini (Next.js SPA) |
| Chumbak | 2 watches | Extracted from JSON-LD |
| Fabindia | 3 shirts + wallet | Extracted from JSON-LD |

---

## Step 2 — Run AgentCart

```bash
python -m web_demo.server --port 8000
# Open http://localhost:8000
```

The terminal prints a live log of every agent action — keep it visible in a split screen for demos.

---

## Scripted Demo (CLI) — Redteam Attack Suite

**`python -m demo.run_demo`** runs five beats, each demonstrating a real attack attempt against the policy engine. Every attack uses the actual `redteam/` module, is scored by a machine-checkable typed predicate, and runs through the real `evaluate()` — no mocked policy logic.

| Beat | Attack | What it proves |
|---|---|---|
| 1. Baseline | None (A0) | Clean purchase gets APPROVED — the happy path works |
| 2. Branded Whisper | A3 | Merchant injects persuasive text into `pitch_text`. Decision is **byte-identical** to the clean run. `pitch_text` has no path to the policy engine. |
| 3. Price Swap | A4 | Merchant re-signs cart with 10× inflated price (Rs 900 → Rs 9,000). Validly signed, dishonest. Caught by G4.1 / G4.3. |
| 4. Quantity Inflation | A5 | Merchant re-signs quantity ×10 (buyer asked for 1, cart says 10). Caught by G4.2. |
| 5. Catalog Overcharge | A6 | The **silent attack** — Rs 900 item charged at Rs 1,980, still inside the Rs 2,000 budget. No budget gate fires. Caught only by G8.1, which compares the cart price against the merchant's own signed catalog. Before G8.1 existed, this cleared all 17 gates. |

Ends with an **attack summary table** (ASR = 0% on all variants) and **audit chain verification** — 6 decisions logged, hash chain intact.

```bash
python -m demo.run_demo                         # fully offline, no API calls
python -m demo.run_demo --live-razorpay         # real Razorpay test-mode
python -m demo.run_demo --live-llm              # real Gemini
```

**`python -m demo.interactive_buy`** — interactive session against a real, signed catalog:

```bash
python -m demo.interactive_buy                          # defaults to chumbak
python -m demo.interactive_buy "buy the ravana magnet, budget 500 rupees"
python -m demo.interactive_buy "..." --merchant fabindia
python -m demo.interactive_buy "..." --merchant bigbasket
```

---

## Repo Layout

```
Vyapaar/
├── web_demo/          AgentCart demo server (server.py) and UI (index.html, static/)
├── mandates/          Intent/Cart/Payment schemas, JCS, Ed25519 sign/verify, hash chain
├── buyer/             interpret, review, negotiate, policy_engine, approval, execute, cancel
├── identity/          RFC 9421 signer + verifier, content-digest, ASGI middleware
├── merchant_agent/    Cart driver + deterministic upsell engine
├── merchant_kit/      robots -> fetch -> extract -> normalize -> sign -> emit -> serve
├── razorpay_client/   Test-mode HTTP client + reconciliation
├── redteam/           Attack variants A0-A6 + machine-checkable goal predicates
├── audit/             Append-only hash-chained JSONL + timeline renderer
├── llm/               Gemini provider, pricing/ban list, record-replay
├── eval/              Corpus, harness, ablation matrix, live-AI evaluation, report
├── demo/              Five scripted beats + interactive_buy
├── out/               Generated signed .well-known/ bundles for the 3 real sites
├── tests/             337 tests
├── README.md          This file
├── PROGRESS.md        Build steps and milestones
└── FIXES.md           Meaningful bugs and fixes
```

---

## Known Limitations

- **No captured payment_id yet.** Server-to-server payment creation returns 404 on this test account (account-level enablement gate). The demo uses Razorpay test-mode order creation, which works, but capture requires a live payment_id.
- **Catalog is point-in-time.** Not real-time sync — prices are from the last merchant_kit.cli run.
- **Real catalogs have one product each.** Only one product page was onboarded per site.
- **required_attributes matching is strict.** Exact-match attribute comparison — deliberate design to avoid fuzzy matching in money-gating logic.
- **Settlement reconciliation not built.** Razorpay test mode does not generate settlements. Tier 2 is webhook-based HMAC reconciliation instead.
