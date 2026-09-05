# Vyapaar — Build Progress

All 15 build steps complete, plus audit pass, live-AI evaluation, and demo polish. 337/337 tests passing.

---

## Completed Steps

### Step 1 — Project Scaffold
- Conda environment (Python 3.11), `.gitignore`, `.env` / `.env.example`
- Project structure established across `mandates/`, `buyer/`, `merchant_kit/`, `merchant_agent/`, `identity/`, `razorpay_client/`, `audit/`, `llm/`, `eval/`, `redteam/`, `demo/`, `web_demo/`

### Step 2 — LLM Provider + Budget Guard (`llm/`)
- Gemini provider with hard timeout, `thinkingBudget=0` (prevents invisible cost inflation)
- Banned model list (`gemini-3.7-flash`, `gemini-flash-latest` — hang indefinitely)
- Record/replay cassette system for offline testing
- Spend guard with abort-before-exceed

### Step 3 — Mandate Schemas (`mandates/`)
- `IntentMandate`, `CartMandate`, `PaymentMandate` schemas
- JCS (RFC 8785) canonicalization for deterministic signing
- Ed25519 sign/verify with key directory and JWKS
- Hash-chained linkage: `cart.intent_hash == hash(intent)`, `payment.cart_hash == hash(cart)`
- All mandate models frozen (`FrozenModel`) to prevent post-verification mutation

### Step 4 — Policy Engine (`buyer/policy_engine.py`)
- 20-gate deterministic enforcement: G0.x (chain/identity), G1.1 (provenance), G2.1 (arithmetic), G3.x (constraints), G4.x (budget), G5.1 (substitution), G6.1 (merchant policy), G7.x (escalation), G8.x (catalog anomaly band)
- Pure function — no I/O, no LLM, no wall-clock reads (clock injected)
- Fail-closed: any uncaught exception → `REJECTED / ENGINE_ERROR`
- G8.x anomaly band added after audit: `CATALOG_PRICE_MISMATCH`, `CATALOG_ITEM_UNKNOWN`, `CART_SHAPE_INVALID`

### Step 5 — Merchant Toolkit (`merchant_kit/`)
- `robots.py` — fail-closed robots.txt enforcement (follows redirects, treats 5xx as disallow)
- `fetch.py` — HTTP GET with snapshot cache and `--offline` mode
- `extract/structured.py` — JSON-LD, microdata, RDFa parsing (no LLM)
- `extract/llm.py` — LLM fallback for SPA/JS-rendered pages (BigBasket)
- `normalize.py` — provenance-tagged `CatalogItem` with synthetic SKU derivation when absent
- `sign.py` / `emit.py` — Ed25519-signed `.well-known/agent-catalog.json`
- `report.py` — per-field readiness report
- `cli.py` — multi-URL intake, persistent merchant key across runs
- Live-tested against Chumbak, Fabindia, BigBasket

### Step 6 — LLM Extraction Fallback (`merchant_kit/extract/llm.py`)
- Handles Next.js SPAs with zero static structured data
- `_clean_str()` normalizes null-like strings from Gemini (`"null"`, `"none"`, `"n/a"`)
- Verified live against real BigBasket snapshot

### Step 7 — Razorpay Client (`razorpay_client/`)
- Live test-mode HTTP client scoped to working endpoints: `create_order`, `fetch_order`, `fetch_order_payments`, `fetch_payment`, `capture_payment`, `create_refund`, `fetch_settlement_recon`
- Cassette record/replay for offline testing
- S2S payment creation not built (404 account-level gating — documented, not worked around)

### Step 8 — Identity Layer (`identity/`)
- RFC 9421 HTTP Message Signatures — signing and verification
- Built on `http_message_signatures` library with three real defects found and fixed:
  1. Library does not recompute Content-Digest from the real body (body-substitution hole)
  2. Library's expiry check has no clock-injection point — production code injects its own clock
  3. Header lookup must be case-insensitive (`httpx.Headers`, not plain `dict`)
- Bidirectional cross-verification against the independent library (self-round-trip proves nothing)
- Browse-only downgrade middleware: catalog reads open, transact endpoints require valid RFC 9421 signature

### Step 9 — Buyer Agent Loop (`buyer/`)
- `interpret.py` — free text → typed `IntentMandate` draft via Gemini; returns `needs_clarification=True` instead of guessing missing budget; constrained category/attribute vocabulary via JSON-schema enum
- `review.py` — single pre-signature human touchpoint (`CliReviewer`)
- `negotiate.py` — bounded loop (`MAX_TURNS=6`); `pitch_text` carried for display only, never read by control flow
- `approval.py` — handles `REQUIRES_HUMAN_APPROVAL`; re-negotiates fresh cart (re-signing intent changes its hash, breaking old cart binding — G0.3 working correctly)
- `execute.py` — only runs on `APPROVED`; binds mandate hash chain into Razorpay order `notes`
- `cancel.py` — chain-bound `CancellationMandate` referencing original `PaymentMandate` by hash
- `explain.py` — plain-language translation of gate codes for human-facing output
- `rank.py` — brand/target-price/stock preferences change which item is bought
- `shop.py` — negotiates against several merchants, picks best APPROVED cart
- `human_key.py` — persistent Ed25519 signing key for the human buyer

### Step 10 — Merchant Agent (`merchant_agent/`)
- `driver.py` — catalog matching (exact category equality, attribute matching), cart mandate generation
- `upsell.py` — `UPSELL_PRIORITY`: `find_premium_substitute` → `find_quantity_bump`
  - Premium substitutes **replace** the primary cart line (not appended)
  - Quantity bumps **modify** the primary line's quantity (not a duplicate SKU)
  - `find_accessory` removed from default priority (unrelated category pitches feel noisy)
- Upsell functions cannot accept `IntentMandate` — buyer budget is structurally inaccessible
- Smart alternative (alternative cart) always provided for the buyer to fall back to if upsell is rejected

### Step 11 — Audit Log (`audit/`)
- Append-only hash-chained JSONL
- Tamper/delete/reorder detection verified against real files on disk
- Timeline renderer

### Step 12 — Red Team (`redteam/`)
- Attack variants A0–A6 with machine-checkable, non-LLM-judged goal predicates:
  - A1/A2: Extraction attacks
  - A3: Branded Whisper (narrative pitch injection) — proven byte-for-byte: injected `pitch_text` produces identical Decision to clean run
  - A4: Price swap
  - A5: Quantity inflation
  - A6: Catalog overcharge (added post-audit — 2.4× overcharge cleared all 17 original gates)

### Step 13 — Evaluation (`eval/`)
- 30-scenario corpus (12 clean, 10 policy-violating, 8 ambiguous); ground truth attached by construction
- C1/C2+C3/C4 ablation configs
- Metrics: benign utility, false-approval/-escalation, targeted ASR per attack, human-touchpoints-avg, upsell offer/would-pass rate, AOV lift % (paired per-scenario), Rs prevented
- Self-detecting HTML report
- Full deterministic matrix: **1,395 instances, $0.00, C2/C3/C4 benign_utility=100%, false_approval=0.0%, 0% ASR on all attacks**
- Live AI evaluation: **304 Gemini calls, ~$0.12, E2E benign_utility=80.0%, false_approval=0.0%**
  - Category vocabulary fix: 6.7% → 96.7% category_accuracy
  - Attribute-key fix: E2E benign_utility 30.8% → 80.0%
  - Simulated-review fix: false_approval_rate confirmed back to 0.0%

### Step 14 — Reconciliation (`razorpay_client/recon.py` + `webhooks.py`)
- Tier 1: `reconcile_capture` + `verify_notes_binding` — compares Razorpay captured amount against merchant's signed cart total and mandate hash binding
- Tier 2: `process_payment_captured_webhook` — real HMAC-SHA256 verification over raw body, dedup by `X-Razorpay-Event-Id`, separate test/live secrets

### Step 15 — Scripted Demo (`demo/`)
- Five beats: onboarding, revenue/upsell, comparison shopping, blocked purchase explained, injection blocked with receipts; plus cancel/recon coda
- Offline by default; `--live-llm` and `--live-razorpay` flags for real APIs
- `interactive_buy.py` — interactive session against real signed catalogs (re-runs `merchant_kit` pipeline offline on startup, persists fresh Ed25519 key, one "Buy this?" confirmation)
- Live-verified end to end against real Gemini and real Razorpay test-mode APIs

---

## Web Demo (AgentCart)

### Step 16 — AgentCart Web UI (`web_demo/`)
- Chat-style interface: merchant selector sidebar, message thread, dual-cart view
- Multi-item basket intent parsing (Gemini extracts multiple products from one message)
- Unavailable items rendered in a dimmed section with "Not Available" badge
- Smart alternative cart filtered to never overlap SKUs with primary cart
- Payment execution disabled when cart total is Rs 0
- Terminal logging active for split-screen judge demos
- Live server: `python -m web_demo.server --port 8000`

---

## Open Items

- **payment_id capture**: S2S payment creation is account-gated on the current Razorpay test key (404). A real `payment_id` requires either Razorpay enabling S2S for this account or a one-time manual test checkout.
- **Multi-product real catalogs**: Each real catalog (Chumbak, Fabindia, BigBasket) has exactly one product. Richer catalogs require onboarding additional product URLs per site.

