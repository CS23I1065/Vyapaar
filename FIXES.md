# Fixes — Meaningful Issues Found and Resolved

A log of the real bugs, design flaws, and wrong assumptions found during development of Vyapaar. Only entries that represent non-trivial decisions, real architectural changes, or confirmed attack vectors are kept here. Minor test fixture issues and infra noise are omitted.

---

### F1 — JWKS Kid Mismatch Check Was a No-Op

**Module:** `mandates/keys.py`

**Problem:** `KeyDirectory.from_jwks` assigned `kid` from the claimed value, then compared it to itself — so a forged `kid` that did not match its own key material was never actually caught. The test `test_jwks_entry_with_mismatched_kid_rejected` failed with "DID NOT RAISE."

**Fix:** The real thumbprint is now computed unconditionally into `real_kid` and compared against `claimed_kid` separately. A mismatch raises.

---

### F2 — Mandate Models Were Mutable (Verify/Evaluate Desync Risk)

**Module:** `mandates/` (all schemas)

**Problem:** `evaluate()` verifies a signature against an `Envelope`'s raw payload, then runs gates against separately-typed `IntentMandate`/`CartMandate` objects. Because those were ordinary mutable `BaseModel`s, a bug or crafted attack could mutate one **after** signature verification but **before** the budget/provenance/constraint gates ran — passing a valid signature check while every other gate saw different data than what was actually signed.

**Fix:** `FrozenModel` introduced as the base for every mandate/value schema (`model_config = ConfigDict(frozen=True)`). `Envelope` stays mutable (tests need to mutate `payload` to simulate tampering). Permanent regression tests added for in-place mutation attempts on both `IntentMandate` and `CartMandate`.

---

### F3 — Gemini Thinking Tokens Silently Inflated Cost ~30x

**Module:** `llm/provider.py`

**Problem:** Thinking is on by default and billed as output tokens — on `gemini-3.5-flash` (priced at $9/M output), a trivial "Say OK" prompt burned 276 thinking tokens against 8 real output tokens.

**Fix:** `GeminiProvider.complete()` forces `thinking_budget=0` by default on every call.

---

### F4 — robots.txt Fail-Open on Redirect and on 5xx

**Module:** `merchant_kit/robots.py`

**Problem:** `check()` used bare `httpx.get(...)`. httpx defaults `follow_redirects=False` — a well-known gotcha distinct from `requests`. A shop serving robots.txt via redirect (e.g. `chumbak.com` — apex → https → www in two hops) got a 301 back, `raw_text` stayed `None`, that parsed to an empty ruleset, and `can_fetch()` returned `True`: "no rules published, crawl freely," on a shop with perfectly valid rules one hop away. Separately, a 5xx response (RFC 9309: assume complete disallow) also collapsed to "allow."

**Fix:** `check()` rewritten with `httpx.Client(follow_redirects=True, max_redirects=5)` and explicit status-class branching: 2xx parses, 4xx is genuinely permissive (RFC-correct), 5xx and network failures disallow. `RobotsCheck.reason` records which branch fired so the readiness report explains why something was allowed.

---

### F5 — No Gate Checked Cart Price Against Merchant's Own Published Catalog (2.4× Overcharge Cleared All 17 Gates)

**Module:** `buyer/policy_engine.py`, `buyer/catalog.py`

**Problem:** A constructed exploit — merchant publishes ₹200 in its own signed `agent-catalog.json`, signs a cart charging ₹480 for the same SKU, against a ₹500 budget — cleared every gate. `grep -rn catalog buyer/` returned zero hits. G1.1 checked that a price was merchant-signed and verified; it never checked that signed price against anything the merchant had separately published. G2.1 saw consistent arithmetic. G4.x saw it under budget. The red-team corpus missed the same hole: `goal_A4_price_swap` scores success only when the charge exceeds the buyer's **budget**, so this exact overcharge — inside the cap — would have scored as DEFENDED.

**Fix:** `buyer/catalog.py` added (signature-verified `VerifiedCatalog` loaded independently by the buyer). G8.x anomaly band added to the policy engine:
- G8.1 `CATALOG_PRICE_MISMATCH` — a line may charge at or below published price, never above
- G8.2 `CATALOG_ITEM_UNKNOWN` — every SKU and category must match a published entry
- G8.3 `CART_SHAPE_INVALID` — duplicate SKUs, too many lines, or >1 upsell line

All three fail CLOSED on a missing catalog. `redteam/A6_catalog_overcharge` added so the corpus can no longer report a clean sheet over this hole.

---

### F6 — Approving an Escalation Broke the Cart's Chain Binding

**Module:** `buyer/approval.py`, `buyer/negotiate.py`

**Problem:** The first `resolve_escalation()` re-signed an amended `IntentMandate` and re-ran `evaluate()` against the **same unchanged `CartMandate`**. But `CartMandate.intent_hash` cryptographically binds a cart to the exact intent it was built against — widening a budget changes the intent's canonical hash, so the old cart's binding is genuinely broken. G0.3 (`MANDATE_CHAIN_BROKEN`) caught this correctly. The test `test_resolve_escalation_approved_reevaluates_from_scratch_and_passes` came back `REJECTED / MANDATE_CHAIN_BROKEN`.

**Fix:** `resolve_escalation()` now requires a `merchant_driver` and **re-negotiates a fresh, correctly-bound cart** from the merchant against the amended intent — via the same bounded `buyer/negotiate.py` loop — before re-evaluating. If re-negotiation doesn't produce a cart, `result.final_decision is None`. This is more correct design: a merchant re-confirming after a budget change is real security behavior.

---

### F7 — RFC 9421 Library Does Not Recompute Content-Digest From the Real Body

**Module:** `identity/content_digest.py`, `identity/verifier.py`

**Problem:** A request was signed, the body was swapped while leaving the old signed `Content-Digest` and `Signature` headers in place, and the `http_message_signatures` library's own `verify()` was called directly — it **passed**. The library only checks that the declared digest value matches what was covered in the signature base at signing time; it never recomputes from the actual current body. This is a live body-substitution hole.

**Fix:** `identity/content_digest.py` independently recomputes `sha-256=:...:` from the real body and compares byte-for-byte against the declared header, as a mandatory extra step in `verify_request()`. Permanent regression test: `test_content_digest_mismatch_after_body_substitution_is_caught`.

---

### F8 — RFC 9421 Library's Expiry Check Has No Clock-Injection Point

**Module:** `identity/verifier.py`

**Problem:** `HTTPMessageVerifier.validate_created_and_expires()` calls `datetime.datetime.now()` unconditionally. `max_age` only gates a separate "created too old" check — the "expires in the past" check runs against the real wall clock no matter what. Tests using a fixed injected clock all started failing with "Signature expires parameter is set to a time in the past."

**Fix:** The library's own clock is frozen in tests via `autouse` fixture (`_freeze_library_wall_clock`), monkeypatching `http_message_signatures.signatures.datetime.datetime` with a subclass whose `.now()` returns the fixed instant. Production code passes an effectively-infinite `max_age` and does all staleness checking against the project's own injected clock.

---

### F9 — CatalogItem.price_paise vs CartLine.unit_price_paise — Silent Field-Name Mismatch

**Module:** `merchant_agent/driver.py`

**Problem:** `merchant_kit/schemas.py`'s `CatalogItem` calls the field `price_paise`; `mandates/schemas.py`'s `CartLine` calls the same concept `unit_price_paise`. `_catalog_item_to_line()` built `field_provenance` keyed by `"price_paise"` instead of `"unit_price_paise"` — the provenance object existed and was correctly stamped, but it lived under a key nothing ever looked up. G1.1 saw an empty slot and correctly reported `PROVENANCE_INSUFFICIENT`. Every unit test for the driver in isolation passed fine; the first integration test that ran output through the real `evaluate()` caught it immediately.

**Fix:** Corrected the dict key to `"unit_price_paise"`.

---

### F10 — AOV Lift Was Computed by Comparing Two Different, Unpaired Populations

**Module:** `eval/metrics.py`

**Problem:** `upsell_mode="benign"` pitches on every revenue-arm scenario, and G4.x applies to the whole cart. A single-item catalog scenario with `max_quantity=1` gets a quantity-bump upsell attempt that fails G4.2, and the **entire cart is REJECTED** — including the primary item. The metric compared the mean order value of ALL approved `upsell_mode="off"` instances against ALL approved `upsell_mode="benign"` instances — two different, unpaired scenario populations — producing a meaningless (and negative: -51.25%) number.

**Fix:** AOV lift paired by scenario: for each scenario where the benign-arm upsell was accepted, compare that scenario's own `upsell_mode="off"` order value against its own accepted-upsell order value, then average the per-scenario differences. Real result: **+40.9% lift**.

---

### F11 — Live Interpretation Vocabulary Gap: Model Names Specific Products, Corpus Uses Abstract Catalog Buckets

**Module:** `eval/live_llm_eval.py`, `buyer/interpret.py`

**Problem:** First live evaluation run — `category_accuracy` and `pass_at_k` came back 6.7%. Not a comprehension failure: the model consistently named the specific product ("basmati rice") while the synthetic corpus ground truth used an abstract catalog bucket ("grains") the model was never told existed. `merchant_agent/driver.py`'s exact-category match (deliberately exact — fuzzy matching in a money-gating path reopens the imprecise-matching vulnerability class this project exists to close) then declined most correctly-understood requests.

**Fix:** `interpret()` gained `available_categories: list[str] | None`. When provided, the category output is constrained to an `enum` over the merchant's real catalog taxonomy via JSON schema. After fix: **category_accuracy 6.7% → 96.7%, pass_at_k 6.7% → 93.3%**.

---

### F12 — Attribute Over-Extraction: required_attributes Blocks 22 of 24 Non-Approved Scenarios

**Module:** `buyer/interpret.py`, `merchant_agent/driver.py`

**Problem:** After the category fix (#F11), E2E benign_utility barely moved (29.6% → 30.8%). The model turns almost any descriptive word into a structured `required_attributes` entry — `{"type": "basmati rice"}`, `{"item_type": "shirt"}`, `{"weight": "2kg"}` — against catalogs that mostly carry empty `attributes` dicts. The merchant driver's exact-attribute matching then declines. 22 of 24 non-approved scenarios failed this way, not on category.

**Fix:** `interpret()` gained `available_attribute_keys: dict[str, list[str]] | None` (per-category, not a flat union — a flat union would let one category's attribute key validate a different category's item). The schema-level enum is the union (JSON Schema can't condition one field's enum on another field's value), but the system prompt states the per-category breakdown explicitly. After fix: **E2E benign_utility 30.8% → 80.0%**.

---

### F13 — Live E2E False Approval Rate: Eval Harness Never Routed Through Review Step

**Module:** `eval/live_llm_eval.py`

**Problem:** After the attribute-key fix, `false_approval_rate` rose from 0.0% to 16% under both defended configs. All 4 false approvals traced to one root cause: the model had set `needs_clarification=True` on the draft (self-contradictory requests, explicit check-with-me tripwires), but `run_live_e2e_matrix`'s `intent_override` path signs the raw draft directly and never checks that flag. `buyer/review.py::conduct_review()` is the only module in the codebase that acts on `needs_clarification`. `evaluate()` itself never approved anything wrongly given the intent it was handed — the gap was in the eval harness, not the policy engine.

**Fix:** `apply_simulated_review()` added to `eval/live_llm_eval.py`. A draft still flagged `needs_clarification` is excluded from the matrix entirely (same treatment as a draft `interpret()` never signed — a real human would stop and ask). Verified by replaying the same saved live data through the fixed code, zero new API calls: **E2E coverage drops from 28/30 to 22/30** (honest cost of no longer scoring drafts a human would have stopped on), `false_approval_rate` back to **0.0%** on that reduced-but-honest population, `benign_utility` at **95%**.

---

### F14 — Upsells Violated G8.3 Cart Shape Policy (Duplicate SKUs / Double-Charging)

**Module:** `merchant_agent/driver.py`

**Problem:** The original driver handled upsells by appending an additional line. For a quantity bump, this created two lines for the same SKU — strictly violating G8.3 (`CART_SHAPE_INVALID`). For a premium substitute, it kept the original item AND added the substitute — double-charging the user. Both caused hard rejections, making the demo fail on any upsell attempt.

**Fix:**
- `QUANTITY` upsells now modify the `quantity` of the primary cart line.
- `PREMIUM_SUB` upsells replace the primary item entirely, marking the line `is_upsell=True`.
- The driver always provides the clean, non-upsold primary cart in `alternative_envelopes` as a guaranteed fallback.

---

### F15 — Smart Alternative Cart Contained Same SKUs as Primary Cart

**Module:** `web_demo/server.py`

**Problem:** The "smart alternative" cart was derived from the merchant's alternative cart envelope, but no check excluded items whose SKUs were already in the primary cart. Showing the same product in both carts is misleading and removes the value of a "smart alternative."

**Fix:** SKU-based filtering in `server.py`: when building the alternative cart display, any line whose SKU is already present in the primary cart is excluded.

---

### F16 — Accessory Upsells Pitch Unrelated Items — Felt Like Noise

**Module:** `merchant_agent/upsell.py`

**Problem:** The original `UPSELL_PRIORITY` tried `find_accessory` first — pitching a cheaper item from a *different* category (e.g. a mug to someone buying a magnet). A real helpful salesperson pitches a better version of what you want, not a random cheaper item from a different aisle.

**Fix:** Removed `find_accessory` from `UPSELL_PRIORITY`. Default priority is now: `find_premium_substitute` → `find_quantity_bump`. `find_accessory` remains available as an explicit manual option.

---

### F17 — Interactive Demo Signature Mismatch Due to Wrong Key

**Module:** `demo/interactive_buy.py`

**Problem:** The buyer's `IntentMandate` was drafted using `merchant_keypair.kid` (legacy hardcoded behavior) but signed with a dynamically generated `human_keypair`. This `kid` mismatch caused G0.2 (`SIGNATURE_INVALID`) — the `principal_kid` in the payload did not match the key actually used to sign it.

**Fix:** `human_keypair` is now generated at the top of the interactive session flow, and its `kid` is passed to `interpret()` so the drafted intent mandate correctly references the signing key from the start.

---

### F18 — Live Demo: Budget Put in Wrong Field; Synthetic Catalog Mismatched Live Interpretation Vocabulary

**Module:** `buyer/interpret.py`, `demo/fixtures.py`

**Problem (1):** Asked to interpret "Buy me a plain cotton shirt, up to 2000 rupees," Gemini returned `budget_per_item_rupees: 2000` and left `budget_total_rupees` unset, with `needs_clarification: false`. `interpret()`'s unconditional check overrode the model's correct judgment and asked a clarifying question about an already-stated amount.

**Fix (1):** At `max_quantity <= 1`, a missing `budget_total_rupees` now falls back to `budget_per_item_rupees`. Safe because per-item at quantity 1 is the total; does not apply at higher quantities.

**Problem (2):** With that fixed, `hard.category` came back as `"shirt"` and `required_attributes` included `material=cotton, pattern=plain` — the demo's synthetic catalog used `"apparel"` and carried no matching attributes.

**Fix (2):** `demo/fixtures.py` catalog aligned to `category="shirt"` with matching `material`/`pattern` attributes.

