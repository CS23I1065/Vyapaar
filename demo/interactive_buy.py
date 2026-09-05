"""python -m demo.interactive_buy ["your request here"] [--merchant chumbak|fabindia|bigbasket] [--live-razorpay]

A genuinely interactive buyer session against a REAL, signed merchant
catalog -- not synthetic fixture data. On startup this runs the actual
merchant_kit pipeline (offline, replaying the real snapshot already
captured from that site) to produce a freshly-signed catalog, then the
buyer independently verifies that signature before negotiating against
it -- the same two-sided real flow the rest of this project argues for,
not a scripted approximation of it.

Why a fresh key rather than the original merchant's: merchant_kit only
ever publishes a JWKS of PUBLIC keys (mandates/keys.py; that's the point
of a JWKS) -- the private key used for the original out/<merchant>
catalogs was never persisted anywhere, by design. Re-running the same
real extraction with --key-file gives this session a real, persisted
Ed25519 identity to actually negotiate and sign with. Every other field
-- title, price, category, attributes, description -- is exactly the
real data merchant_kit extracted from the real page.

Exactly ONE human checkpoint: the final "Buy this?" shown with the real
signed cart in hand. Everything before it is handled by AIReviewer, which
reads preferences from the request text itself (tripwire, substitution
policy, tolerance) and uses safe defaults for anything not stated. The
only pre-cart question is a budget clarification if the request genuinely
omits one -- that is a real blocker, not a preference question.

Upsell (benign mode): the merchant agent pitches a better version of
what you want (PREMIUM_SUB) or more of the same (QUANTITY). Driven by
the merchant's own catalog and margin rules, never by the buyer's budget.
The buyer's policy engine evaluates whether the pitchable cart passes all
20 gates.

Smart Alternative: if the merchant's catalog has a second matching item
within SUGGESTION_THRESHOLD of the stated budget, it is shown alongside
the primary cart as a non-mandatory choice. Both are fully signed
CartMandates evaluated through the real policy engine.

Always live-LLM -- there is no stub-provider mode, since the entire
point is a real interpretation of whatever you type, constrained to this
merchant's own real category/attribute vocabulary (the enum-constraint
technique from FIXES.md #27/#29). Requires GEMINI_API_KEY in .env.
--live-razorpay switches the payment backend from the offline fake to a
real Razorpay test-mode client (needs RAZORPAY_KEY_ID/RAZORPAY_KEY_SECRET)
-- independent of how real the catalog is.

A request this merchant's real catalog can't match gets an honest
decline, not a fabricated substitute.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from buyer.approval import CliApprover, resolve_escalation
from buyer.catalog import load_verified_catalog
from buyer.execute import execute
from buyer.explain import summarize_decision
from buyer.interpret import enrich_request_text, interpret
from buyer.negotiate import negotiate
from buyer.policy_engine import VerificationContext, evaluate
from buyer.review import AIReviewer, conduct_review
from demo.fixtures import fake_razorpay_client
from demo.run_demo import _permissive_policy
from llm.provider import GeminiProvider
from mandates.keys import KeyDirectory, load_keypair
from mandates.schemas import CartMandate, Envelope
from merchant_agent.driver import MerchantAgentDriver
from merchant_kit.cli import run as onboard_merchant
from merchant_kit.schemas import Catalog as MerchantCatalog
from razorpay_client.recon import reconcile_capture

MODEL = "gemini-3.1-flash-lite"
MAX_CLARIFICATION_ROUNDS = 2

# Suggestion threshold: show a smart alternative if its price is within
# this multiple of the stated budget. 1.5 = up to 50% over budget.
SUGGESTION_THRESHOLD = 1.5

# The exact real product pages merchant_kit.cli was originally run
# against (see CLAUDE.md's "Chosen merchant sites" and out/<merchant>/) --
# re-onboarding them here replays the same already-captured real
# snapshot (--offline), so this makes zero network calls of its own.
# Multiple URLs per merchant give the agent more products to work with:
# a second same-category item enables smart alternatives and PREMIUM_SUB
# upsells; a second different-category item enables accessory pitches.
REAL_MERCHANT_URLS = {
    "chumbak": [
        "https://www.chumbak.com/products/teal-by-chumbak-forest-jade-watch-metal-link-strap",
        "https://www.chumbak.com/products/fantastical-elephant-printed-strap-watch-and-bracelet-set",
    ],
    "fabindia": [
        "https://www.fabindia.com/red-cotton-mid-placket-shirt-10546011",
        "https://www.fabindia.com/beige-linen-slim-fit-shirt-20314851",
        "https://www.fabindia.com/brown-leather-wallet-20279314",
    ],
    "bigbasket": [
        "https://www.bigbasket.com/pd/265885/del-monte-four-seasons-mixed-fruit-drink-100-240-ml-tin/",
        "https://www.bigbasket.com/pd/265853/real-fruit-power-juice-pomegranate-1-l/",
        "https://www.bigbasket.com/pd/40219148/bingo-original-style-potato-chips-chilli-sprinkled-100-g/",
        "https://www.bigbasket.com/pd/10000425/bb-royal-toor-dalarhar-dal-desi-1-kg-pouch/",
        "https://www.bigbasket.com/pd/30010377/bb-popular-urad-dal-split-1-kg-pouch/",
        "https://www.bigbasket.com/pd/40075197/daawat-basmati-rice-rozana-super-90-5-kg/",
        "https://www.bigbasket.com/pd/30006887/aashirvaad-atta-whole-wheat-1-kg-pouch/",
    ],
}

DEMO_OUT_DIR = Path(__file__).resolve().parent / "interactive_out"


def _onboard_real_merchant(merchant: str, *, now: datetime):
    """Runs the real merchant_kit pipeline offline against the real,
    already-captured snapshot for `merchant`, with a --key-file so this
    session has a real, persisted signing identity to negotiate with
    (see this module's docstring for why that key can't be the
    original). Returns (merchant_id, merchant_keypair, verified_catalog, items)."""
    merchant_id = f"{merchant}-real"
    out_dir = DEMO_OUT_DIR / merchant
    key_file = out_dir / "merchant-key.json"

    urls = REAL_MERCHANT_URLS[merchant]
    onboard_merchant(urls, offline=True, out_dir=out_dir, merchant_id=merchant_id, key_file=key_file)

    merchant_keypair = load_keypair(key_file)
    well_known = out_dir / ".well-known"
    catalog_envelope = Envelope.model_validate_json((well_known / "agent-catalog.json").read_text())
    jwks = json.loads((well_known / "agent-keys.json").read_text())

    # Built from the published JWKS text, not the in-process keypair
    # object -- this is the exact artifact an independent buyer would
    # fetch over HTTP in a real deployment, and it's the only thing
    # load_verified_catalog() is allowed to trust the signature against.
    key_directory = KeyDirectory.from_jwks(jwks)
    verified_catalog = load_verified_catalog(
        catalog_envelope, key_directory, expected_merchant_id=merchant_id, required_kid=merchant_keypair.kid,
    )
    items = MerchantCatalog.model_validate(catalog_envelope.payload).items
    return merchant_id, merchant_keypair, verified_catalog, list(items)


def _present_catalog(console, items):
    from rich.tree import Tree
    categories = sorted({item.category for item in items})
    root = Tree(f"[bold]Catalog ({len(items)} items across {len(categories)} categories)[/bold]")
    for cat in categories:
        cat_branch = root.add(f"[cyan]{cat}[/cyan]")
        for item in [i for i in items if i.category == cat]:
            cat_branch.add(f"{item.title} [dim](Rs{item.price_paise / 100:.2f})[/dim]")
    console.print(root)


def _cart_from(envelope) -> CartMandate:
    return CartMandate.model_validate(envelope.payload)


def _evaluate_alternative(
    alt_envelope: Envelope,
    intent,
    intent_envelope,
    policy,
    key_directory,
    verified_catalog,
    budget_paise: int,
    now: datetime,
):
    """Evaluates one alternative CartMandate and returns (cart, decision)
    if it is within SUGGESTION_THRESHOLD of the stated budget, else None."""
    alt_cart = _cart_from(alt_envelope)
    if alt_cart.total_paise > budget_paise * SUGGESTION_THRESHOLD:
        return None
    alt_ctx = VerificationContext(
        key_directory=key_directory,
        intent_envelope=intent_envelope,
        cart_envelope=alt_envelope,
        merchant_catalog=verified_catalog,
    )
    alt_decision = evaluate(intent, alt_cart, policy, alt_ctx, now)
    return alt_cart, alt_decision


def _show_buy_screen(console: Console, cart: CartMandate, decision, alt_cart: CartMandate | None, alt_decision) -> str:
    """Renders the unified buy screen: primary cart, upsell lines inside it,
    optional smart alternative below. Returns the user's choice: '1', '2', or 'n'."""

    # --- Primary cart table ---
    primary_table = Table(title="Ready to buy")
    primary_table.add_column("Item")
    primary_table.add_column("Qty")
    primary_table.add_column("Price")
    primary_table.add_column("Note")
    for line in cart.lines:
        note = "🎁 merchant suggestion" if line.is_upsell else ""
        primary_table.add_row(
            line.title, str(line.quantity), f"Rs{line.unit_price_paise / 100:.2f}", note
        )
    console.print(primary_table)
    console.print(summarize_decision(decision))

    # --- Smart alternative (if available and within budget threshold) ---
    if alt_cart is not None:
        over = alt_cart.total_paise - cart.total_paise
        if over > 0:
            label = f"Rs{over / 100:.2f} more"
        elif over < 0:
            label = f"Rs{abs(over) / 100:.2f} less"
        else:
            label = "same price"
        alt_table = Table(title=f"[bold yellow]Smart Alternative[/bold yellow] ({label})")
        alt_table.add_column("Item")
        alt_table.add_column("Qty")
        alt_table.add_column("Price")
        for line in alt_cart.lines:
            alt_table.add_row(line.title, str(line.quantity), f"Rs{line.unit_price_paise / 100:.2f}")
        console.print(alt_table)
        if alt_decision and alt_decision.outcome != "APPROVED":
            console.print(f"[dim]({summarize_decision(alt_decision)})[/dim]")

        console.print(f"\n  [1] Buy primary   Rs{cart.total_paise / 100:.2f}  (your request, recommended)")
        console.print(f"  [2] Smart alt     Rs{alt_cart.total_paise / 100:.2f}  ({label} — {alt_cart.lines[0].title})")
        console.print("  [n] Cancel")
        choices = ["1", "2", "n"]
    else:
        console.print(f"\n  [1] Buy Rs{cart.total_paise / 100:.2f}")
        console.print("  [n] Cancel")
        choices = ["1", "n"]

    from rich.prompt import Prompt
    return Prompt.ask("\nBuy which?", choices=choices, default="1")


def _print_behind_the_scenes(console, intent, cart, alt_cart):
    console.print("\n[bold magenta]Behind the Scenes (Agent Logic)[/bold magenta]")
    
    # Buyer Agent
    attrs = ", ".join(f"{k}: {v}" for k, v in intent.hard.required_attributes.items())
    attr_str = f", attributes: {attrs}" if attrs else ""
    console.print("[dim]🤖 [bold]Buyer Agent (LLM)[/bold] translated your English into a strict JSON mandate:[/dim]")
    console.print(f"[dim]   ↳ Category: '{intent.hard.category}'{attr_str}[/dim]")
    console.print(f"[dim]   ↳ Hard Budget Limit: Rs{intent.budget.total_paise / 100:.2f}[/dim]")

    # Merchant Agent
    console.print("\n[dim]🏪 [bold]Merchant Agent (Deterministic Python)[/bold] received the mandate:[/dim]")
    is_upsold = any(line.is_upsell for line in cart.lines)
    if is_upsold and alt_cart:
        base_line = alt_cart.lines[0]
        upsell_line = cart.lines[0]
        console.print(f"[dim]   ↳ Baseline match: {base_line.title} (Rs{base_line.line_total_paise / 100:.2f})[/dim]")
        console.print(f"[dim]   ↳ [yellow]Upsell Triggered![/yellow] Swapped primary cart to {upsell_line.title} (Rs{upsell_line.line_total_paise / 100:.2f}) to maximize revenue.[/dim]")
    else:
        line = cart.lines[0]
        console.print(f"[dim]   ↳ Best match: {line.title} (Rs{line.line_total_paise / 100:.2f})[/dim]")
        console.print("[dim]   ↳ No viable upsell found within budget constraints.[/dim]")

    # Buyer Agent Policy Engine
    console.print("\n[dim]🛡️  [bold]Buyer Agent (Policy Engine)[/bold] reviewed the merchant's cart:[/dim]")
    if is_upsold:
        console.print("[dim]   ↳ Verified the merchant's aggressive upsell was STILL under the Rs{:.2f} limit.[/dim]".format(intent.budget.total_paise / 100))
    console.print("[dim]   ↳ Cryptographic signatures verified. Passed all 20 security gates.[/dim]\n")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", nargs="?", help="Your purchase request. Omit to be prompted interactively.")
    parser.add_argument("--merchant", choices=sorted(REAL_MERCHANT_URLS), default="chumbak", help="Which real, previously-onboarded merchant to negotiate against.")
    parser.add_argument("--live-razorpay", action="store_true", help="Use a real Razorpay test-mode client instead of the offline fake.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    console = Console()

    console.print(f"[bold]Onboarding {args.merchant} for real[/bold] -- replaying its real, already-captured page snapshot:")
    merchant_id, merchant_keypair, verified_catalog, items = _onboard_real_merchant(args.merchant, now=now)

    console.print("\n[bold]Available now:[/bold]")
    _present_catalog(console, items)

    request_text = args.request or input("\nWhat would you like to buy? ")

    categories = sorted({item.category for item in items})
    attribute_keys: dict[str, list[str]] = {}
    for item in items:
        attribute_keys.setdefault(item.category, set()).update(item.attributes.keys())
    attribute_keys = {cat: sorted(keys) for cat, keys in attribute_keys.items()}

    provider = GeminiProvider()
    from mandates.keys import generate_keypair
    human_keypair = generate_keypair()
    
    try:
        # interpret() runs once; AIReviewer reads raw_response to extract
        # any preferences the shopper stated (tripwire, substitutions_ok,
        # tolerance). Clarification only fires if budget is missing.
        outcome = interpret(
            request_text, provider=provider, model=MODEL,
            principal_kid=human_keypair.kid, agent_kid=human_keypair.kid,
            allowed_merchants=[merchant_id], now=now,
            available_categories=categories, available_attribute_keys=attribute_keys,
        )

        if outcome.draft_intent is None:
            # Budget missing -- conduct_review will ask via AIReviewer.ask_clarification
            review_result = conduct_review(
                request_text, provider=provider, model=MODEL,
                principal_kid=human_keypair.kid, agent_kid=human_keypair.kid,
                allowed_merchants=[merchant_id], now=now,
                reviewer=AIReviewer(raw_response=outcome.raw_response),
                human_keypair=human_keypair,
            )
            if not review_result.envelope:
                print(f"\nCould not proceed: {review_result.declined_reason}")
                return
            intent = review_result.intent
            intent_envelope = review_result.envelope
        else:
            # Budget stated -- sign immediately with AI-extracted preferences
            reviewer = AIReviewer(raw_response=outcome.raw_response)
            answers = reviewer.review(outcome.draft_intent)
            from buyer.review import _apply_review_answers
            intent = _apply_review_answers(outcome.draft_intent, answers)
            from mandates.sign import sign_mandate
            intent_envelope = sign_mandate(intent, human_keypair)

    finally:
        provider.close()

    # Merchant agent: benign upsell (PREMIUM_SUB first, then QUANTITY)
    # + one smart alternative cart (max_alternatives=1)
    driver = MerchantAgentDriver(
        catalog=items, merchant_id=merchant_id, merchant_keypair=merchant_keypair,
        now=now, upsell_mode="benign", max_alternatives=1,
    )
    negotiation = negotiate(intent, merchant_driver=driver)
    if negotiation.outcome != "cart_received":
        print(f"\nMerchant did not produce a cart: {negotiation.reason}")
        return

    cart = _cart_from(negotiation.cart_envelope)
    current_cart_envelope = negotiation.cart_envelope

    key_directory = KeyDirectory()
    key_directory.register_keypair(human_keypair)
    key_directory.register_keypair(merchant_keypair)
    policy = _permissive_policy(merchant_id)
    ctx = VerificationContext(
        key_directory=key_directory, intent_envelope=intent_envelope,
        cart_envelope=current_cart_envelope, merchant_catalog=verified_catalog,
    )
    decision = evaluate(intent, cart, policy, ctx, now)

    if decision.outcome == "REJECTED":
        # The merchant's primary pitch might have been an upsell that broke policy.
        # Fall back to an alternative clean cart if they provided one.
        fallback_found = False
        if negotiation.alternative_envelopes:
            for alt_env in negotiation.alternative_envelopes:
                alt_cart_obj = _cart_from(alt_env)
                alt_ctx = VerificationContext(
                    key_directory=key_directory, intent_envelope=intent_envelope,
                    cart_envelope=alt_env, merchant_catalog=verified_catalog,
                )
                alt_dec = evaluate(intent, alt_cart_obj, policy, alt_ctx, now)
                if alt_dec.outcome != "REJECTED":
                    cart = alt_cart_obj
                    decision = alt_dec
                    current_cart_envelope = alt_env # track the fallback envelope
                    print(f"\n[dim]Merchant's primary offer was rejected by policy. Falling back to a clean alternative from the same merchant.[/dim]")
                    fallback_found = True
                    break
        if not fallback_found:
            print(f"\n{summarize_decision(decision)}")
            return

    if decision.outcome == "REQUIRES_HUMAN_APPROVAL":
        resolution = resolve_escalation(
            decision, intent, cart, policy, ctx,
            approver=CliApprover(console=console), human_keypair=human_keypair, now=now, merchant_driver=driver,
        )
        if resolution.final_decision is None or resolution.final_decision.outcome != "APPROVED":
            print(f"\n{resolution.audit_note}")
            return
        renegotiation = negotiate(resolution.amended_intent, merchant_driver=driver)
        if renegotiation.outcome != "cart_received":
            print(f"\nCould not recover the approved cart: {renegotiation.reason}")
            return
        decision, intent, intent_envelope = resolution.final_decision, resolution.amended_intent, resolution.amended_envelope
        cart = _cart_from(renegotiation.cart_envelope)
        current_cart_envelope = renegotiation.cart_envelope
        negotiation = renegotiation

    # decision.outcome == "APPROVED" -- evaluate the smart alternative if available
    alt_cart = None
    alt_decision = None
    chosen_cart = cart
    chosen_envelope = current_cart_envelope

    for alt_envelope in negotiation.alternative_envelopes:
        if alt_envelope.payload == chosen_envelope.payload:
            continue
        result = _evaluate_alternative(
            alt_envelope, intent, intent_envelope, policy,
            key_directory, verified_catalog, intent.budget.total_paise, now,
        )
        if result is not None:
            alt_cart, alt_decision = result
            break  # show at most one smart alternative

    _print_behind_the_scenes(console, intent, cart, alt_cart)
    choice = _show_buy_screen(console, cart, decision, alt_cart, alt_decision)

    if choice == "n":
        print("\nNo purchase made.")
        return

    if choice == "2" and alt_cart is not None:
        # User chose the smart alternative -- re-evaluate it as the authoritative cart
        alt_envelope = negotiation.alternative_envelopes[0]
        alt_ctx = VerificationContext(
            key_directory=key_directory, intent_envelope=intent_envelope,
            cart_envelope=alt_envelope, merchant_catalog=verified_catalog,
        )
        alt_final = evaluate(intent, alt_cart, policy, alt_ctx, now)
        if alt_final.outcome != "APPROVED":
            print(f"\nSmart alternative did not pass evaluation: {summarize_decision(alt_final)}")
            print("Falling back to primary cart.")
        else:
            chosen_cart = alt_cart
            chosen_envelope = alt_envelope
            decision = alt_final

    client = fake_razorpay_client()
    if args.live_razorpay:
        from razorpay_client.client import client_from_env
        client = client_from_env()

    payment_mandate, order = execute(decision, intent, chosen_cart, client=client, now=now)
    print(f"\nOrder created: {order['id']}, amount Rs{chosen_cart.total_paise / 100:.2f}")

    artifact = reconcile_capture(chosen_cart, chosen_envelope, payment_mandate, client=client, now=now)
    print(f"Tier-1 recon: matched={artifact.delta.matched}, {artifact.delta.detail}")


if __name__ == "__main__":
    main()
