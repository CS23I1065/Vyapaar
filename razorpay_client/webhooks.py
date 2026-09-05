"""
Tier 2, webhook reconciliation -- the INDEPENDENT channel.

Tier 1 (recon.py) reconciles against what the buyer agent itself fetched
from Razorpay's REST API, so a channel that could fool the agent's own
HTTP client could in principle fool Tier 1 too. A webhook event arrives
from Razorpay out of band, pushed to a URL this process did not request
at the moment it matters -- that is what makes it a genuinely
independent check rather than the same check run twice.

Real HMAC-SHA256-signed events, not a mock: Razorpay fires real webhooks
in test mode with genuine signatures over the same payload shape as live
mode -- unlike settlements, which test mode cannot produce at all, which
is why this is the real Tier 2.

Three implementation footguns, all handled below:

  1. The signature is computed over the RAW, unparsed request body. Any
     re-serialization (even key-order-preserving json.dumps of the
     parsed dict) can produce different bytes than what Razorpay signed.
     verify_signature() takes `bytes`, never a dict, for exactly this
     reason.
  2. Signature comparison must be constant-time (hmac.compare_digest),
     not `==` -- a timing side-channel on webhook verification is a real
     attack surface, however small.
  3. Webhooks can be delivered more than once. Dedup on
     X-Razorpay-Event-Id, not on payload content (two genuinely different
     events can otherwise collide on a naive content hash).

Test/live secrets are separate on purpose (`RAZORPAY_WEBHOOK_SECRET_TEST`
vs `RAZORPAY_WEBHOOK_SECRET_LIVE`) -- a test-mode secret must never
successfully verify a live-mode event or vice versa.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field

from mandates.hashing import mandate_hash
from mandates.schemas import CartMandate, IntentMandate, PaymentMandate

SIGNATURE_HEADER = "X-Razorpay-Signature"
EVENT_ID_HEADER = "X-Razorpay-Event-Id"


class WebhookVerificationError(Exception):
    """Raised when a webhook's signature does not verify. The caller
    must not act on the payload at all -- not log it as a probable
    event, not partially process it. An unverified webhook is
    indistinguishable from one forged by an attacker who has no
    relationship with Razorpay at all."""


class WebhookReplay(Exception):
    """Raised when an event_id has already been processed. Not a
    verification failure -- the signature may be perfectly genuine, this
    is simply a redelivery, which Razorpay's own docs say can happen."""


def verify_signature(raw_body: bytes, signature: str, secret: str) -> bool:
    """Constant-time comparison over the RAW body -- never parse-then-
    reserialize before this call, since that can change the exact bytes
    Razorpay actually signed."""
    computed = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)


@dataclass
class WebhookDeduper:
    """In-memory dedup by event id. A real deployment would back this
    with persistent storage (a webhook can arrive after a process
    restart), but the DEDUP LOGIC -- check-then-record, atomic from the
    caller's perspective -- is what this class exists to get right
    regardless of backing store."""

    _seen: set[str] = field(default_factory=set)

    def is_new(self, event_id: str) -> bool:
        if event_id in self._seen:
            return False
        self._seen.add(event_id)
        return True


@dataclass(frozen=True)
class PaymentCapturedEvent:
    event_id: str
    event_type: str
    payment_id: str
    order_id: str
    amount_paise: int
    status: str
    notes: dict


class WebhookPayloadError(Exception):
    """The signature verified, but the payload does not have the shape a
    payment.captured event is documented to have. Distinct from
    WebhookVerificationError because the trust decision (is this really
    from Razorpay?) and the shape decision (is this the event we know how
    to handle?) are different failures with different remedies."""


def parse_payment_captured_event(raw_body: bytes, event_id: str) -> PaymentCapturedEvent:
    try:
        data = json.loads(raw_body)
        payment = data["payload"]["payment"]["entity"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise WebhookPayloadError(f"could not parse payment.captured event: {e}") from e

    return PaymentCapturedEvent(
        event_id=event_id,
        event_type=data.get("event", ""),
        payment_id=payment["id"],
        order_id=payment["order_id"],
        amount_paise=payment["amount"],
        status=payment.get("status", ""),
        notes=payment.get("notes", {}) or {},
    )


@dataclass(frozen=True)
class WebhookReconResult:
    matched: bool
    violations: tuple[str, ...]
    event: PaymentCapturedEvent


def reconcile_webhook_event(
    event: PaymentCapturedEvent,
    *,
    intent: IntentMandate,
    cart: CartMandate,
    payment_mandate: PaymentMandate,
) -> WebhookReconResult:
    """Checks the independently-arrived event against the SAME
    mandate-binding claim Tier 1 checks, so a discrepancy that only shows
    up in one channel is exactly the signal worth escalating: it would
    mean the buyer agent's own view of the order (fetched via REST) and
    Razorpay's own out-of-band notification disagree."""
    violations: list[str] = []

    if event.order_id != payment_mandate.razorpay_order_id:
        violations.append(
            f"event order_id {event.order_id!r} != payment_mandate.razorpay_order_id "
            f"{payment_mandate.razorpay_order_id!r}"
        )
    if event.amount_paise != cart.total_paise:
        violations.append(
            f"event amount {event.amount_paise} paise != signed cart total {cart.total_paise} paise"
        )
    if event.status != "captured":
        violations.append(f"event status {event.status!r} is not 'captured'")

    expected_intent_hash = mandate_hash(intent)
    expected_cart_hash = mandate_hash(cart)
    if event.notes.get("intent_hash") != expected_intent_hash:
        violations.append(f"event notes intent_hash {event.notes.get('intent_hash')!r} != {expected_intent_hash!r}")
    if event.notes.get("cart_hash") != expected_cart_hash:
        violations.append(f"event notes cart_hash {event.notes.get('cart_hash')!r} != {expected_cart_hash!r}")
    if event.notes.get("payment_mandate_id") != payment_mandate.mandate_id:
        violations.append(
            f"event notes payment_mandate_id {event.notes.get('payment_mandate_id')!r} != "
            f"{payment_mandate.mandate_id!r}"
        )

    return WebhookReconResult(matched=not violations, violations=tuple(violations), event=event)


def process_payment_captured_webhook(
    raw_body: bytes,
    *,
    signature: str,
    event_id: str,
    secret: str,
    deduper: WebhookDeduper,
    intent: IntentMandate,
    cart: CartMandate,
    payment_mandate: PaymentMandate,
) -> WebhookReconResult:
    """The full inbound path: verify signature -> dedup -> parse ->
    reconcile. Raises WebhookVerificationError / WebhookReplay /
    WebhookPayloadError rather than returning a sentinel -- an unverified
    or replayed webhook is not "a result to reconcile," it is a request
    this process must refuse to act on at all."""
    if not verify_signature(raw_body, signature, secret):
        raise WebhookVerificationError(f"signature did not verify for event {event_id!r}")
    if not deduper.is_new(event_id):
        raise WebhookReplay(f"event {event_id!r} already processed")

    event = parse_payment_captured_event(raw_body, event_id)
    return reconcile_webhook_event(event, intent=intent, cart=cart, payment_mandate=payment_mandate)
