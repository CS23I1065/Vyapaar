"""
Real Razorpay test-mode HTTP client. Plain httpx over documented REST
endpoints (not the official SDK) so the cassette layer (cassettes.py) can
record/replay real JSON without depending on an SDK's internal request
shape.

Endpoint paths for the S2S payment-creation calls (create/upi,
create/json, otp_generate, otp/submit) were confirmed by reading the
official razorpay-python SDK's source directly, not guessed -- but those
two endpoints return 404 on a standard test account (S2S direct payment
creation needs per-merchant enablement Razorpay grants on request,
standard PCI-related gating). This client therefore implements every
endpoint that is live-verified working; the methods that would create a
payment from scratch are not included here because they cannot be
exercised for real without that enablement or a one-time human-driven
test checkout to mint a real payment_id.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

BASE_URL = "https://api.razorpay.com/v1"


class RazorpayError(RuntimeError):
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self.body = body
        description = ""
        if isinstance(body, dict):
            description = body.get("error", {}).get("description", "")
        super().__init__(f"Razorpay API error {status_code}: {description or body}")


@dataclass
class RazorpayClient:
    key_id: str
    key_secret: str
    transport: httpx.BaseTransport | None = None  # cassette injection point (see cassettes.py)
    timeout_s: float = 15.0

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=BASE_URL, auth=(self.key_id, self.key_secret),
            timeout=self.timeout_s, transport=self.transport,
        )

    @staticmethod
    def _handle(resp: httpx.Response) -> dict:
        data = resp.json()
        if resp.status_code >= 400:
            raise RazorpayError(resp.status_code, data)
        return data

    # -- Orders -----------------------------------------------------

    def create_order(self, *, amount_paise: int, receipt: str, notes: dict[str, str], currency: str = "INR") -> dict:
        """`notes` is where the mandate chain hash binding lives --
        intent_hash / cart_hash / payment_mandate_id go here so the
        Razorpay ledger record is cryptographically tied to what was
        actually authorized (see buyer/execute.py)."""
        body = {"amount": amount_paise, "currency": currency, "receipt": receipt, "notes": notes}
        with self._client() as client:
            resp = client.post("/orders", json=body)
        return self._handle(resp)

    def fetch_order(self, order_id: str) -> dict:
        with self._client() as client:
            resp = client.get(f"/orders/{order_id}")
        return self._handle(resp)

    def fetch_order_payments(self, order_id: str) -> dict:
        with self._client() as client:
            resp = client.get(f"/orders/{order_id}/payments")
        return self._handle(resp)

    # -- Payments -----------------------------------------------------

    def fetch_payment(self, payment_id: str) -> dict:
        with self._client() as client:
            resp = client.get(f"/payments/{payment_id}")
        return self._handle(resp)

    def capture_payment(self, payment_id: str, *, amount_paise: int, currency: str = "INR") -> dict:
        with self._client() as client:
            resp = client.post(f"/payments/{payment_id}/capture", json={"amount": amount_paise, "currency": currency})
        return self._handle(resp)

    def create_refund(self, payment_id: str, *, amount_paise: int | None = None, notes: dict[str, str] | None = None) -> dict:
        body: dict = {}
        if amount_paise is not None:
            body["amount"] = amount_paise
        if notes:
            body["notes"] = notes
        with self._client() as client:
            resp = client.post(f"/payments/{payment_id}/refund", json=body)
        return self._handle(resp)

    # -- Settlement recon (live mode only) -------------------------------

    def fetch_settlement_recon(self, *, year: int, month: int, day: int | None = None) -> dict:
        params: dict[str, str] = {"year": str(year), "month": f"{month:02d}"}
        if day is not None:
            params["day"] = f"{day:02d}"
        with self._client() as client:
            resp = client.get("/settlements/recon/combined", params=params)
        return self._handle(resp)


def client_from_env() -> RazorpayClient:
    return RazorpayClient(key_id=os.environ["RAZORPAY_KEY_ID"], key_secret=os.environ["RAZORPAY_KEY_SECRET"])
