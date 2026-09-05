"""
The human's signing key -- the root of the user's authority over every
purchase. Outside tests there was no code anywhere in this repo that
created it, stored it, loaded it, or protected it: no answer to where it
lives, what happens if it's lost, or what an attacker with the file can
do. That last one is total compromise -- anyone holding this file can
sign any intent they like, for any amount, and every gate in
buyer/policy_engine.py will honour it, because a valid signature from
principal_kid is exactly what G0.2 checks for.

This mirrors merchant_kit.cli.load_or_create_keypair deliberately, not
by coincidence: both are the same problem (a keypair that must have a
stable identity across runs and must not be silently regenerated), and
solving it twice with two different shapes would be its own kind of bug.

Scope, stated plainly rather than implied: a file on disk, 0600, is the
right scope for a hackathon build. It is NOT what production should use
-- a hardware-backed key (a YubiKey, a TPM) or an OS keystore (Keychain,
Windows Credential Manager, a Linux kernel keyring) is the real answer,
because a file can be copied by anything that can read the filesystem.
Naming this limitation is better than letting the Ed25519 signatures
elsewhere in this repo imply an assurance the key STORAGE does not
actually provide.
"""

from __future__ import annotations

from pathlib import Path

from mandates.keys import KeyPair, generate_keypair, load_keypair

DEFAULT_HUMAN_KEY_PATH = Path.home() / ".hope" / "human-key.json"


def load_or_create_human_keypair(path: Path = DEFAULT_HUMAN_KEY_PATH) -> tuple[KeyPair, bool]:
    """Returns (keypair, created). First run generates and saves a new
    key at 0600 (KeyPair.save's existing behaviour); every later run
    loads the same one, so principal_kid -- the thing every signed
    IntentMandate in this user's history refers back to -- never
    silently changes underneath them."""
    if path.exists():
        return load_keypair(path), False
    keypair = generate_keypair()
    keypair.save(path)
    return keypair, True
