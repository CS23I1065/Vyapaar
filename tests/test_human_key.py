"""
Tests for buyer/human_key.py -- the human signing key's lifecycle,
mirroring the same persistent-key pattern as the merchant side
(merchant_kit.cli's --key-file).
"""

from __future__ import annotations

from buyer.human_key import load_or_create_human_keypair


def test_first_run_generates_and_saves(tmp_path):
    path = tmp_path / "human-key.json"
    keypair, created = load_or_create_human_keypair(path)
    assert created is True
    assert path.exists()
    assert keypair.kid


def test_later_runs_load_the_same_identity(tmp_path):
    """principal_kid must never silently change underneath a user --
    every IntentMandate they ever signed refers back to it."""
    path = tmp_path / "human-key.json"
    first, _ = load_or_create_human_keypair(path)
    second, created_second = load_or_create_human_keypair(path)
    assert created_second is False
    assert first.kid == second.kid


def test_key_file_is_written_private(tmp_path):
    path = tmp_path / "human-key.json"
    load_or_create_human_keypair(path)
    assert (path.stat().st_mode & 0o777) == 0o600
