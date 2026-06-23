"""Breeze session management (IMPLEMENTATION.md sec.3) -- WRITE OURSELVES.

Breeze's session token is interactive and DAILY (expires 24h/midnight). Flow:
  1. open login_url(api_key) in a browser, log in with ICICIdirect creds
  2. copy the `API_Session` value from the redirect (DevTools -> Network -> Form Data)
  3. make_breeze(api_key, api_secret, api_session) -> a ready BreezeConnect
Only API *calls* must egress from the whitelisted IP; the browser login can be from anywhere.
"""
from __future__ import annotations

import os
import urllib.parse

from breeze_connect import BreezeConnect

LOGIN_BASE = "https://api.icicidirect.com/apiuser/login"


def login_url(api_key: str) -> str:
    """Browser login URL; after login the redirect carries the daily API_Session token."""
    return f"{LOGIN_BASE}?api_key={urllib.parse.quote_plus(api_key)}"


def make_breeze(api_key: str, api_secret: str, api_session: str) -> BreezeConnect:
    """Return a session-generated BreezeConnect. `api_session` = the daily API_Session token."""
    bz = BreezeConnect(api_key=api_key)
    bz.generate_session(api_secret=api_secret, session_token=api_session)
    return bz


def make_breeze_from_env() -> BreezeConnect:
    """Build from BREEZE_API_KEY / BREEZE_API_SECRET / BREEZE_SESSION_TOKEN (.env)."""
    return make_breeze(
        os.environ["BREEZE_API_KEY"],
        os.environ["BREEZE_API_SECRET"],
        os.environ["BREEZE_SESSION_TOKEN"],
    )
