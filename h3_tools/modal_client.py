"""HTTP client for the modal_inferencito /generate API, with account rotation.

Stdlib at import time; `requests` is imported lazily inside generate() so the
module stays testable with injected doubles (same pattern as enhancer.py).
Contract: modal_inferencito/API.md. Account-level failures (unreachable,
suspended) rotate to the next account; workflow-level failures raise
immediately — retrying them elsewhere would burn credits on the same bug.
"""

import json
import logging
import os
import time

GPU_RATES_USD_H = {"RTX-PRO-6000": 3.03, "L40S": 1.95, "A100-80GB": 2.50,
                   "H100": 3.95, "H200": 4.54, "B200": 6.25}


class ModalApiError(RuntimeError):
    """All accounts failed (or config is unusable); message shown as-is."""


class WorkflowFailed(ModalApiError):
    """The server-side ComfyUI graph failed — reproducible, never rotated."""


def load_config(accounts_json="", api_url="", api_key=""):
    path = (accounts_json or "").strip() or os.environ.get(
        "H3_MODAL_ACCOUNTS_JSON", "")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise ModalApiError("could not read accounts json %r: %s"
                                % (path, e)) from e
        accounts = [{"name": a.get("name", "?"), "url": a["url"]}
                    for a in data.get("accounts", [])
                    if a.get("url") and a.get("enabled", True)]
        if not accounts:
            raise ModalApiError("accounts json %r has no enabled account with "
                                "a url (run deploy_all.py first)" % path)
        gpu = data.get("gpu", "RTX-PRO-6000")
        return {"api_key": data.get("api_key", ""), "accounts": accounts,
                "gpu": gpu, "rate_usd_h": GPU_RATES_USD_H.get(gpu, 3.03)}
    if api_url.strip() and api_key.strip():
        return {"api_key": api_key.strip(),
                "accounts": [{"name": "api_url", "url": api_url.strip().rstrip("/")}],
                "gpu": "RTX-PRO-6000", "rate_usd_h": 3.03}
    raise ModalApiError(
        "no API configured: set the accounts_json widget (path to "
        "modal_inferencito/accounts.json), or the H3_MODAL_ACCOUNTS_JSON "
        "environment variable, or fill api_url + api_key")


def generate(cfg, body, files, *, timeout_s, account="auto", progress=None,
             interrupt=None, post=None, get=None, sleep=None, clock=None):
    if post is None or get is None:
        import requests
        post = post or requests.post
        get = get or requests.get
    sleep = sleep or time.sleep
    clock = clock or time.monotonic

    accounts = cfg["accounts"]
    if account != "auto":
        chosen = [a for a in accounts if a["name"] == account]
        if not chosen:
            raise ModalApiError("account %r not found (have: %s)" % (
                account, ", ".join(a["name"] for a in accounts)))
        accounts = chosen
    headers = {"X-API-Key": cfg["api_key"]}
    payload = dict(body, files=files)

    errors = []
    for acc in accounts:
        try:
            r = get(acc["url"] + "/health", headers=headers, timeout=20)
            if getattr(r, "status_code", 0) != 200:
                raise OSError("health HTTP %s" % getattr(r, "status_code", "?"))
        except Exception as e:
            errors.append("%s: health failed (%s)" % (acc["name"], e))
            continue
        try:
            r = post(acc["url"] + "/generate", json=payload, headers=headers,
                     timeout=300)
        except Exception as e:
            errors.append("%s: submit failed (%s)" % (acc["name"], e))
            continue
        status = getattr(r, "status_code", 0)
        if status == 400:
            # invalid request body: our bug or the user's — same everywhere
            raise WorkflowFailed("generate rejected: %s"
                                 % r.json().get("error", "HTTP 400"))
        if status != 200:
            errors.append("%s: submit HTTP %s" % (acc["name"], status))
            continue
        call_id = r.json()["call_id"]
        logging.info("h3_tools: api job %s submitted to %s", call_id,
                     acc["name"])

        started = clock()
        while True:
            if interrupt is not None:
                interrupt()
            if clock() - started > timeout_s:
                raise ModalApiError("timed out after %ds waiting for %s on %s"
                                    % (timeout_s, call_id, acc["name"]))
            try:
                s = get(acc["url"] + "/status/" + call_id, headers=headers,
                        timeout=60)
            except Exception as e:
                logging.warning("h3_tools: poll error (%s); retrying", e)
                sleep(5.0)
                continue
            if getattr(s, "status_code", 0) == 202:
                if progress is not None:
                    try:
                        progress(s.json().get("progress") or {})
                    except Exception:
                        logging.exception("h3_tools: progress callback failed")
                sleep(3.0)
                continue
            data = s.json()
            if data.get("status") == "failed":
                raise WorkflowFailed("workflow failed on %s: %s"
                                     % (acc["name"], data.get("error", "?")))
            result = data["result"]
            duration = float(result.get("duration_s") or (clock() - started))
            return result, {"account": acc["name"], "duration_s": duration}
    raise ModalApiError("all accounts failed:\n  " + "\n  ".join(
        errors or ["(no accounts configured)"]))
