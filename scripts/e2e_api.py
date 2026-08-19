#!/usr/bin/env python3
"""E2E of api_payload + modal_client against the real deployed API (no ComfyUI).

Usage (WSL):
  python3 scripts/e2e_api.py --dry-run     # validates the 4 bodies, free
  python3 scripts/e2e_api.py               # one real flf fast 3s job (~$0.03)
"""

import argparse
import base64
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h3_tools import api_payload, modal_client  # noqa: E402

ACCOUNTS = "/home/adolfocesar/projects/modal_inferencito/accounts.json"
ASSETS = Path("/home/adolfocesar/projects/modal_inferencito/scripts/test_assets")


def b64(p):
    return base64.b64encode(p.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "e2e_api_flf.mp4"))
    args = ap.parse_args()
    cfg = modal_client.load_config(accounts_json=ACCOUNTS)

    bodies = {
        "ref": (api_payload.ref_body(
            prompt="a @garota acena", duration_s=3,
            references=[{"name": "garota", "type": "image", "file": "ref_image.png"}]),
            [{"name": "ref_image.png", "b64": b64(ASSETS / "ref_image.png")}]),
        "ref_extend": (dict(api_payload.ref_body(
            prompt="a @garota acena", duration_s=6, duration_mode="total",
            references=[{"name": "garota", "type": "image", "file": "ref_image.png"}]),
            **api_payload.extend_extra(2.0, "source_video.mp4")),
            [{"name": "ref_image.png", "b64": b64(ASSETS / "ref_image.png")},
             {"name": "source_video.mp4", "b64": b64(ASSETS / "source_video.mp4")}]),
        "flf": (api_payload.flf_body(
            prompt="ela sorri para a camera", duration_s=3,
            first_name="first_frame.jpg"),
            [{"name": "first_frame.jpg", "b64": b64(ASSETS / "first_frame.jpg")}]),
        "flf_extend": (dict(api_payload.flf_body(
            prompt="a cena continua", duration_s=6),
            **api_payload.extend_extra(2.0, "source_video.mp4")),
            [{"name": "source_video.mp4", "b64": b64(ASSETS / "source_video.mp4")}]),
    }
    bodies["flf_extend"][0]["mode"] = "flf_extend"

    if args.dry_run:
        import requests
        acc = cfg["accounts"][0]
        for mode, (body, files) in bodies.items():
            r = requests.post(acc["url"] + "/generate",
                              json=dict(body, files=files, dry_run=True,
                                        quality="fast", seed=1),
                              headers={"X-API-Key": cfg["api_key"]}, timeout=120)
            ok = r.status_code == 200 and "workflow" in r.json()
            print(f"{mode}: {'ok' if ok else 'FAIL ' + r.text[:200]}", flush=True)
            if not ok:
                sys.exit(1)
        return

    body, files = bodies["flf"]
    t0 = time.time()
    result, meta = modal_client.generate(
        cfg, dict(body, quality="fast", seed=7), files, timeout_s=1800,
        progress=lambda p: print("  progress", p.get("value"), "/",
                                 p.get("max"), flush=True))
    out = Path(args.out)
    out.write_bytes(base64.b64decode(result["outputs"][0]["b64"]))
    print(f"ok: {out} | account {meta['account']} | gpu {meta['duration_s']}s "
          f"| wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
