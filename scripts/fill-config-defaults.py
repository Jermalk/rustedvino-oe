#!/usr/bin/env python3
"""Write every RustedVINO setting into a config.json, explicitly.

A setting that is not in the file silently takes a built-in default - and a
setting you cannot see is one you forget you have. This adds each missing global
setting and each missing per-model setting to the file(s) you name, using the
defaults the server itself reports (`rustedvino --print-default-config` and
`--print-default-model`), so the values written are exactly the ones already in
effect: running it changes nothing about how the server behaves.

  * values already in the file are never changed;
  * keys the server does not know are kept;
  * the file is rewritten in a fixed order (globals as documented in
    CONFIG.md, `models` last, each model: vram_gb, kind, then policy);
  * `--check` writes nothing and exits 1 if any setting is missing (for CI);
  * `--diff` writes nothing and lists, per file, every setting whose value differs from the
    CURRENT default - which includes settings you changed on purpose, so read it as "what
    would behave differently if I deleted this line", e.g. after an upgrade that changed a
    default (a filled file keeps the old value; `--check` cannot see that).

--binary must be the SAME version you run (default: target/release/rustedvino). A binary
that predates `--print-default-config` is refused without being started: an older server
ignores unknown flags and would boot a full server instead.

Usage: fill-config-defaults.py [--binary PATH] [--check | --diff] [--no-models] CONFIG...
"""
import argparse
import json
import os
import subprocess
import sys


FLAG_MARKER = b"RUSTEDVINO-TEMPLATE-FLAGS-V1-OK"  # embedded by src/main.rs


def knows_template_flags(binary):
    """True if `binary` (or, for the bundle launcher script, its sibling
    `rustedvino-bin`) contains the flag text. Never executes an unknown binary."""
    candidates = [binary, os.path.join(os.path.dirname(os.path.abspath(binary)), "rustedvino-bin")]
    for path in candidates:
        try:
            with open(path, "rb") as f:
                if FLAG_MARKER in f.read():
                    return True
        except OSError:
            continue
    return False


def template(binary, flag):
    proc = subprocess.Popen([binary, flag], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()  # SIGTERM only: a server must be allowed to shut down cleanly
        proc.communicate()
        sys.exit(f"{binary} {flag} did not finish in 30 s - is it the right binary?")
    if proc.returncode != 0:
        sys.exit(f"{binary} {flag} failed ({proc.returncode}): {err.strip()[:200]}")
    return json.loads(out)


def no_duplicates(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            sys.exit(f"duplicate key {key!r} in a config file: the server would reject it; fix it by hand first")
        seen[key] = value
    return seen


def changed_from_default(config, global_tpl, model_tpl):
    """Settings whose value differs from the template default (placeholders excluded)."""
    skip = {"models_dir", "device", "total_vram_gb", "models"}
    out = []
    for k, d in global_tpl.items():
        if k in skip or k not in config:
            continue
        if isinstance(d, dict) and isinstance(config[k], dict):
            # A nested block: list the keys that differ, not a block that only lacks new ones.
            out += [f"{k}.{n}" for n, nd in d.items() if n in config[k] and config[k][n] != nd]
        elif config[k] != d:
            out.append(k)
    for mid, stanza in config.get("models", {}).items():
        out += [f"models.{mid}.{k}" for k, d in model_tpl.items() if k not in ("vram_gb", "kind") and k in stanza and stanza[k] != d]
    return out


def fill_stanza(stanza, model_tpl, missing, where):
    out = {}
    for key, default in model_tpl.items():
        if key in stanza:
            out[key] = stanza[key]
        else:
            out[key] = default
            missing.append(f"{where}.{key}")
    for key, value in stanza.items():  # keys this build does not template: keep
        out.setdefault(key, value)
    return out


def fill(config, global_tpl, model_tpl, with_models):
    missing = []
    out = {}
    for key, default in global_tpl.items():
        if key == "models":
            continue
        if key in config and isinstance(default, dict) and isinstance(config[key], dict):
            # A nested block (`supervisor`) is merged key by key, so a key a newer build added
            # inside it is reported as missing too.
            out[key] = fill_stanza(config[key], default, missing, key)
        elif key in config:
            out[key] = config[key]
        else:
            out[key] = default
            missing.append(key)
    for key, value in config.items():
        if key not in out and key != "models":
            out[key] = value
    models = config.get("models", {})
    if with_models:
        models = {
            mid: fill_stanza(stanza, model_tpl, missing, f"models.{mid}")
            for mid, stanza in models.items()
        }
    out["models"] = models
    return out, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--binary", default="target/release/rustedvino")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--diff", action="store_true")
    ap.add_argument("--no-models", action="store_true", help="only the global settings")
    args = ap.parse_args()

    if not knows_template_flags(args.binary):
        sys.exit(
            f"{args.binary} does not know --print-default-config (older than 0.8.0, or not a "
            "RustedVINO binary); refusing to start it. Pass --binary for the build you run."
        )
    global_tpl = template(args.binary, "--print-default-config")
    model_tpl = template(args.binary, "--print-default-model")
    incomplete = False
    for path in args.configs:
        with open(path, encoding="utf-8-sig") as f:
            config = json.load(f, object_pairs_hook=no_duplicates)
        filled, missing = fill(config, global_tpl, model_tpl, not args.no_models)
        if args.diff:
            changed = changed_from_default(config, global_tpl, model_tpl)
            print(f"{path}: {len(changed)} setting(s) differ from the current default" + (f", e.g. {changed[:4]}" if changed else ""))
            continue
        if args.check:
            if missing:
                incomplete = True
                print(f"{path}: {len(missing)} setting(s) not written out, e.g. {missing[:3]}")
            continue
        with open(path, "w", encoding="utf-8") as f:
            json.dump(filled, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"{path}: wrote {len(missing)} setting(s) explicitly")
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
