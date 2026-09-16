#!/usr/bin/env bash
# rv-cargo.sh — run any cargo command with the OpenVINO build + runtime env wired in.
#
# Every LINKING cargo command (build, test, clippy --all-targets, build --release)
# needs the OV_*_DIR vars pointed at the real OpenVINO install, plus
# OPENVINO_TOKENIZERS_PATH_GENAI and a matching LD_LIBRARY_PATH at runtime —
# otherwise build.rs relinks against its fallback default and the binary dies at
# load with `libopenvino_genai.so.2620: cannot open shared object file`. This
# wrapper centralises that recipe so it is one command, not a multi-line export
# block.
#
# REQUIRES: OpenVINO GenAI 2026.2 installed via pip into some Python venv
# (`pip install openvino openvino-genai openvino-tokenizers`). Point this
# script at that venv's site-packages directory — auto-detected below, or set
# RV_OV_VENV / activate the venv first.
#
# Usage:
#   scripts/rv-cargo.sh test
#   scripts/rv-cargo.sh build --release
#   scripts/rv-cargo.sh clippy --all-targets -- -D warnings
set -euo pipefail

# Auto-detect OV site-packages if RV_OV_VENV is not set explicitly.
if [[ -z "${RV_OV_VENV:-}" ]]; then
    # 1. An active venv (`source .../bin/activate` or `$VIRTUAL_ENV` set).
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        for _py in "$VIRTUAL_ENV"/lib/python3.*/site-packages; do
            if [[ -d "$_py/openvino_genai" ]]; then
                RV_OV_VENV="$_py"
                break
            fi
        done
    fi
    # 2. Whatever `python3` resolves to right now (covers venvs activated via
    #    other means, e.g. direnv, or a system install with OV pip-installed).
    if [[ -z "${RV_OV_VENV:-}" ]] && command -v python3 >/dev/null 2>&1; then
        _py_site="$(python3 -c 'import openvino_genai, os; print(os.path.dirname(os.path.dirname(openvino_genai.__file__)))' 2>/dev/null || true)"
        if [[ -n "$_py_site" && -d "$_py_site/openvino_genai" ]]; then
            RV_OV_VENV="$_py_site"
        fi
    fi
    # 3. Common manual-venv naming conventions under $HOME, as a last resort.
    if [[ -z "${RV_OV_VENV:-}" ]]; then
        for _cand in "$HOME"/ov*/lib/python3.*/site-packages "$HOME"/.venv*/lib/python3.*/site-packages; do
            if [[ -d "$_cand/openvino_genai" ]]; then
                RV_OV_VENV="$_cand"
                break
            fi
        done
    fi
    if [[ -z "${RV_OV_VENV:-}" ]]; then
        printf 'error: cannot find an OpenVINO GenAI install.\n' >&2
        printf '  Activate the venv you installed openvino-genai into (source .../bin/activate),\n' >&2
        printf '  or set RV_OV_VENV to its site-packages dir containing openvino_genai/.\n' >&2
        exit 1
    fi
fi

VENV="$RV_OV_VENV"

export OV_GENAI_DIR="$VENV/openvino_genai"
export OV_DIR="$VENV/openvino/libs"
export OV_TOKENIZERS_DIR="$VENV/openvino_tokenizers/lib"
# Base OpenVINO headers — build.rs reads OV_INCLUDE_DIR (2026.2 headers live in
# the venv, not ~/.local). Override with OV_INCLUDE_DIR if they move.
export OV_INCLUDE_DIR="${OV_INCLUDE_DIR:-$VENV/openvino/include}"
export OPENVINO_TOKENIZERS_PATH_GENAI="$OV_TOKENIZERS_DIR/libopenvino_tokenizers.so"
export LD_LIBRARY_PATH="$OV_GENAI_DIR:$OV_DIR:$OV_TOKENIZERS_DIR:${LD_LIBRARY_PATH:-}"

exec cargo "$@"
