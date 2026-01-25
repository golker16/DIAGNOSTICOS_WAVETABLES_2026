# wtdiag-2.py
# CLI + comandos (cmd_index/cmd_diag/cmd_match).
# Carga el core desde wtdiag-1.py vía importlib (porque el archivo tiene guion).

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import importlib.util

import numpy as np


def _resource_path(rel_name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel_name


def _load_core():
    core_path = _resource_path("wtdiag-1.py")
    if not core_path.exists():
        core_path = Path(__file__).resolve().parent / "wtdiag-1.py"

    spec = importlib.util.spec_from_file_location("wtdiag_core", str(core_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude crear spec para {core_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


core = _load_core()
FEATURE_SR = core.FEATURE_SR


# ----------------------------
# Indexado
# ----------------------------

def cmd_index(args) -> int:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted([p for p in in_dir.rglob("*.wav")])
    if not wavs:
        print(f"[index] No se encontraron .wav en: {in_dir}")
        return 2

    include_samples = bool(getattr(args, "include_samples", False))

    db_entries: List[Dict[str, Any]] = []
    kept = 0
    skipped_samples = 0

    for i, p in enumerate(wavs, 1):
        try:
            desc = core.process_wavetable(p, args.table_size, args.harmonics, args.bands)

            # NUEVO: evitar “contaminar” la DB con archivos no-wavetable
            # type="sample" (antes era unknown)
            if desc.type == "sample" and not include_samples:
                skipped_samples += 1
                continue

            rel = p.relative_to(in_dir).as_posix()
            safe = rel.replace("/", "__").replace("\\", "__")
            out_json = out_dir / (Path(safe).with_suffix(".json").name)

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(core.descriptor_to_spec(desc), f, ensure_ascii=False, indent=2)

            db_entries.append({
                "path": desc.path,
                "descriptor": out_json.name,
                "type": desc.type,
                "family": desc.diagnosis.family,
                "best_for": desc.diagnosis.best_for
            })
            kept += 1

            if i % 50 == 0:
                print(f"[index] {i}/{len(wavs)} ... (kept={kept}, skipped_samples={skipped_samples})")
        except Exception as e:
            print(f"[index] ERROR en {p}: {e}")

    index = {
        "tableSize": int(args.table_size),
        "harmonics": int(args.harmonics),
        "bands": int(args.bands),
        "count": len(db_entries),
        "skipped_samples": int(skipped_samples),
        "include_samples": bool(include_samples),
        "entries": db_entries
    }

    with open(out_dir / "_INDEX.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    msg = f"[index] OK. Descriptores: {len(db_entries)} en {out_dir}"
    if skipped_samples and not include_samples:
        msg += f" (skipped_samples={skipped_samples})"
    print(msg)
    return 0


# ----------------------------
# Diagnóstico puntual
# ----------------------------

def cmd_diag(args) -> int:
    p = Path(args.wav)
    if not p.exists():
        print(f"[diag] No existe: {p}")
        return 2

    desc = core.process_wavetable(p, args.table_size, args.harmonics, args.bands)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(core.descriptor_to_spec(desc), f, ensure_ascii=False, indent=2)

    print(f"[diag] OK -> {out}")
    return 0


# ----------------------------
# Matching
# ----------------------------

def _get_arg(args, name: str, default):
    return getattr(args, name, default)


def cmd_match(args) -> int:
    db_dir = Path(_get_arg(args, "db_dir", _get_arg(args, "db", "")))
    index_path = db_dir / "_INDEX.json"
    if not index_path.exists():
        print("[match] No existe _INDEX.json. Primero corre index.")
        return 2

    target_path = Path(_get_arg(args, "target", ""))
    if not target_path.exists():
        print(f"[match] Target no existe: {target_path}")
        return 2

    out_path = Path(_get_arg(args, "out", "match_report.json"))

    tx, _tsr = core.safe_read_wav(target_path)
    tx = core.standardize_audio(tx)

    # Tramo estable (evita el centro a ciegas)
    tseg = core.select_stable_segment(tx, int(args.table_size))
    tframe = core.standardize_frame(tseg, int(args.table_size), loop_fix=True)
    tfeat = core.compute_features_for_frame(tframe, int(args.harmonics), int(args.bands))

    th = np.array(tfeat.harmonics, dtype=np.float32)
    tenv = np.array(tfeat.spectral_env_db, dtype=np.float32)

    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    entries = idx.get("entries", [])
    if not entries:
        print("[match] Índice vacío.")
        return 2

    topk = int(_get_arg(args, "topk", 50))
    topn = int(_get_arg(args, "topn", 10))
    include_samples = bool(_get_arg(args, "include_samples", False))
    gain_mode = str(_get_arg(args, "gain_mode", "rms")).lower()
    gain_mode = "lufs" if gain_mode == "lufs" else "rms"

    # 1) Top-K por armónicos (ADN)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ent in entries:
        dpath = db_dir / ent["descriptor"]
        try:
            desc = core.load_descriptor(dpath)
        except Exception:
            continue

        # NUEVO: el tipo “no wavetable” ahora es sample
        if desc.get("type") == "sample" and not include_samples:
            continue

        try:
            h, _e, _r, _l = core.get_desc_features(desc)
        except Exception:
            continue

        ch = np.array(h, dtype=np.float32)
        scored.append((core.cosine_distance(th, ch), desc))

    if not scored:
        print("[match] No hay candidatos tras filtros.")
        return 2

    scored.sort(key=lambda t: t[0])
    top = scored[: max(1, topk)]

    # 2) Refinar por envelope + penalización EQ
    eq_limit_db = float(_get_arg(args, "eq_limit_db", 6.0))
    eq_smooth = float(_get_arg(args, "eq_smooth", 1.5))
    max_filters = int(_get_arg(args, "max_filters", 6))
    min_sep_bands = int(_get_arg(args, "min_sep_bands", 6))
    w_harm = float(_get_arg(args, "w_harm", 1.0))
    w_eq = float(_get_arg(args, "w_eq", 0.8))

    results: List[Dict[str, Any]] = []
    for harm_err, desc in top:
        _h, e, r, l = core.get_desc_features(desc)
        cand_env = np.array(e, dtype=np.float32)

        delta = tenv - cand_env
        delta = core.smooth_and_limit_delta(delta, limit_db=eq_limit_db, smooth_sigma=eq_smooth)

        peq = core.delta_to_peq(delta, sr=core.FEATURE_SR, max_filters=max_filters, min_sep_bands=min_sep_bands)

        eq_cost = float(np.mean(np.abs(delta))) / max(1e-6, eq_limit_db)

        cand_gain = float(l if gain_mode == "lufs" else r)
        tgt_gain = float(tfeat.lufs if gain_mode == "lufs" else tfeat.rms_dbfs)
        gain_delta_db = tgt_gain - cand_gain

        score = float(harm_err * w_harm + eq_cost * w_eq)

        results.append({
            "score": score,
            "harmonic_error": float(harm_err),
            "eq_cost": eq_cost,
            "path": desc.get("path"),
            "type": desc.get("type"),
            "family": desc.get("diagnosis", {}).get("family"),
            "best_for": desc.get("diagnosis", {}).get("best_for", []),
            "gain_match": {
                "mode": "lufs" if gain_mode == "lufs" else "rms_dbfs",
                "target": tgt_gain,
                "candidate": cand_gain,
                "gain_delta_db": gain_delta_db
            },
            "delta_env_db": delta.tolist(),
            "peq_filters": peq,
            "diagnosis": desc.get("diagnosis", {})
        })

    results.sort(key=lambda r: r["score"])
    results = results[: max(1, topn)]

    suggestions: List[str] = []
    if tfeat.noise_ratio_db > -12 or tfeat.tonalness < 0.6:
        suggestions.append("Target tiene bastante componente ruidosa: considera capa de noise/attack residual.")
    if tfeat.brightness > 0.7:
        suggestions.append("Target muy brillante: limita stacks o aplica lowpass suave en capas auxiliares.")

    report: Dict[str, Any] = {
        "target": str(target_path.as_posix()),
        "target_selection": {"method": "stable_segment", "tableSize": int(args.table_size)},
        "target_features": {
            "tonalness": tfeat.tonalness,
            "noise_ratio_db": tfeat.noise_ratio_db,
            "brightness": tfeat.brightness,
            "odd_even_ratio": tfeat.odd_even_ratio,
            "rolloff": tfeat.rolloff,
            "crest_db": tfeat.crest_db,
            "rms_dbfs": tfeat.rms_dbfs,
            "lufs": tfeat.lufs,
            "feature_sr": core.FEATURE_SR
        },
        "matching": {
            "topk": topk,
            "topn": topn,
            "gain_mode": gain_mode,
            "w_harm": w_harm,
            "w_eq": w_eq,
            "eq_limit_db": eq_limit_db,
            "eq_smooth": eq_smooth,
            "max_filters": max_filters,
            "min_sep_bands": min_sep_bands,
            "include_samples": bool(include_samples),
        },
        "suggestions": suggestions,
        "matches": results
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # TXT legible
    txt_path = out_path.with_suffix(".txt")
    lines: List[str] = []
    lines.append("WAVETABLE MATCH REPORT")
    lines.append(f"Target: {report['target']}")
    lines.append(f"Gain mode: {'lufs' if gain_mode=='lufs' else 'rms_dbfs'}")
    lines.append("")

    if suggestions:
        lines.append("SUGERENCIAS:")
        for s in suggestions:
            lines.append(f" - {s}")
        lines.append("")

    for i, m in enumerate(results, 1):
        lines.append(f"#{i} score={m['score']:.4f}  harm={m['harmonic_error']:.4f}  eq={m['eq_cost']:.4f}")
        lines.append(f"   {m['path']}  [{m.get('type')}/{m.get('family')}] best_for={m.get('best_for')}")
        gm = m["gain_match"]
        lines.append(f"   gainMatch({gm['mode']}): target={gm['target']:.2f} cand={gm['candidate']:.2f} delta={gm['gain_delta_db']:.2f} dB")

        if m.get("peq_filters"):
            lines.append("   PEQ:")
            for flt in m["peq_filters"]:
                ftype = flt.get("type", "peaking")
                f0 = float(flt.get("f0_hz", 0.0))
                gain = float(flt.get("gain_db", 0.0))

                # Compatible con shelves (no siempre tienen Q)
                if ftype in ("low_shelf", "high_shelf"):
                    slope = float(flt.get("slope", 1.0))
                    lines.append(f"     - {ftype} f0={f0:.1f}Hz gain={gain:.2f}dB slope={slope:.2f}")
                else:
                    q = float(flt.get("q", 1.2))
                    lines.append(f"     - peaking f0={f0:.1f}Hz gain={gain:.2f}dB Q={q:.2f}")
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[match] OK -> {out_path} (+ {txt_path.name})")
    return 0


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(prog="wtdiag", description="Wavetable diagnoser + matcher (batch)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Indexa carpeta de wavetables y genera descriptores JSON + _INDEX.json")
    p_index.add_argument("--in", dest="in_dir", required=True)
    p_index.add_argument("--out", dest="out_dir", required=True)
    p_index.add_argument("--table-size", type=int, default=2048)
    p_index.add_argument("--harmonics", type=int, default=64)
    p_index.add_argument("--bands", type=int, default=128)

    # NUEVO: evitar contaminar DB (por defecto NO indexa type='sample')
    p_index.add_argument(
        "--include-samples",
        action="store_true",
        help="Incluye type='sample' (archivos no-wavetable)"
    )

    p_index.set_defaults(func=cmd_index)

    p_diag = sub.add_parser("diag", help="Diagnostica 1 wavetable y genera descriptor JSON")
    p_diag.add_argument("--wav", required=True)
    p_diag.add_argument("--out", required=True)
    p_diag.add_argument("--table-size", type=int, default=2048)
    p_diag.add_argument("--harmonics", type=int, default=64)
    p_diag.add_argument("--bands", type=int, default=128)
    p_diag.set_defaults(func=cmd_diag)

    p_match = sub.add_parser("match", help="Match de un target (layer) contra DB indexada")
    p_match.add_argument("--db", dest="db_dir", required=True)
    p_match.add_argument("--target", required=True)
    p_match.add_argument("--out", required=True)
    p_match.add_argument("--table-size", type=int, default=2048)
    p_match.add_argument("--harmonics", type=int, default=64)
    p_match.add_argument("--bands", type=int, default=128)
    p_match.add_argument("--topk", type=int, default=15)
    p_match.add_argument("--topn", type=int, default=10)
    p_match.add_argument("--eq-limit-db", type=float, default=6.0)
    p_match.add_argument("--eq-smooth", type=float, default=1.5)

    # Ajustado: ahora realmente es sample (no unknown)
    p_match.add_argument("--include-samples", action="store_true", help="Incluye type='sample'")

    p_match.add_argument("--gain-mode", choices=["rms", "lufs"], default="rms")
    p_match.add_argument("--max-filters", type=int, default=6)
    p_match.add_argument("--min-sep-bands", type=int, default=6)
    p_match.add_argument("--w-harm", type=float, default=1.0)
    p_match.add_argument("--w-eq", type=float, default=0.8)
    p_match.set_defaults(func=cmd_match)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

