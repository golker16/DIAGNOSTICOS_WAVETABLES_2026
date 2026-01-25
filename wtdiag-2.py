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
    # Compatible con PyInstaller (sys._MEIPASS) y con ejecución normal
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel_name


def _load_core():
    core_path = _resource_path("wtdiag-1.py")
    if not core_path.exists():
        # fallback por si ejecutan desde repo y no está en _MEIPASS
        core_path = Path(__file__).resolve().parent / "wtdiag-1.py"

    spec = importlib.util.spec_from_file_location("wtdiag_core", str(core_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"No pude crear spec para {core_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


core = _load_core()

# Re-export útil (por si alguien quiere importar desde este "frente")
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

    db_entries: List[Dict[str, Any]] = []
    for i, p in enumerate(wavs, 1):
        try:
            desc = core.process_wavetable(p, args.table_size, args.harmonics, args.bands)
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
            print(f"[{i:5d}/{len(wavs):5d}] OK  {p.name}  -> {out_json.name}")
        except Exception as e:
            print(f"[{i:5d}/{len(wavs):5d}] FAIL {p.name}  ({e})")

    index_path = out_dir / "_INDEX.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"root": str(in_dir.as_posix()), "entries": db_entries}, f, ensure_ascii=False, indent=2)

    print(f"\n[index] Listo. Index global: {index_path}")
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

def cmd_match(args) -> int:
    db_dir = Path(args.db_dir)
    index_path = db_dir / "_INDEX.json"
    if not index_path.exists():
        print("[match] No existe _INDEX.json. Primero corre index.")
        return 2

    target_path = Path(args.target)
    if not target_path.exists():
        print(f"[match] Target no existe: {target_path}")
        return 2

    tx, _tsr = core.safe_read_wav(target_path)
    tx = core.standardize_audio(tx)

    if len(tx) >= args.table_size:
        start = (len(tx) - args.table_size) // 2
        tframe = tx[start:start + args.table_size]
    else:
        tframe = tx

    tframe = core.standardize_frame(tframe, args.table_size)
    tfeat = core.compute_features_for_frame(tframe, args.harmonics, args.bands)

    th = np.array(tfeat.harmonics, dtype=np.float32)
    tenv = np.array(tfeat.spectral_env_db, dtype=np.float32)

    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    entries = idx.get("entries", [])

    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for it in entries:
        try:
            dpath = db_dir / it["descriptor"]
            desc = core.load_descriptor(dpath)

            if desc.get("type") == "sample" and not args.include_samples:
                continue

            ch_list, _cenv_list, _c_rms = core.get_desc_features(desc)
            ch = np.array(ch_list, dtype=np.float32)

            dist = core.cosine_distance(th, ch)
            scored.append((dist, it, desc))
        except Exception:
            continue

    if not scored:
        print("[match] Base vacía o no compatible.")
        return 3

    scored.sort(key=lambda x: x[0])
    top = scored[:args.topk]

    results: List[Dict[str, Any]] = []
    for rank, (hdist, it, desc) in enumerate(top, 1):
        _ch_list, cenv_list, c_rms = core.get_desc_features(desc)
        cenv = np.array(cenv_list, dtype=np.float32)

        delta = tenv - cenv
        delta = core.smooth_and_limit_delta(delta, limit_db=args.eq_limit_db, smooth_sigma=args.eq_smooth)

        peq = core.delta_to_peq(delta, sr=core.FEATURE_SR, max_filters=6, min_sep_bands=6)

        t_rms = float(tfeat.rms_dbfs)
        gain_db = float(t_rms - float(c_rms))

        eq_cost = float(np.mean(np.abs(delta)) / max(1e-6, args.eq_limit_db))
        score = float(hdist * 0.75 + eq_cost * 0.25)

        results.append({
            "rank": rank,
            "wavetable_path": desc.get("path", it.get("path")),
            "descriptor": it["descriptor"],
            "score": score,
            "harmonic_cosine_dist": float(hdist),
            "gain_db_suggested": gain_db,
            "delta_env_db": delta.tolist(),
            "peq_filters": peq,
            "diagnosis": desc.get("diagnosis", {})
        })

    suggestions: List[str] = []
    if tfeat.noise_ratio_db > -12 or tfeat.tonalness < 0.6:
        suggestions.append("Target tiene bastante componente ruidosa: considera capa de noise/attack residual.")
    if tfeat.brightness > 0.7:
        suggestions.append("Target muy brillante: limita stacks o aplica lowpass suave en capas auxiliares.")

    report: Dict[str, Any] = {
        "target": str(target_path.as_posix()),
        "target_features": {
            "tonalness": tfeat.tonalness,
            "noise_ratio_db": tfeat.noise_ratio_db,
            "brightness": tfeat.brightness,
            "odd_even_ratio": tfeat.odd_even_ratio,
            "rms_dbfs": tfeat.rms_dbfs,
            "crest_db": tfeat.crest_db
        },
        "search_params": {
            "table_size": args.table_size,
            "harmonics": args.harmonics,
            "bands": args.bands,
            "topk": args.topk,
            "eq_limit_db": args.eq_limit_db,
            "eq_smooth": args.eq_smooth,
            "feature_sr": core.FEATURE_SR
        },
        "suggestions": suggestions,
        "matches": results
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # TXT humano (igual que antes)
    txt_path = out_path.with_suffix(".txt")
    lines: List[str] = []
    lines.append("WAVETABLE MATCH REPORT")
    lines.append(f"Target: {target_path}")
    lines.append(f"Target tonalness={tfeat.tonalness:.3f} noise_ratio_db={tfeat.noise_ratio_db:.2f} bright={tfeat.brightness:.3f}")
    if suggestions:
        lines.append("Suggestions:")
        for s in suggestions:
            lines.append(f" - {s}")
    lines.append("")
    for r in results:
        lines.append(f"#{r['rank']} score={r['score']:.4f} harmDist={r['harmonic_cosine_dist']:.4f} gain={r['gain_db_suggested']:+.2f} dB")
        lines.append(f"   WT: {r['wavetable_path']}")
        d = r.get("diagnosis") or {}
        if d:
            lines.append(f"   family={d.get('family')} best_for={d.get('best_for')} pitch_friendly={d.get('pitch_friendly')}")
            lines.append(f"   notes: {d.get('notes')}")
        lines.append(f"   delta_env_db: meanAbs={float(np.mean(np.abs(r['delta_env_db']))):.2f} dB (limit {args.eq_limit_db} dB)")
        peq = r.get("peq_filters", [])
        if peq:
            lines.append("   PEQ approx:")
            for flt in peq:
                lines.append(f"     - peaking f0={flt['f0_hz']:.1f}Hz gain={flt['gain_db']:+.2f}dB Q={flt['q']:.2f}")
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[match] OK -> {out_path}")
    print(f"[match] TXT -> {txt_path}")
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
    p_match.add_argument("--eq-limit-db", type=float, default=6.0)
    p_match.add_argument("--eq-smooth", type=float, default=1.5)
    p_match.add_argument("--include-samples", action="store_true")
    p_match.set_defaults(func=cmd_match)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
