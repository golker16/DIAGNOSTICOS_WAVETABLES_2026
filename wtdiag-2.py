# wtdiag-2.py
# CLI + comandos (cmd_index/cmd_diag/cmd_match).
# Carga el core desde wtdiag-1.py vía importlib (porque el archivo tiene guion).

import argparse

# --- FIX: fallback si el build “rompe” stdlib json ---
try:
    import json  # stdlib
except ModuleNotFoundError:
    import simplejson as json  # fallback para builds rotos

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def _safe_descriptor_name(in_dir: Path, wav_path: Path) -> str:
    rel = wav_path.relative_to(in_dir).as_posix()
    safe = rel.replace("/", "__").replace("\\", "__")
    return Path(safe).with_suffix(".json").name


def _safe_export_base_name(in_dir: Path, wav_path: Path) -> str:
    """
    Nombre estable para export PRO basado en ruta relativa.
    Ej:
      in_dir/a/b/Cool.wav -> a__b__Cool
    """
    rel = wav_path.relative_to(in_dir).as_posix()
    safe = rel.replace("/", "__").replace("\\", "__")
    # sin sufijo .wav
    return str(Path(safe).with_suffix("").name)


def _index_one(
    wav_path: Path,
    in_dir: Path,
    out_dir: Path,
    table_size: int,
    harmonics: int,
    bands: int,
    include_samples: bool,
    *,
    pro: bool = False,
) -> Dict[str, Any]:
    """
    Worker para index en paralelo.
    Devuelve dict con:
      - ok: bool
      - skipped_sample: bool
      - out_json_name: str (si ok)
      - entry: dict (si ok)
      - error: str (si falla)
      - wav: str
    """
    try:
        source_path_str = str(wav_path.as_posix())
        export_meta: Optional[Dict[str, Any]] = None

        if pro:
            # ✅ PRO: exporta WAV canónico + descriptor grande EN LA MISMA CARPETA (out_dir)
            # y con el MISMO nombre base.

            base = _safe_export_base_name(in_dir, wav_path)
            base_can = f"{base}__CAN__2048x64"  # recomendado

            export_path = out_dir / f"{base_can}.wav"
            out_json_name = f"{base_can}.json"
            out_json = out_dir / out_json_name

            export_meta = core.export_canonical_wavetable(
                wav_path,
                export_path,
                table_size=core.PRO_TABLE_SIZE,
                num_frames=core.PRO_NUM_FRAMES,
                sr_out=core.PRO_SR,
                loop_fix=True,
                # ✅ sidecar apagado (lo quitamos del export)
                write_sidecar_json=False,
            )

            # Si el original era sample y no queremos incluir samples, lo saltamos.
            # (Importante: tras export, el archivo ya parece multi_frame 64, así que
            # desc.type ya no será "sample"; por eso usamos type_original del meta).
            if str(export_meta.get("type_original", "")) == "sample" and not include_samples:
                return {
                    "ok": False,
                    "skipped_sample": True,
                    "wav": str(wav_path),
                }

            # Diagnosticar el export (no el original), para que desc.path apunte al canónico.
            desc = core.process_wavetable(export_path, core.PRO_TABLE_SIZE, harmonics, bands)

            # Guardar descriptor grande al lado del wav canónico (mismo nombre base)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(core.descriptor_to_spec(desc), f, ensure_ascii=False, indent=2)

            entry: Dict[str, Any] = {
                "path": str(export_path.as_posix()),   # ✅ wav canónico (plugin-safe)
                "descriptor": out_json_name,           # ✅ json grande al lado
                "type": desc.type,
                "family": desc.diagnosis.family,
                "best_for": desc.diagnosis.best_for,
                "source_path": source_path_str,        # ✅ auditoría opcional
                "pro": True,
            }

            return {
                "ok": True,
                "skipped_sample": False,
                "out_json_name": out_json_name,
                "entry": entry,
                "wav": str(wav_path),
            }

        # -------------------------
        # NO PRO: comportamiento original
        # -------------------------
        desc = core.process_wavetable(wav_path, table_size, harmonics, bands)

        # Evitar “contaminar” la DB con archivos no-wavetable
        if desc.type == "sample" and not include_samples:
            return {
                "ok": False,
                "skipped_sample": True,
                "wav": str(wav_path),
            }

        out_name = _safe_descriptor_name(in_dir, wav_path)
        out_json = out_dir / out_name

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(core.descriptor_to_spec(desc), f, ensure_ascii=False, indent=2)

        entry = {
            "path": desc.path,
            "descriptor": out_name,
            "type": desc.type,
            "family": desc.diagnosis.family,
            "best_for": desc.diagnosis.best_for,
        }
        return {
            "ok": True,
            "skipped_sample": False,
            "out_json_name": out_name,
            "entry": entry,
            "wav": str(wav_path),
        }

    except Exception as e:
        return {
            "ok": False,
            "skipped_sample": False,
            "error": str(e),
            "wav": str(wav_path),
        }


def cmd_index(args) -> int:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted([p for p in in_dir.rglob("*.wav")])
    if not wavs:
        print(f"[index] No se encontraron .wav en: {in_dir}")
        return 2

    include_samples = bool(getattr(args, "include_samples", False))
    jobs = int(getattr(args, "jobs", 1) or 1)
    jobs = max(1, jobs)

    pro = bool(getattr(args, "pro", False))
    pro_wav_dir_arg = getattr(args, "pro_wav_dir", None)

    # ✅ Nota: ahora en PRO ya NO usamos out_dir/EXPORT/WAV.
    # Todo va directo a out_dir (un solo output folder).
    if pro_wav_dir_arg:
        # mantenemos el flag por compat, pero ya no se usa
        print(f"[index] NOTE: --pro-wav-dir ya no se usa (outputs PRO van directo a --out). Ignorando: {pro_wav_dir_arg}")

    # En modo PRO, forzamos el idioma único (2048x64).
    if pro:
        if int(getattr(args, "table_size", core.PRO_TABLE_SIZE)) != int(core.PRO_TABLE_SIZE):
            print(f"[index] --pro activo: forzando table_size={core.PRO_TABLE_SIZE} (ignorando {args.table_size})")
        table_size = int(core.PRO_TABLE_SIZE)
    else:
        table_size = int(args.table_size)

    processed = 0
    kept = 0
    skipped_samples = 0
    failed_files = 0

    db_entries: List[Dict[str, Any]] = []

    def _progress(i_done: int):
        if i_done % 50 == 0 or i_done == len(wavs):
            print(
                f"[index] {i_done}/{len(wavs)} ... "
                f"(kept={kept}, failed={failed_files}, skipped_samples={skipped_samples})"
            )

    if jobs == 1:
        for i, p in enumerate(wavs, 1):
            processed += 1
            res = _index_one(
                wav_path=p,
                in_dir=in_dir,
                out_dir=out_dir,
                table_size=table_size,
                harmonics=int(args.harmonics),
                bands=int(args.bands),
                include_samples=include_samples,
                pro=pro,
            )

            if res.get("skipped_sample"):
                skipped_samples += 1
            elif res.get("ok"):
                db_entries.append(res["entry"])
                kept += 1
            else:
                failed_files += 1
                print(f"[index] ERROR en {p}: {res.get('error', 'error desconocido')}")

            _progress(i)

    else:
        # Paralelo: el trabajo pesado está en numpy/scipy; threads ayudan.
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [
                ex.submit(
                    _index_one,
                    p, in_dir, out_dir,
                    table_size, int(args.harmonics), int(args.bands),
                    include_samples,
                    pro=pro,
                )
                for p in wavs
            ]

            done = 0
            for fut in as_completed(futures):
                processed += 1
                done += 1
                res = fut.result()

                if res.get("skipped_sample"):
                    skipped_samples += 1
                elif res.get("ok"):
                    db_entries.append(res["entry"])
                    kept += 1
                else:
                    failed_files += 1
                    print(f"[index] ERROR en {res.get('wav', '(wav?)')}: {res.get('error', 'error desconocido')}")

                _progress(done)

    # Mantener orden estable en entries (por nombre descriptor)
    db_entries.sort(key=lambda e: str(e.get("descriptor", "")))

    index: Dict[str, Any] = {
        "tableSize": int(table_size),
        "harmonics": int(args.harmonics),
        "bands": int(args.bands),
        "feature_sr": int(core.FEATURE_SR),
        "count": int(len(db_entries)),
        "processed": int(len(wavs)),
        "kept": int(kept),
        "failed_files": int(failed_files),
        "skipped_samples": int(skipped_samples),
        "include_samples": bool(include_samples),
        "jobs": int(jobs),
        "entries": db_entries,
    }

    # Info extra PRO (útil para tooling y para saber “qué idioma” usa la DB)
    if pro:
        index["pro"] = True
        index["pro_format"] = {
            "tableSize": int(core.PRO_TABLE_SIZE),
            "numFrames": int(core.PRO_NUM_FRAMES),
            "sr_out": int(core.PRO_SR),
            "pcm": "PCM_16",
            "layout": "mono_concatenated_frames",
            "naming": "__CAN__2048x64",
        }
        index["pro_outputs_dir"] = str(out_dir.as_posix())

    with open(out_dir / "_INDEX.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # Resumen final
    print(
        "[index] DONE\n"
        f"  in_dir:          {in_dir}\n"
        f"  out_dir:         {out_dir}\n"
        f"  processed:       {len(wavs)}\n"
        f"  kept:            {kept}\n"
        f"  failed_files:    {failed_files}\n"
        f"  skipped_samples: {skipped_samples} (include_samples={include_samples})\n"
        f"  params:          tableSize={int(table_size)} harmonics={int(args.harmonics)} bands={int(args.bands)} feature_sr={int(core.FEATURE_SR)}\n"
        f"  pro:             {pro}\n"
        + (f"  pro_format:       2048x64 PCM16 sr_out={core.PRO_SR} (WAV+JSON en out_dir)\n" if pro else "")
        + f"  jobs:            {jobs}"
    )
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


def _compat_warning(index_meta: Dict[str, Any], args) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Compara parámetros de index vs parámetros actuales del match.
    Devuelve: (mismatch, warning_str, compat_obj)
    """
    idx_ts = int(index_meta.get("tableSize", -1))
    idx_h = int(index_meta.get("harmonics", -1))
    idx_b = int(index_meta.get("bands", -1))
    idx_fs = index_meta.get("feature_sr", None)

    req_ts = int(getattr(args, "table_size", -1))
    req_h = int(getattr(args, "harmonics", -1))
    req_b = int(getattr(args, "bands", -1))
    req_fs = int(core.FEATURE_SR)

    mismatch = (idx_ts != req_ts) or (idx_h != req_h) or (idx_b != req_b)
    if idx_fs is not None:
        try:
            mismatch = mismatch or (int(idx_fs) != req_fs)
        except Exception:
            mismatch = True

    compat_obj = {
        "indexed": {"tableSize": idx_ts, "harmonics": idx_h, "bands": idx_b, "feature_sr": idx_fs},
        "requested": {"tableSize": req_ts, "harmonics": req_h, "bands": req_b, "feature_sr": req_fs},
        "mismatch": bool(mismatch),
    }

    if mismatch:
        warning = (
            "WARNING: Tu DB fue indexada con parámetros distintos a los del match.\n"
            f"  DB(index): tableSize={idx_ts}, harmonics={idx_h}, bands={idx_b}, feature_sr={idx_fs}\n"
            f"  Match:     tableSize={req_ts}, harmonics={req_h}, bands={req_b}, feature_sr={req_fs}\n"
            "  Esto puede producir resultados raros. Recomendación: reindexa la DB con los mismos parámetros."
        )
    else:
        warning = ""

    return mismatch, warning, compat_obj


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

    # Cargar index y verificar compat
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    mismatch, warning, compat_obj = _compat_warning(idx, args)
    if warning:
        print(warning)

    entries = idx.get("entries", [])
    if not entries:
        print("[match] Índice vacío.")
        return 2

    # Preparar target
    tx, _tsr = core.safe_read_wav(target_path)
    tx = core.standardize_audio(tx)

    # Tramo estable (evita el centro a ciegas)
    tseg = core.select_stable_segment(tx, int(args.table_size))
    tframe = core.standardize_frame(tseg, int(args.table_size), loop_fix=True)
    tfeat = core.compute_features_for_frame(tframe, int(args.harmonics), int(args.bands))

    th = np.array(tfeat.harmonics, dtype=np.float32)
    tenv = np.array(tfeat.spectral_env_db, dtype=np.float32)

    topk = int(_get_arg(args, "topk", 50))
    topn = int(_get_arg(args, "topn", 10))
    include_samples = bool(_get_arg(args, "include_samples", False))
    gain_mode = str(_get_arg(args, "gain_mode", "rms")).lower()
    gain_mode = "lufs" if gain_mode == "lufs" else "rms"

    counters = {
        "entries_total": int(len(entries)),
        "descriptor_loaded_ok": 0,
        "skipped_samples": 0,
        "skipped_bad_json": 0,
        "skipped_schema": 0,
        "skipped_compat": 0,
        "skipped_features": 0,
        "other_errors": 0,
    }

    # 1) Top-K por armónicos (ADN)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ent in entries:
        dpath = db_dir / ent.get("descriptor", "")
        try:
            desc = core.load_descriptor(dpath)
            counters["descriptor_loaded_ok"] += 1
        except Exception as e:
            msg = str(e).lower()
            if "json" in msg and ("invál" in msg or "inval" in msg or "decode" in msg):
                counters["skipped_bad_json"] += 1
            elif "feature_sr" in msg or "reindex" in msg or "parámetros" in msg or "parametros" in msg:
                counters["skipped_compat"] += 1
            elif "falta key" in msg or "schema" in msg or "features" in msg or "descriptor" in msg:
                counters["skipped_schema"] += 1
            else:
                counters["other_errors"] += 1
            continue

        if desc.get("type") == "sample" and not include_samples:
            counters["skipped_samples"] += 1
            continue

        try:
            h, _e, _r, _l = core.get_desc_features(desc)
        except Exception:
            counters["skipped_features"] += 1
            continue

        ch = np.array(h, dtype=np.float32)
        scored.append((core.cosine_distance(th, ch), desc))

    if not scored:
        print(
            "[match] No hay candidatos tras filtros.\n"
            f"  summary: {counters}"
        )
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

        if cand_env.shape != tenv.shape:
            counters["skipped_compat"] += 1
            continue

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

    if not results:
        print(
            "[match] No hay resultados tras refinamiento (posible mismatch bands o filtros).\n"
            f"  summary: {counters}"
        )
        return 2

    results.sort(key=lambda r: r["score"])
    results = results[: max(1, topn)]

    suggestions: List[str] = []
    if tfeat.noise_ratio_db > -12 or tfeat.tonalness < 0.6:
        suggestions.append("Target tiene bastante componente ruidosa: considera capa de noise/attack residual.")
    if tfeat.brightness > 0.7:
        suggestions.append("Target muy brillante: limita stacks o aplica lowpass suave en capas auxiliares.")
    if mismatch:
        suggestions.append("DB/Match con parámetros distintos: reindexar DB para resultados más confiables.")

    report: Dict[str, Any] = {
        "target": str(target_path.as_posix()),
        "target_selection": {"method": "stable_segment", "tableSize": int(args.table_size)},
        "db_compat": compat_obj,
        "summary": counters,
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

    txt_path = out_path.with_suffix(".txt")
    lines: List[str] = []
    lines.append("WAVETABLE MATCH REPORT")
    lines.append(f"Target: {report['target']}")
    lines.append(f"Gain mode: {'lufs' if gain_mode=='lufs' else 'rms_dbfs'}")
    lines.append("")
    lines.append("SUMMARY:")
    for k in (
        "entries_total",
        "descriptor_loaded_ok",
        "skipped_bad_json",
        "skipped_schema",
        "skipped_compat",
        "skipped_features",
        "skipped_samples",
        "other_errors",
    ):
        lines.append(f" - {k}: {counters.get(k)}")
    lines.append("")

    if mismatch:
        lines.append("DB PARAM WARNING:")
        lines.append(warning)
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
    print(f"[match] summary: {counters}")
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

    p_index.add_argument(
        "--include-samples",
        action="store_true",
        help="Incluye type='sample' (archivos no-wavetable)"
    )

    p_index.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Número de workers para index paralelo (default=1)."
    )

    # ✅ Modo PRO export canónico
    p_index.add_argument(
        "--pro",
        action="store_true",
        help="Exporta cada wavetable a formato canónico (2048x64 PCM16) y hace que el índice apunte a ese WAV (plugin-safe)."
    )

    # ✅ Se mantiene por compat, pero ya no se usa
    p_index.add_argument(
        "--pro-wav-dir",
        default=None,
        help="(DEPRECATED) Ya no se usa. En PRO, WAV+JSON se guardan en --out."
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


