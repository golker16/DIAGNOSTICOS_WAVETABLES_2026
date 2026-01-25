# wtdiag.py
# Wavetable Diagnoser + Matcher (batch/CLI)
# - Indexa carpeta de wavetables a descriptores JSON
# - Diagnostica y sugiere uso
# - Hace matching contra un target (layer) y sugiere gain + "EQ/filtro" (curva delta suavizada)

import argparse, json, math, os, sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, get_window
from scipy.ndimage import gaussian_filter1d

# ----------------------------
# Utilidades básicas
# ----------------------------

def db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))

def undb(x_db: np.ndarray) -> np.ndarray:
    return 10.0 ** (x_db / 20.0)

def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.maximum(x * x, eps))))

def crest_factor_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) + 1e-12
    return 20.0 * math.log10(peak / (rms(x) + 1e-12))

def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)

def normalize_peak(x: np.ndarray, target_peak: float = 0.999) -> np.ndarray:
    p = np.max(np.abs(x)) + 1e-12
    return x * (target_peak / p)

def safe_read_wav(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    x = to_mono(np.asarray(x, dtype=np.float32))
    return x, int(sr)

def ensure_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) < n:
        out = np.zeros(n, dtype=np.float32)
        out[:len(x)] = x
        return out
    return x[:n]

def resample_to_len(x: np.ndarray, n: int) -> np.ndarray:
    # Resample por razón racional aproximada usando resample_poly (mejor que FFT-resample para ciclos).
    if len(x) == n:
        return x
    # Factor aproximado
    from fractions import Fraction
    frac = Fraction(n, len(x)).limit_denominator(4096)
    y = resample_poly(x, frac.numerator, frac.denominator)
    return ensure_len(y.astype(np.float32), n)

# ----------------------------
# Especificación de features
# ----------------------------

@dataclass
class WTFeatures:
    harmonics: List[float]              # H1..Hn normalizado
    spectral_env_db: List[float]        # bandas log suavizadas en dB
    tonalness: float                    # 0..1 (armónico vs total)
    noise_ratio_db: float               # dB (ruido vs armónico aprox)
    brightness: float                   # 0..1
    odd_even_ratio: float               # energía impares / pares
    rolloff: float                      # 0..1 aprox (caída)
    crest_db: float                     # crest factor dB
    rms_dbfs: float                     # RMS en dBFS (a pico ~0)

@dataclass
class WTDiagnosis:
    family: str
    best_for: List[str]
    pitch_friendly: float
    needs_filtering: float
    notes: str

@dataclass
class WTDescriptor:
    path: str
    type: str                 # single_cycle | multi_frame | sample | unknown
    tableSize: int
    numFrames: int
    sampleRate: int
    features_mean: WTFeatures
    features_std: Optional[WTFeatures]  # solo multi-frame
    diagnosis: WTDiagnosis

# ----------------------------
# Lógica DSP (wavetable)
# ----------------------------

def detect_type_and_frames(x: np.ndarray, table_size: int) -> Tuple[str, int]:
    n = len(x)
    if n in (512, 1024, 2048, 4096) or abs(n - table_size) < 4:
        return "single_cycle", 1
    if n % table_size == 0 and n // table_size >= 2:
        return "multi_frame", n // table_size
    # puede ser sample normal o wavetable rara
    return "sample", 1

def split_frames(x: np.ndarray, table_size: int, num_frames: int) -> np.ndarray:
    if num_frames == 1:
        return x.reshape(1, -1)
    x = ensure_len(x, table_size * num_frames)
    return x.reshape(num_frames, table_size)

def log_band_edges(fmin: float, fmax: float, n_bands: int) -> np.ndarray:
    return np.geomspace(fmin, fmax, n_bands + 1)

def spectral_envelope_db(mag: np.ndarray, sr: int, n_fft: int, n_bands: int = 128,
                         fmin: float = 20.0, fmax: float = 20000.0) -> np.ndarray:
    # mag: magnitud rfft (len = n_fft//2+1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = log_band_edges(fmin, min(fmax, sr/2 - 1), n_bands)
    env = np.zeros(n_bands, dtype=np.float32)
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            env[i] = -120.0
        else:
            env[i] = float(np.mean(db(mag[idx])))
    # suavizado ligero para “color”
    env = gaussian_filter1d(env, sigma=1.0)
    return env

def harmonic_vector(frame: np.ndarray, sr: int, n_harm: int = 64) -> Tuple[np.ndarray, Dict]:
    """
    Extrae vector armónico H1..Hn de un frame single-cycle:
    - Asume un ciclo => fundamental = 1*sr/N
    - Harmónico k corresponde al bin k en FFT (porque 1 ciclo por ventana).
    """
    N = len(frame)
    w = get_window("hann", N, fftbins=True).astype(np.float32)
    xw = frame * w

    X = np.fft.rfft(xw, n=N)
    mag = np.abs(X).astype(np.float32)

    # Fundamental en bin 1 (bin0 es DC)
    # Tomamos bins 1..n_harm (si existen)
    max_bin = min(n_harm, len(mag) - 1)
    h = mag[1:max_bin + 1].copy()
    if h.size < n_harm:
        h = np.pad(h, (0, n_harm - h.size), mode="constant")

    # Normalizar por energía total armónica
    denom = float(np.sum(h) + 1e-12)
    h_norm = h / denom

    # Odd/even ratio
    odd = float(np.sum(h_norm[0::2]) + 1e-12)   # H1,H3,... índices 0,2,...
    even = float(np.sum(h_norm[1::2]) + 1e-12)
    odd_even = odd / even

    # Rolloff (qué tan rápido cae): correlación con log(k)
    k = np.arange(1, n_harm + 1, dtype=np.float32)
    # peso altas vs bajas
    roll = float(np.sum(h_norm * (k / n_harm)))

    # Brightness simple: energía relativa en armónicos > n_harm*0.25
    split = int(max(1, n_harm * 0.25))
    bright = float(np.sum(h_norm[split:]))

    extras = {"odd_even_ratio": odd_even, "rolloff": roll, "brightness": bright}
    return h_norm.astype(np.float32), extras

def tonalness_and_noise(frame: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Estima tonalness: energía cerca de armónicos vs total (en espectro).
    Para single-cycle, armónicos caen en bins enteros, así que esto funciona bien.
    """
    N = len(frame)
    w = get_window("hann", N, fftbins=True).astype(np.float32)
    X = np.fft.rfft(frame * w, n=N)
    mag = np.abs(X).astype(np.float32)

    total = float(np.sum(mag[1:]) + 1e-12)

    # Armónicos: bins 1..N//2; tomamos vecinos +/-1 bin como "armónico"
    harmonic_bins = []
    max_bin = len(mag) - 1
    # limitamos a 128 armónicos para tonalness (si N lo permite)
    n_h = min(128, max_bin)
    for k in range(1, n_h + 1):
        for b in (k - 1, k, k + 1):
            if 1 <= b <= max_bin:
                harmonic_bins.append(b)
    harmonic_bins = np.unique(harmonic_bins)

    harm_energy = float(np.sum(mag[harmonic_bins]) + 1e-12)
    noise_energy = max(total - harm_energy, 1e-12)

    tonal = float(np.clip(harm_energy / total, 0.0, 1.0))
    noise_ratio_db = float(10.0 * math.log10(noise_energy / harm_energy + 1e-12))
    return tonal, noise_ratio_db

def compute_features_for_frame(frame: np.ndarray, sr: int, n_harm: int, n_bands: int) -> WTFeatures:
    f = remove_dc(frame)
    f = normalize_peak(f)

    hvec, extra = harmonic_vector(f, sr, n_harm=n_harm)

    # Espectro y envelope
    N = len(f)
    w = get_window("hann", N, fftbins=True).astype(np.float32)
    X = np.fft.rfft(f * w, n=N)
    mag = np.abs(X).astype(np.float32)
    env = spectral_envelope_db(mag, sr, N, n_bands=n_bands)

    tonal, noise_db = tonalness_and_noise(f, sr)
    cdb = crest_factor_db(f)
    rdb = float(20.0 * math.log10(rms(f) + 1e-12))

    # Notas: brightness/rolloff ya están, normalizamos a 0..1 de forma “aprox”
    brightness = float(np.clip(extra["brightness"], 0.0, 1.0))
    rolloff = float(np.clip(extra["rolloff"], 0.0, 1.0))
    odd_even = float(extra["odd_even_ratio"])

    return WTFeatures(
        harmonics=hvec.tolist(),
        spectral_env_db=env.tolist(),
        tonalness=tonal,
        noise_ratio_db=noise_db,
        brightness=brightness,
        odd_even_ratio=odd_even,
        rolloff=rolloff,
        crest_db=cdb,
        rms_dbfs=rdb
    )

def aggregate_features(frames_feats: List[WTFeatures]) -> Tuple[WTFeatures, Optional[WTFeatures]]:
    def stack(field: str) -> np.ndarray:
        vals = [np.array(getattr(ff, field), dtype=np.float32) for ff in frames_feats]
        return np.stack(vals, axis=0)

    # Vectores
    H = stack("harmonics")
    E = stack("spectral_env_db")

    # Escalares
    tonal = np.array([ff.tonalness for ff in frames_feats], dtype=np.float32)
    noise = np.array([ff.noise_ratio_db for ff in frames_feats], dtype=np.float32)
    bright = np.array([ff.brightness for ff in frames_feats], dtype=np.float32)
    oe = np.array([ff.odd_even_ratio for ff in frames_feats], dtype=np.float32)
    roll = np.array([ff.rolloff for ff in frames_feats], dtype=np.float32)
    crest = np.array([ff.crest_db for ff in frames_feats], dtype=np.float32)
    rdb = np.array([ff.rms_dbfs for ff in frames_feats], dtype=np.float32)

    mean = WTFeatures(
        harmonics=np.mean(H, axis=0).tolist(),
        spectral_env_db=np.mean(E, axis=0).tolist(),
        tonalness=float(np.mean(tonal)),
        noise_ratio_db=float(np.mean(noise)),
        brightness=float(np.mean(bright)),
        odd_even_ratio=float(np.mean(oe)),
        rolloff=float(np.mean(roll)),
        crest_db=float(np.mean(crest)),
        rms_dbfs=float(np.mean(rdb)),
    )

    if len(frames_feats) <= 1:
        return mean, None

    std = WTFeatures(
        harmonics=np.std(H, axis=0).tolist(),
        spectral_env_db=np.std(E, axis=0).tolist(),
        tonalness=float(np.std(tonal)),
        noise_ratio_db=float(np.std(noise)),
        brightness=float(np.std(bright)),
        odd_even_ratio=float(np.std(oe)),
        rolloff=float(np.std(roll)),
        crest_db=float(np.std(crest)),
        rms_dbfs=float(np.std(rdb)),
    )
    return mean, std

# ----------------------------
# Diagnóstico humano
# ----------------------------

def classify_family(feat: WTFeatures) -> str:
    # Heurística simple (orientativa)
    tonal = feat.tonalness
    bright = feat.brightness
    oe = feat.odd_even_ratio
    noise = feat.noise_ratio_db

    if tonal < 0.35 or noise > -6:
        return "noise"
    if oe > 2.2 and bright < 0.45:
        return "square_like"
    if bright > 0.65 and tonal > 0.7:
        return "saw_like"
    if bright < 0.25 and tonal > 0.85:
        return "sine_like"
    return "complex"

def best_for_tags(feat: WTFeatures, motion_std: Optional[WTFeatures]) -> List[str]:
    tags = []
    if feat.tonalness > 0.8 and feat.brightness < 0.35:
        tags += ["bass", "sub_layer"]
    if feat.tonalness > 0.7 and feat.brightness >= 0.35:
        tags += ["lead", "pluck_layer"]
    if feat.tonalness < 0.5:
        tags += ["texture", "air_noise", "fx"]
    if motion_std is not None:
        # si hay “movimiento” fuerte en envelope/brightness -> pad/morph
        if motion_std.brightness > 0.08 or np.mean(motion_std.spectral_env_db) > 3.0:
            tags += ["pad", "morphing"]
    if not tags:
        tags = ["general"]
    return sorted(list(set(tags)))

def build_diagnosis(feat: WTFeatures, std: Optional[WTFeatures]) -> WTDiagnosis:
    fam = classify_family(feat)

    pitch_friendly = float(np.clip((feat.tonalness * 1.15) - (max(0.0, (feat.noise_ratio_db + 20) / 40.0)), 0, 1))
    needs_filter = float(np.clip((feat.brightness * 0.8) + (max(0.0, (feat.noise_ratio_db + 30) / 30.0) * 0.6), 0, 1))

    notes_parts = []
    if feat.odd_even_ratio > 2.0:
        notes_parts.append("Impares dominantes (timbre tipo square/clarinet).")
    if feat.brightness > 0.65:
        notes_parts.append("Brillo alto (posible necesidad de lowpass al apilar).")
    if feat.noise_ratio_db > -10:
        notes_parts.append("Componente ruidosa notable.")
    if std is not None and (std.brightness > 0.08 or np.mean(std.spectral_env_db) > 3.0):
        notes_parts.append("Multi-frame con movimiento tímbrico apreciable.")

    if not notes_parts:
        notes_parts.append("Armónicos ordenados, color estable.")

    best = best_for_tags(feat, std)
    return WTDiagnosis(
        family=fam,
        best_for=best,
        pitch_friendly=pitch_friendly,
        needs_filtering=needs_filter,
        notes=" ".join(notes_parts)
    )

# ----------------------------
# Indexado
# ----------------------------

def process_wavetable(path: Path, table_size: int, n_harm: int, n_bands: int) -> WTDescriptor:
    x, sr = safe_read_wav(path)
    x = remove_dc(x)

    wtype, num_frames = detect_type_and_frames(x, table_size)

    # Si parece single-cycle pero no es table_size, lo resampleamos
    if wtype == "single_cycle":
        frame = resample_to_len(x, table_size)
        frames = frame.reshape(1, -1)
        num_frames = 1
    elif wtype == "multi_frame":
        # Aseguramos frames exactos
        frames = split_frames(x, table_size, num_frames)
    else:
        # sample/unknown: tratamos una ventana para diagnóstico básico
        # (no recomendado para matching, pero se indexa igual como "sample")
        frame = resample_to_len(x[:min(len(x), table_size)], table_size)
        frames = frame.reshape(1, -1)
        num_frames = 1

    feats = [compute_features_for_frame(frames[i], sr, n_harm=n_harm, n_bands=n_bands) for i in range(num_frames)]
    mean, std = aggregate_features(feats)
    diag = build_diagnosis(mean, std)

    return WTDescriptor(
        path=str(path.as_posix()),
        type=wtype,
        tableSize=table_size,
        numFrames=num_frames,
        sampleRate=sr,
        features_mean=mean,
        features_std=std,
        diagnosis=diag
    )

def cmd_index(args) -> int:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted([p for p in in_dir.rglob("*.wav")])
    if not wavs:
        print(f"[index] No se encontraron .wav en: {in_dir}")
        return 2

    db_entries = []
    for i, p in enumerate(wavs, 1):
        try:
            desc = process_wavetable(p, args.table_size, args.harmonics, args.bands)
            out_json = out_dir / (p.stem + ".json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(asdict(desc), f, ensure_ascii=False, indent=2)
            db_entries.append({
                "path": desc.path,
                "descriptor": str(out_json.as_posix()),
                "type": desc.type,
                "family": desc.diagnosis.family,
                "best_for": desc.diagnosis.best_for
            })
            print(f"[{i:5d}/{len(wavs):5d}] OK  {p.name}  -> {out_json.name}")
        except Exception as e:
            print(f"[{i:5d}/{len(wavs):5d}] FAIL {p.name}  ({e})")

    # índice global
    index_path = out_dir / "_INDEX.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"root": str(in_dir.as_posix()), "entries": db_entries}, f, ensure_ascii=False, indent=2)

    print(f"\n[index] Listo. Index global: {index_path}")
    return 0

# ----------------------------
# Matching
# ----------------------------

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(1.0 - (np.dot(a, b) / denom))

def smooth_and_limit_delta(delta_db: np.ndarray, limit_db: float, smooth_sigma: float) -> np.ndarray:
    d = gaussian_filter1d(delta_db, sigma=smooth_sigma)
    d = np.clip(d, -limit_db, limit_db)
    return d

def load_descriptor(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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

    # Target features
    tx, tsr = safe_read_wav(target_path)
    tx = remove_dc(tx)
    # Para target: tomamos ventana estable: primeros tableSize samples (o el centro)
    if len(tx) >= args.table_size:
        start = (len(tx) - args.table_size) // 2
        tframe = tx[start:start + args.table_size]
    else:
        tframe = resample_to_len(tx, args.table_size)

    tfeat = compute_features_for_frame(resample_to_len(tframe, args.table_size), tsr, args.harmonics, args.bands)
    th = np.array(tfeat.harmonics, dtype=np.float32)
    tenv = np.array(tfeat.spectral_env_db, dtype=np.float32)

    # Cargar base
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    entries = idx["entries"]
    scored = []
    for it in entries:
        dpath = Path(it["descriptor"])
        try:
            desc = load_descriptor(dpath)
            if desc.get("type") == "sample" and not args.include_samples:
                continue
            ch = np.array(desc["features_mean"]["harmonics"], dtype=np.float32)
            dist = cosine_distance(th, ch)
            scored.append((dist, it, desc))
        except Exception:
            continue

    if not scored:
        print("[match] Base vacía o no compatible.")
        return 3

    scored.sort(key=lambda x: x[0])
    top = scored[:args.topk]

    # Armar resultados
    results = []
    for rank, (hdist, it, desc) in enumerate(top, 1):
        cenv = np.array(desc["features_mean"]["spectral_env_db"], dtype=np.float32)
        delta = tenv - cenv
        delta = smooth_and_limit_delta(delta, limit_db=args.eq_limit_db, smooth_sigma=args.eq_smooth)

        # Gain match: RMS target vs RMS candidate (dB)
        # Nota: esto es "por capa" para que suene con impacto similar.
        t_rms = tfeat.rms_dbfs
        c_rms = float(desc["features_mean"]["rms_dbfs"])
        gain_db = float(t_rms - c_rms)

        # Penalización por EQ extremo
        eq_cost = float(np.mean(np.abs(delta)) / max(1e-6, args.eq_limit_db))
        score = float(hdist * 0.75 + eq_cost * 0.25)

        results.append({
            "rank": rank,
            "wavetable_path": desc["path"],
            "descriptor": it["descriptor"],
            "score": score,
            "harmonic_cosine_dist": float(hdist),
            "gain_db_suggested": gain_db,
            "delta_env_db": delta.tolist(),
            "diagnosis": desc.get("diagnosis", {})
        })

    report = {
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
            "eq_smooth": args.eq_smooth
        },
        "matches": results
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # TXT humano
    txt_path = out_path.with_suffix(".txt")
    lines = []
    lines.append("WAVETABLE MATCH REPORT")
    lines.append(f"Target: {target_path}")
    lines.append(f"Target tonalness={tfeat.tonalness:.3f} noise_ratio_db={tfeat.noise_ratio_db:.2f} bright={tfeat.brightness:.3f}")
    lines.append("")
    for r in results:
        lines.append(f"#{r['rank']} score={r['score']:.4f} harmDist={r['harmonic_cosine_dist']:.4f} gain={r['gain_db_suggested']:+.2f} dB")
        lines.append(f"   WT: {r['wavetable_path']}")
        d = r["diagnosis"]
        if d:
            lines.append(f"   family={d.get('family')} best_for={d.get('best_for')} pitch_friendly={d.get('pitch_friendly')}")
            lines.append(f"   notes: {d.get('notes')}")
        lines.append(f"   delta_env_db (bands): meanAbs={np.mean(np.abs(r['delta_env_db'])):.2f} dB (limit {args.eq_limit_db} dB)")
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[match] OK -> {out_path}")
    print(f"[match] TXT -> {txt_path}")
    return 0

# ----------------------------
# Diagnóstico puntual
# ----------------------------

def cmd_diag(args) -> int:
    p = Path(args.wav)
    if not p.exists():
        print(f"[diag] No existe: {p}")
        return 2

    desc = process_wavetable(p, args.table_size, args.harmonics, args.bands)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(asdict(desc), f, ensure_ascii=False, indent=2)
    print(f"[diag] OK -> {out}")
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
