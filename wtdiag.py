# wtdiag.py
# Wavetable Diagnoser + Matcher (batch/CLI)
# - Indexa carpeta de wavetables a descriptores JSON (spec + compat)
# - Diagnostica y sugiere uso
# - Hace matching contra un target (layer) y sugiere gain + EQ (curva delta suavizada + PEQ aproximado)

import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, get_window
from scipy.ndimage import gaussian_filter1d

# ----------------------------
# Parámetros estándar offline
# ----------------------------

FEATURE_SR = 48000       # SR "canónico" solo para mapear bandas (comparabilidad del envelope)
TARGET_PEAK = 0.999
COMMON_TABLE_SIZES = (256, 512, 1024, 2048, 4096, 8192)

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
    # shape típico: (n, ch)
    return np.mean(x, axis=1)

def normalize_peak(x: np.ndarray, target_peak: float = TARGET_PEAK) -> np.ndarray:
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
        return x.astype(np.float32)
    from fractions import Fraction
    frac = Fraction(n, len(x)).limit_denominator(4096)
    y = resample_poly(x, frac.numerator, frac.denominator)
    return ensure_len(y.astype(np.float32), n)

# ----------------------------
# Normalización obligatoria (Paso 0)
# ----------------------------

def standardize_audio(x: np.ndarray) -> np.ndarray:
    # Mono ya lo hace safe_read_wav()
    x = remove_dc(x)
    x = normalize_peak(x, target_peak=TARGET_PEAK)
    return x.astype(np.float32)

def standardize_frame(frame: np.ndarray, table_size: int) -> np.ndarray:
    frame = resample_to_len(frame, table_size)
    frame = remove_dc(frame)
    frame = normalize_peak(frame, target_peak=TARGET_PEAK)
    return frame.astype(np.float32)

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
# Detección de layout (single vs multi vs sample)
# ----------------------------

def infer_layout(n_samples: int, preferred_table: int) -> Tuple[str, int, int]:
    """
    Returns: (type, in_table_size, num_frames)
    type: single_cycle | multi_frame | sample
    """
    # 1) single-cycle típico
    if n_samples in COMMON_TABLE_SIZES:
        return "single_cycle", n_samples, 1

    # 2) multi-frame exacto con table común
    for ts in COMMON_TABLE_SIZES:
        if n_samples % ts == 0:
            nf = n_samples // ts
            if nf >= 2:
                return "multi_frame", ts, nf

    # 3) casi coincide con tu table_size
    if abs(n_samples - preferred_table) <= 4:
        return "single_cycle", preferred_table, 1

    return "sample", preferred_table, 1

# ----------------------------
# Lógica DSP (features)
# ----------------------------

def log_band_edges(fmin: float, fmax: float, n_bands: int) -> np.ndarray:
    return np.geomspace(fmin, fmax, n_bands + 1)

def spectral_envelope_db(
    mag: np.ndarray,
    sr: int,
    n_fft: int,
    n_bands: int = 128,
    fmin: float = 20.0,
    fmax: float = 20000.0
) -> np.ndarray:
    # mag: magnitud rfft (len = n_fft//2+1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = log_band_edges(fmin, min(fmax, sr / 2 - 1), n_bands)
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

def harmonic_vector(frame: np.ndarray, n_harm: int = 64) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extrae vector armónico H1..Hn de un frame single-cycle:
    - Asume un ciclo => fundamental corresponde a bin 1.
    """
    N = len(frame)
    # OJO: Hann puede sesgar un poco; se deja por robustez ante bordes imperfectos.
    w = get_window("hann", N, fftbins=True).astype(np.float32)
    X = np.fft.rfft(frame * w, n=N)
    mag = np.abs(X).astype(np.float32)

    max_bin = min(n_harm, len(mag) - 1)
    h = mag[1:max_bin + 1].copy()
    if h.size < n_harm:
        h = np.pad(h, (0, n_harm - h.size), mode="constant")

    denom = float(np.sum(h) + 1e-12)
    h_norm = (h / denom).astype(np.float32)

    odd = float(np.sum(h_norm[0::2]) + 1e-12)   # H1,H3,...
    even = float(np.sum(h_norm[1::2]) + 1e-12)
    odd_even = odd / even

    k = np.arange(1, n_harm + 1, dtype=np.float32)
    roll = float(np.sum(h_norm * (k / n_harm)))

    split = int(max(1, n_harm * 0.25))
    bright = float(np.sum(h_norm[split:]))

    extras = {"odd_even_ratio": odd_even, "rolloff": roll, "brightness": bright}
    return h_norm, extras

def tonalness_and_noise(frame: np.ndarray) -> Tuple[float, float]:
    """
    Estima tonalness: energía cerca de armónicos vs total (en espectro).
    Para single-cycle, armónicos caen en bins enteros.
    """
    N = len(frame)
    w = get_window("hann", N, fftbins=True).astype(np.float32)
    X = np.fft.rfft(frame * w, n=N)
    mag = np.abs(X).astype(np.float32)

    total = float(np.sum(mag[1:]) + 1e-12)

    harmonic_bins = []
    max_bin = len(mag) - 1
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

def compute_features_for_frame(frame: np.ndarray, n_harm: int, n_bands: int) -> WTFeatures:
    # frame ya viene estandarizado (DC + peak + tableSize), pero lo reaseguramos barato:
    f = standardize_frame(frame, len(frame))

    hvec, extra = harmonic_vector(f, n_harm=n_harm)

    # Espectro y envelope (SR canónico para bordes de banda => comparable)
    N = len(f)
    w = get_window("hann", N, fftbins=True).astype(np.float32)
    X = np.fft.rfft(f * w, n=N)
    mag = np.abs(X).astype(np.float32)
    env = spectral_envelope_db(mag, FEATURE_SR, N, n_bands=n_bands)

    tonal, noise_db = tonalness_and_noise(f)
    cdb = crest_factor_db(f)
    rdb = float(20.0 * math.log10(rms(f) + 1e-12))

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

    H = stack("harmonics")
    E = stack("spectral_env_db")

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
# Diagnóstico humano (spec family + best_for)
# ----------------------------

def classify_family(feat: WTFeatures) -> str:
    tonal = feat.tonalness
    bright = feat.brightness
    oe = feat.odd_even_ratio
    noise = feat.noise_ratio_db

    if tonal < 0.35 or noise > -6:
        return "noise"
    if oe > 2.2 and bright < 0.45:
        return "square"
    if bright > 0.65 and tonal > 0.7:
        return "saw"
    if bright < 0.25 and tonal > 0.85:
        return "sine"
    if bright < 0.35 and tonal > 0.75 and oe < 1.6:
        return "triangle"
    # (formant es difícil sin análisis extra; lo dejamos como complex por ahora)
    return "complex"

def best_for_tags(feat: WTFeatures, motion_std: Optional[WTFeatures]) -> List[str]:
    # set pedido: bass, lead, pad, pluck_layer, air_noise, fx
    tags: List[str] = []

    if feat.tonalness > 0.82 and feat.brightness < 0.35 and feat.noise_ratio_db < -14:
        tags += ["bass"]
    if feat.tonalness > 0.70 and feat.brightness >= 0.35 and feat.noise_ratio_db < -10:
        tags += ["lead", "pluck_layer"]
    if feat.tonalness < 0.55 or feat.noise_ratio_db > -10:
        tags += ["air_noise", "fx"]

    if motion_std is not None:
        # si hay movimiento fuerte -> pad
        if motion_std.brightness > 0.08 or float(np.mean(motion_std.spectral_env_db)) > 3.0:
            tags += ["pad"]

    if not tags:
        tags = ["lead"]

    return sorted(list(set(tags)))

def build_diagnosis(feat: WTFeatures, std: Optional[WTFeatures]) -> WTDiagnosis:
    fam = classify_family(feat)

    pitch_friendly = float(np.clip((feat.tonalness * 1.15) - (max(0.0, (feat.noise_ratio_db + 20) / 40.0)), 0, 1))
    needs_filter = float(np.clip((feat.brightness * 0.8) + (max(0.0, (feat.noise_ratio_db + 30) / 30.0) * 0.6), 0, 1))

    notes_parts: List[str] = []
    if feat.odd_even_ratio > 2.0:
        notes_parts.append("Armónicos impares fuertes (timbre tipo square/clarinet).")
    if feat.brightness > 0.65:
        notes_parts.append("Brillo alto (posible necesidad de lowpass al apilar).")
    if feat.noise_ratio_db > -10:
        notes_parts.append("Componente ruidosa notable.")
    if std is not None and (std.brightness > 0.08 or float(np.mean(std.spectral_env_db)) > 3.0):
        notes_parts.append("Multi-frame con movimiento tímbrico apreciable (bueno para pads/morph).")

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
# Descriptor JSON (spec + compat)
# ----------------------------

def descriptor_to_spec(desc: WTDescriptor) -> Dict[str, Any]:
    feat = desc.features_mean
    spec: Dict[str, Any] = {
        "path": desc.path,
        "type": desc.type,
        "tableSize": desc.tableSize,
        "numFrames": desc.numFrames,
        "sampleRate": desc.sampleRate,
        # spec pedido:
        "features": {
            "harmonics_64": feat.harmonics,
            "spectral_env_db_128": feat.spectral_env_db,
            "tonalness": feat.tonalness,
            "noise_ratio_db": feat.noise_ratio_db,
            "brightness": feat.brightness,
            "odd_even_ratio": feat.odd_even_ratio,
            "rolloff": feat.rolloff,
            "crest_db": feat.crest_db,
            "rms_dbfs": feat.rms_dbfs,
        },
        "diagnosis": asdict(desc.diagnosis),
        "how_to_use": {
            "layer_role": ["main_tone"] if feat.tonalness > 0.75 else ["texture_layer"],
            "recommended_filter": "lowpass gentle if stacking" if feat.brightness > 0.65 else "none",
            "recommended_gain_db_range": [-12, -3] if feat.tonalness > 0.7 else [-18, -6],
        },
        # compat (para no romper cosas viejas):
        "features_mean": asdict(desc.features_mean),
        "features_std": asdict(desc.features_std) if desc.features_std is not None else None,
    }

    if desc.features_std is not None:
        spec["features_std_spec"] = {
            "harmonics_std": desc.features_std.harmonics,
            "spectral_env_db_std": desc.features_std.spectral_env_db,
            "brightness_std": desc.features_std.brightness,
        }
    return spec

def _get_desc_features(desc: Dict[str, Any]) -> Tuple[List[float], List[float], float]:
    """
    Soporta:
    - spec nuevo: desc["features"]["harmonics_64"], ["spectral_env_db_128"], ["rms_dbfs"]
    - compat viejo: desc["features_mean"]["harmonics"], ["spectral_env_db"], ["rms_dbfs"]
    """
    if "features" in desc:
        h = desc["features"].get("harmonics_64")
        e = desc["features"].get("spectral_env_db_128")
        r = desc["features"].get("rms_dbfs")
        if h is not None and e is not None and r is not None:
            return h, e, float(r)

    # fallback
    fm = desc.get("features_mean", {})
    h = fm.get("harmonics")
    e = fm.get("spectral_env_db")
    r = fm.get("rms_dbfs")
    if h is None or e is None or r is None:
        raise KeyError("Descriptor sin features esperadas (ni spec ni compat).")
    return h, e, float(r)

# ----------------------------
# Indexado
# ----------------------------

def process_wavetable(path: Path, table_size: int, n_harm: int, n_bands: int) -> WTDescriptor:
    x, sr = safe_read_wav(path)
    x = standardize_audio(x)

    wtype, in_table, num_frames = infer_layout(len(x), table_size)

    if wtype == "single_cycle":
        frame = standardize_frame(x[:in_table], table_size)
        frames = frame.reshape(1, -1)
        num_frames = 1

    elif wtype == "multi_frame":
        x = ensure_len(x, in_table * num_frames)
        raw_frames = x.reshape(num_frames, in_table)
        frames = np.stack([standardize_frame(raw_frames[i], table_size) for i in range(num_frames)], axis=0)

    else:
        # sample/unknown -> diagnosticable pero no ideal para matching
        frame = standardize_frame(x[:min(len(x), table_size)], table_size)
        frames = frame.reshape(1, -1)
        num_frames = 1

    feats = [compute_features_for_frame(frames[i], n_harm=n_harm, n_bands=n_bands) for i in range(num_frames)]
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

    db_entries: List[Dict[str, Any]] = []
    for i, p in enumerate(wavs, 1):
        try:
            desc = process_wavetable(p, args.table_size, args.harmonics, args.bands)
            # Evitar colisiones: nombre basado en ruta relativa
            rel = p.relative_to(in_dir).as_posix()
            safe = rel.replace("/", "__").replace("\\", "__")
            out_json = out_dir / (Path(safe).with_suffix(".json").name)

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(descriptor_to_spec(desc), f, ensure_ascii=False, indent=2)

            db_entries.append({
                "path": desc.path,
                "descriptor": out_json.name,  # relativo al db_dir (portable)
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

def bands_centers_hz(sr: int, n_bands: int, fmin: float = 20.0, fmax: float = 20000.0) -> np.ndarray:
    fmax = min(fmax, sr / 2 - 1.0)
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return centers

def delta_to_peq(delta_db: np.ndarray, sr: int, max_filters: int = 6, min_sep_bands: int = 6) -> List[Dict[str, float]]:
    """
    Heurística:
    - toma los |delta| más altos
    - evita poner filtros muy cerca
    - Q fijo moderado (se puede mejorar)
    """
    n = len(delta_db)
    centers = bands_centers_hz(sr, n)
    idx_sorted = np.argsort(np.abs(delta_db))[::-1]

    chosen: List[int] = []
    for idx in idx_sorted:
        if len(chosen) >= max_filters:
            break
        if any(abs(int(idx) - c) < min_sep_bands for c in chosen):
            continue
        chosen.append(int(idx))

    filters: List[Dict[str, float]] = []
    for idx in chosen:
        filters.append({
            "type": "peaking",
            "f0_hz": float(centers[idx]),
            "gain_db": float(delta_db[idx]),
            "q": 1.2
        })
    return filters

def load_descriptor(path: Path) -> Dict[str, Any]:
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

    # Target: normalización + recorte centro + frame estándar
    tx, tsr = safe_read_wav(target_path)
    tx = standardize_audio(tx)

    if len(tx) >= args.table_size:
        start = (len(tx) - args.table_size) // 2
        tframe = tx[start:start + args.table_size]
    else:
        tframe = tx

    tframe = standardize_frame(tframe, args.table_size)
    tfeat = compute_features_for_frame(tframe, args.harmonics, args.bands)

    th = np.array(tfeat.harmonics, dtype=np.float32)
    tenv = np.array(tfeat.spectral_env_db, dtype=np.float32)

    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    entries = idx.get("entries", [])

    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for it in entries:
        try:
            dpath = db_dir / it["descriptor"]  # relativo al db_dir
            desc = load_descriptor(dpath)

            if desc.get("type") == "sample" and not args.include_samples:
                continue

            ch_list, cenv_list, _c_rms = _get_desc_features(desc)
            ch = np.array(ch_list, dtype=np.float32)

            dist = cosine_distance(th, ch)
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
        ch_list, cenv_list, c_rms = _get_desc_features(desc)
        cenv = np.array(cenv_list, dtype=np.float32)

        delta = tenv - cenv
        delta = smooth_and_limit_delta(delta, limit_db=args.eq_limit_db, smooth_sigma=args.eq_smooth)

        # PEQ (3–6 filtros)
        peq = delta_to_peq(delta, sr=FEATURE_SR, max_filters=6, min_sep_bands=6)

        # Gain match: RMS target vs RMS candidato (dB)
        t_rms = float(tfeat.rms_dbfs)
        gain_db = float(t_rms - float(c_rms))

        # Penalización por EQ extremo
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

    # Recomendaciones automáticas target (attack/noise)
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
            "feature_sr": FEATURE_SR
        },
        "suggestions": suggestions,
        "matches": results
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # TXT humano
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
        json.dump(descriptor_to_spec(desc), f, ensure_ascii=False, indent=2)
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
