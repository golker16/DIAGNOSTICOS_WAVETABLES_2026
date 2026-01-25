# wtdiag-1.py
# Core del motor (DSP/features/descriptor/matching helpers)
# Nota: se carga dinámicamente desde wtdiag-2.py y wtgui.py (porque el nombre tiene guion).

import json, math
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
    harmonics: List[float]
    spectral_env_db: List[float]
    tonalness: float
    noise_ratio_db: float
    brightness: float
    odd_even_ratio: float
    rolloff: float
    crest_db: float
    rms_dbfs: float

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
    type: str
    tableSize: int
    numFrames: int
    sampleRate: int
    features_mean: WTFeatures
    features_std: Optional[WTFeatures]
    diagnosis: WTDiagnosis

# ----------------------------
# Detección de layout (single vs multi vs sample)
# ----------------------------

def infer_layout(n_samples: int, preferred_table: int) -> Tuple[str, int, int]:
    if n_samples in COMMON_TABLE_SIZES:
        return "single_cycle", n_samples, 1

    for ts in COMMON_TABLE_SIZES:
        if n_samples % ts == 0:
            nf = n_samples // ts
            if nf >= 2:
                return "multi_frame", ts, nf

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

    env = gaussian_filter1d(env, sigma=1.0)
    return env

def harmonic_vector(frame: np.ndarray, n_harm: int = 64) -> Tuple[np.ndarray, Dict[str, float]]:
    N = len(frame)
    w = get_window("hann", N, fftbins=True).astype(np.float32)
    X = np.fft.rfft(frame * w, n=N)
    mag = np.abs(X).astype(np.float32)

    max_bin = min(n_harm, len(mag) - 1)
    h = mag[1:max_bin + 1].copy()
    if h.size < n_harm:
        h = np.pad(h, (0, n_harm - h.size), mode="constant")

    denom = float(np.sum(h) + 1e-12)
    h_norm = (h / denom).astype(np.float32)

    odd = float(np.sum(h_norm[0::2]) + 1e-12)
    even = float(np.sum(h_norm[1::2]) + 1e-12)
    odd_even = odd / even

    k = np.arange(1, n_harm + 1, dtype=np.float32)
    roll = float(np.sum(h_norm * (k / n_harm)))

    split = int(max(1, n_harm * 0.25))
    bright = float(np.sum(h_norm[split:]))

    extras = {"odd_even_ratio": odd_even, "rolloff": roll, "brightness": bright}
    return h_norm, extras

def tonalness_and_noise(frame: np.ndarray) -> Tuple[float, float]:
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
    f = standardize_frame(frame, len(frame))

    hvec, extra = harmonic_vector(f, n_harm=n_harm)

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
    return "complex"

def best_for_tags(feat: WTFeatures, motion_std: Optional[WTFeatures]) -> List[str]:
    tags: List[str] = []

    if feat.tonalness > 0.82 and feat.brightness < 0.35 and feat.noise_ratio_db < -14:
        tags += ["bass"]
    if feat.tonalness > 0.70 and feat.brightness >= 0.35 and feat.noise_ratio_db < -10:
        tags += ["lead", "pluck_layer"]
    if feat.tonalness < 0.55 or feat.noise_ratio_db > -10:
        tags += ["air_noise", "fx"]

    if motion_std is not None:
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
        # compat:
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

def get_desc_features(desc: Dict[str, Any]) -> Tuple[List[float], List[float], float]:
    if "features" in desc:
        h = desc["features"].get("harmonics_64")
        e = desc["features"].get("spectral_env_db_128")
        r = desc["features"].get("rms_dbfs")
        if h is not None and e is not None and r is not None:
            return h, e, float(r)

    fm = desc.get("features_mean", {})
    h = fm.get("harmonics")
    e = fm.get("spectral_env_db")
    r = fm.get("rms_dbfs")
    if h is None or e is None or r is None:
        raise KeyError("Descriptor sin features esperadas (ni spec ni compat).")
    return h, e, float(r)

# ----------------------------
# Indexado / procesamiento
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

# ----------------------------
# Matching helpers
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
