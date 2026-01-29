# wtdiag-1.py
# Core del motor (DSP/features/descriptor/matching helpers)
# Nota: se carga dinámicamente desde wtdiag-2.py y wtgui.py (porque el nombre tiene guion).

# ✅ FIX: fallback si el build “rompe” stdlib json
try:
    import json  # stdlib
except ModuleNotFoundError:
    import simplejson as json  # fallback para builds rotos

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, get_window, lfilter, resample_poly, peak_widths

# ----------------------------
# Parámetros estándar offline
# ----------------------------

FEATURE_SR = 48000  # SR "canónico" solo para mapear bandas (comparabilidad del envelope)
TARGET_PEAK = 0.999
COMMON_TABLE_SIZES = (256, 512, 1024, 2048, 4096, 8192)

# ----------------------------
# Modo PRO: formato canónico fijo (para DB y export __CAN__2048x64)
# ----------------------------
PRO_TABLE_SIZE = 2048
PRO_NUM_FRAMES = 64
# SR de salida del WAV canónico (no afecta el análisis: el análisis usa FEATURE_SR)
PRO_SR = FEATURE_SR

# Loop/phase repair (wavetables reales)
LOOP_FIX_FADE = 32          # samples (en tableSize)
LOOP_FIX_VALUE_THR = 0.02   # umbral de discontinuidad (valor)
LOOP_FIX_SLOPE_THR = 0.10   # umbral de discontinuidad (pendiente)
PHASE_ALIGN_COARSE_STEP = 8 # búsqueda rápida de shift circular

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
    p = float(np.max(np.abs(x)) + 1e-12)
    return x * (target_peak / p)

def safe_read_wav(path: Path) -> Tuple[np.ndarray, int]:
    """
    Lee WAV de forma robusta y devuelve (audio_mono_float32, sr).
    Lanza ValueError/IOError con mensajes claros si el archivo es inválido.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el WAV: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"No es un archivo WAV: {p}")

    try:
        x, sr = sf.read(str(p), always_2d=False)
    except Exception as e:
        raise IOError(f"No pude leer WAV '{p}': {e}") from e

    if sr is None:
        raise ValueError(f"WAV '{p}': sampleRate inválido (None).")
    sr_i = int(sr)

    # Rangos típicos para evitar basura/errores (no bloquea casos raros, pero atrapa cosas rotas)
    if sr_i < 4000 or sr_i > 384000:
        raise ValueError(f"WAV '{p}': sampleRate sospechoso: {sr_i} Hz")

    x = np.asarray(x, dtype=np.float32)
    x = to_mono(x)

    if x.size == 0:
        raise ValueError(f"WAV '{p}': audio vacío.")
    if not np.isfinite(x).all():
        # Mejor fallar temprano: NaN/Inf rompe DSP, features y matching.
        raise ValueError(f"WAV '{p}': contiene NaN/Inf (audio corrupto o conversión inválida).")

    return x, sr_i

def ensure_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x.astype(np.float32)
    if len(x) < n:
        out = np.zeros(n, dtype=np.float32)
        out[:len(x)] = x.astype(np.float32)
        return out
    return x[:n].astype(np.float32)

def resample_to_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x.astype(np.float32)
    from fractions import Fraction
    frac = Fraction(n, len(x)).limit_denominator(4096)
    y = resample_poly(x, frac.numerator, frac.denominator)
    return ensure_len(y.astype(np.float32), n)

# ----------------------------
# Reparación de loop / alineación de fase (Paso 0 - pro)
# ----------------------------

def _boundary_mismatch(frame: np.ndarray) -> Tuple[float, float]:
    """Retorna (mismatch_valor, mismatch_pendiente) entre fin→inicio."""
    if len(frame) < 4:
        return 0.0, 0.0
    v = float(abs(frame[0] - frame[-1]))
    s0 = float(frame[1] - frame[0])
    s1 = float(frame[-1] - frame[-2])
    s = float(abs(s0 - s1))
    return v, s

def _shift_cost(frame: np.ndarray) -> float:
    v, s = _boundary_mismatch(frame)
    return v * 1.0 + s * 0.5  # valor pesa más

def best_circular_shift(frame: np.ndarray, coarse_step: int = PHASE_ALIGN_COARSE_STEP, refine_radius: int = 8) -> int:
    """
    Busca un shift circular que minimice el click de loop.
    Estrategia: barrido grueso + refinamiento local.
    """
    N = len(frame)
    if N <= 32:
        costs = [(_shift_cost(np.roll(frame, -s)), s) for s in range(N)]
        return min(costs, key=lambda t: t[0])[1]

    best_s = 0
    best_c = _shift_cost(frame)
    for s in range(0, N, max(1, coarse_step)):
        c = _shift_cost(np.roll(frame, -s))
        if c < best_c:
            best_c, best_s = c, s

    start = max(0, best_s - refine_radius)
    end = min(N - 1, best_s + refine_radius)
    for s in range(start, end + 1):
        c = _shift_cost(np.roll(frame, -s))
        if c < best_c:
            best_c, best_s = c, s

    return int(best_s)

def loop_smooth(frame: np.ndarray, fade_len: int = LOOP_FIX_FADE) -> np.ndarray:
    """
    Suaviza discontinuidad fin→inicio usando mezcla circular en ambas puntas.
    """
    N = len(frame)
    if fade_len <= 1 or N < 2 * fade_len + 4:
        return frame.astype(np.float32)

    out = frame.astype(np.float32).copy()
    for i in range(fade_len):
        t = (i + 1) / (fade_len + 1)
        a = frame[i]
        b = frame[N - fade_len + i]
        out[i] = (t * a) + ((1.0 - t) * b)
        out[N - fade_len + i] = (t * b) + ((1.0 - t) * a)
    return out

def fix_loop_and_phase(frame: np.ndarray) -> np.ndarray:
    """
    1) Alinea fase circular para minimizar click.
    2) Aplica smoothing si el mismatch supera umbrales.
    """
    f = frame.astype(np.float32)
    v, s = _boundary_mismatch(f)
    if v < LOOP_FIX_VALUE_THR and s < LOOP_FIX_SLOPE_THR:
        return f

    shift = best_circular_shift(f)
    f2 = np.roll(f, -shift)

    v2, s2 = _boundary_mismatch(f2)
    if v2 >= LOOP_FIX_VALUE_THR or s2 >= LOOP_FIX_SLOPE_THR:
        f2 = loop_smooth(f2, fade_len=LOOP_FIX_FADE)

    return f2.astype(np.float32)

# ----------------------------
# Normalización obligatoria (Paso 0)
# ----------------------------

def standardize_audio(x: np.ndarray) -> np.ndarray:
    x = remove_dc(x)
    x = normalize_peak(x, target_peak=TARGET_PEAK)
    return x.astype(np.float32)

def standardize_frame(frame: np.ndarray, table_size: int, *, loop_fix: bool = True) -> np.ndarray:
    """
    Estandariza un frame de wavetable:
    - resample a table_size
    - DC offset
    - (opcional) phase align + loop smoothing
    - normalize peak
    """
    f = resample_to_len(frame, table_size)
    f = remove_dc(f)
    if loop_fix:
        f = fix_loop_and_phase(f)
        f = remove_dc(f)
    f = normalize_peak(f, target_peak=TARGET_PEAK)
    return f.astype(np.float32)

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
    lufs: float  # LUFS aprox (ungated)

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
# Detección de layout (single vs multi vs sample) — FIX BUG DIVISORES
# ----------------------------

def infer_layout(n_samples: int, preferred_table: int) -> Tuple[str, int, int]:
    """
    type: "single_cycle" | "multi_frame" | "sample"
    Corrige el bug: si hay múltiples divisores, elige el ts más cercano a preferred_table
    y con numFrames razonable.
    """
    if n_samples in COMMON_TABLE_SIZES:
        return "single_cycle", n_samples, 1

    if abs(n_samples - preferred_table) <= 4:
        return "single_cycle", preferred_table, 1

    candidates: List[Tuple[float, int, int]] = []
    for ts in COMMON_TABLE_SIZES:
        if n_samples % ts != 0:
            continue
        nf = n_samples // ts
        if nf < 2:
            continue

        score = abs(ts - preferred_table) / max(preferred_table, 1)

        if nf < 8:
            score += 2.0
        elif nf < 16:
            score += 1.0
        elif nf > 256:
            score += min(2.0, (nf - 256) / 256.0)

        if ts == preferred_table:
            score -= 0.25

        candidates.append((score, ts, nf))

    if candidates:
        candidates.sort(key=lambda t: t[0])
        _, ts, nf = candidates[0]
        return "multi_frame", int(ts), int(nf)

    # Antes: "unknown"
    return "sample", preferred_table, 1

# ----------------------------
# Export PRO: WAV canónico __CAN__2048x64 (mono, frames concatenados)
# ----------------------------

def _resample_to_sr(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample polyphase (rápido y robusto)."""
    if int(sr_in) == int(sr_out):
        return x.astype(np.float32)
    sr_in = int(sr_in)
    sr_out = int(sr_out)
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = resample_poly(x.astype(np.float32), up, down).astype(np.float32)
    return y

def _interp_frames_count(frames: np.ndarray, n_out: int) -> np.ndarray:
    """Interpola en el eje de frames: (n_in, table) -> (n_out, table)."""
    n_in, table = frames.shape
    if n_in == n_out:
        return frames.astype(np.float32)
    x_src = np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    x_tgt = np.linspace(0.0, 1.0, n_out, dtype=np.float32)

    out = np.empty((n_out, table), dtype=np.float32)
    for s in range(table):
        out[:, s] = np.interp(x_tgt, x_src, frames[:, s]).astype(np.float32)
    return out

def export_canonical_wavetable(
    in_wav: Path,
    out_wav: Path,
    *,
    table_size: int = PRO_TABLE_SIZE,
    num_frames: int = PRO_NUM_FRAMES,
    sr_out: int = PRO_SR,
    loop_fix: bool = True,
    write_sidecar_json: bool = False,
) -> Dict[str, Any]:
    """
    Exporta un WAV 'canónico' mono con frames concatenados:
      - table_size samples por frame
      - num_frames frames
      - sr_out de salida (PCM16)

    Devuelve un meta dict con 'type_original' para que el index pueda decidir si saltar samples.
    """
    in_wav = Path(in_wav)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    x, sr_in = safe_read_wav(in_wav)
    x = standardize_audio(x)
    x = _resample_to_sr(x, int(sr_in), int(sr_out))

    # Inferir layout sobre el audio ya en sr_out (solo para dividir/etiquetar)
    type_original, ts_in, nf_in = infer_layout(int(len(x)), int(table_size))

    frames: List[np.ndarray] = []

    if type_original == "multi_frame" and nf_in > 1 and ts_in > 0 and (ts_in * nf_in) <= len(x):
        raw = [x[i * ts_in : (i + 1) * ts_in] for i in range(nf_in)]
        raw_std = [standardize_frame(fr, int(table_size), loop_fix=loop_fix) for fr in raw]
        fr_np = np.stack(raw_std, axis=0).astype(np.float32)  # (nf_in, table_size)
        fr_np = _interp_frames_count(fr_np, int(num_frames))
        frames = [fr_np[i, :].copy() for i in range(int(num_frames))]

    elif type_original == "single_cycle":
        fr = standardize_frame(x, int(table_size), loop_fix=loop_fix)
        frames = [fr.copy() for _ in range(int(num_frames))]

    else:
        # type_original == "sample" (o cualquier caso raro):
        # tomamos num_frames ventanas a lo largo del audio.
        if len(x) < int(table_size):
            fr = standardize_frame(x, int(table_size), loop_fix=loop_fix)
            frames = [fr.copy() for _ in range(int(num_frames))]
        else:
            max_start = max(0, len(x) - int(table_size))
            starts = np.linspace(0, max_start, int(num_frames), dtype=np.float32)
            for st in starts:
                s0 = int(st)
                seg = x[s0 : s0 + int(table_size)]
                if len(seg) < int(table_size):
                    seg = np.pad(seg, (0, int(table_size) - len(seg)), mode="constant")
                frames.append(standardize_frame(seg, int(table_size), loop_fix=loop_fix))

    can = np.concatenate(frames, axis=0).astype(np.float32)
    sf.write(str(out_wav), can, int(sr_out), subtype="PCM_16")

    meta: Dict[str, Any] = {
        "type_original": str(type_original),
        "sr_in": int(sr_in),
        "sr_out": int(sr_out),
        "table_size": int(table_size),
        "num_frames": int(num_frames),
        "input_samples": int(len(x)),
        "output_samples": int(len(can)),
        "in_wav": str(in_wav),
        "out_wav": str(out_wav),
    }

    if write_sidecar_json:
        side = out_wav.with_suffix(".export_meta.json")
        with open(side, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta

# ----------------------------
# DSP features (ADN, envelope, tono/ruido, LUFS)
# ----------------------------

def log_band_edges(fmin: float, fmax: float, n_bands: int) -> np.ndarray:
    return np.geomspace(fmin, fmax, n_bands + 1)

def bands_centers_hz(sr: int, n_bands: int, fmin: float = 20.0, fmax: float = 20000.0) -> np.ndarray:
    fmax = min(fmax, sr / 2 - 1.0)
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    return np.sqrt(edges[:-1] * edges[1:])

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
        env[i] = -120.0 if idx.size == 0 else float(np.mean(db(mag[idx])))

    return gaussian_filter1d(env, sigma=1.0)

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

    return h_norm, {"odd_even_ratio": odd_even, "rolloff": roll, "brightness": bright}

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

# ---- LUFS aprox (K-weighting simple) ----

def _biquad_highshelf(sr: int, f0: float, gain_db: float, slope: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (f0 / sr)
    cosw0 = math.cos(w0)
    sinw0 = math.sin(w0)
    S = max(1e-6, slope)
    alpha = sinw0 / 2 * math.sqrt((A + 1 / A) * (1 / S - 1) + 2)

    b0 = A * ((A + 1) + (A - 1) * cosw0 + 2 * math.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cosw0)
    b2 = A * ((A + 1) + (A - 1) * cosw0 - 2 * math.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cosw0 + 2 * math.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cosw0)
    a2 = (A + 1) - (A - 1) * cosw0 - 2 * math.sqrt(A) * alpha

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a

def _biquad_highpass(sr: int, f0: float, q: float = 0.707) -> Tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * (f0 / sr)
    cosw0 = math.cos(w0)
    sinw0 = math.sin(w0)
    alpha = sinw0 / (2 * max(1e-6, q))

    b0 = (1 + cosw0) / 2
    b1 = -(1 + cosw0)
    b2 = (1 + cosw0) / 2
    a0 = 1 + alpha
    a1 = -2 * cosw0
    a2 = 1 - alpha

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a

def k_weighted(x: np.ndarray, sr: int) -> np.ndarray:
    x = x.astype(np.float64)
    b_hp, a_hp = _biquad_highpass(sr, 60.0, q=0.707)
    y = lfilter(b_hp, a_hp, x)
    b_sh, a_sh = _biquad_highshelf(sr, 4000.0, gain_db=4.0, slope=1.0)
    y = lfilter(b_sh, a_sh, y)
    return y.astype(np.float32)

def lufs_approx(x: np.ndarray, sr: int = FEATURE_SR) -> float:
    y = k_weighted(x, sr)
    ms = float(np.mean(y * y) + 1e-12)
    return float(-0.691 + 10.0 * math.log10(ms))

def compute_features_for_frame(frame: np.ndarray, n_harm: int, n_bands: int) -> WTFeatures:
    f = standardize_frame(frame, len(frame), loop_fix=True)

    hvec, extra = harmonic_vector(f, n_harm=n_harm)

    N = len(f)
    w = get_window("hann", N, fftbins=True).astype(np.float32)
    X = np.fft.rfft(f * w, n=N)
    mag = np.abs(X).astype(np.float32)
    env = spectral_envelope_db(mag, FEATURE_SR, N, n_bands=n_bands)

    tonal, noise_db = tonalness_and_noise(f)
    cdb = crest_factor_db(f)
    rdb = float(20.0 * math.log10(rms(f) + 1e-12))
    lufs = lufs_approx(f, sr=FEATURE_SR)

    return WTFeatures(
        harmonics=hvec.tolist(),
        spectral_env_db=env.tolist(),
        tonalness=tonal,
        noise_ratio_db=noise_db,
        brightness=float(np.clip(extra["brightness"], 0.0, 1.0)),
        odd_even_ratio=float(extra["odd_even_ratio"]),
        rolloff=float(np.clip(extra["rolloff"], 0.0, 1.0)),
        crest_db=cdb,
        rms_dbfs=rdb,
        lufs=lufs,
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
    lufs = np.array([ff.lufs for ff in frames_feats], dtype=np.float32)

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
        lufs=float(np.mean(lufs)),
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
        lufs=float(np.std(lufs)),
    )
    return mean, std

# ----------------------------
# Selección de tramo estable (para target con ataque)
# ----------------------------

def select_stable_segment(x: np.ndarray, table_size: int, *, hop: Optional[int] = None, eval_bands: int = 48) -> np.ndarray:
    """
    Devuelve un segmento de longitud table_size "estable" dentro de x.
    Score: baja variación espectral (flux) + penalización si energía es muy baja.
    """
    if len(x) <= table_size:
        return ensure_len(x, table_size)

    hop = hop or max(16, table_size // 4)
    n_frames = 1 + (len(x) - table_size) // hop
    if n_frames <= 1:
        start = (len(x) - table_size) // 2
        return x[start:start + table_size].astype(np.float32)

    envs = []
    energies = []
    for i in range(n_frames):
        seg = x[i * hop:i * hop + table_size].astype(np.float32)
        seg = remove_dc(seg)
        seg = normalize_peak(seg, TARGET_PEAK)
        w = get_window("hann", table_size, fftbins=True).astype(np.float32)
        X = np.fft.rfft(seg * w, n=table_size)
        mag = np.abs(X).astype(np.float32)
        env = spectral_envelope_db(mag, FEATURE_SR, table_size, n_bands=eval_bands)
        envs.append(env)
        energies.append(rms(seg))

    envs_np = np.stack(envs, axis=0)
    energies_np = np.array(energies, dtype=np.float32)

    diffs = np.mean((envs_np[1:] - envs_np[:-1]) ** 2, axis=1)
    flux = np.concatenate([[diffs[0]], diffs], axis=0)

    e_db = 20.0 * np.log10(np.maximum(energies_np, 1e-12))
    low_energy_pen = np.clip((-45.0 - e_db) / 10.0, 0.0, 5.0)

    score = flux + 0.5 * low_energy_pen
    best_i = int(np.argmin(score))
    start = best_i * hop
    return x[start:start + table_size].astype(np.float32)

# ----------------------------
# Diagnóstico humano (incluye formant)
# ----------------------------

def _is_formant_like(feat: WTFeatures) -> bool:
    if feat.tonalness < 0.55:
        return False
    if feat.noise_ratio_db > -8:
        return False

    env = np.array(feat.spectral_env_db, dtype=np.float32)
    centers = bands_centers_hz(FEATURE_SR, len(env))
    mask = (centers >= 250.0) & (centers <= 5000.0)
    sub = env[mask]
    if sub.size < 16:
        return False

    sub = sub - np.median(sub)
    peaks, _props = find_peaks(sub, prominence=6.0, distance=max(2, sub.size // 12))
    if len(peaks) < 2:
        return False

    x = np.linspace(0, 1, sub.size, dtype=np.float32)
    slope = float(np.polyfit(x, sub, 1)[0])
    if slope > 8.0:
        return False

    return True

def classify_family(feat: WTFeatures) -> str:
    tonal = feat.tonalness
    bright = feat.brightness
    oe = feat.odd_even_ratio
    noise = feat.noise_ratio_db

    if tonal < 0.35 or noise > -6:
        return "noise"

    if _is_formant_like(feat):
        return "formant"

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
        env_std_mean = float(np.mean(np.abs(np.array(motion_std.spectral_env_db, dtype=np.float32))))
        if motion_std.brightness > 0.08 or env_std_mean > 3.0:
            tags += ["pad"]

    if classify_family(feat) == "formant":
        tags += ["lead", "pad"]

    if not tags:
        tags = ["lead"]

    return sorted(list(set(tags)))

def build_diagnosis(feat: WTFeatures, std: Optional[WTFeatures]) -> WTDiagnosis:
    fam = classify_family(feat)

    pitch_friendly = float(np.clip((feat.tonalness * 1.15) - (max(0.0, (feat.noise_ratio_db + 20) / 40.0)), 0, 1))
    needs_filter = float(np.clip((feat.brightness * 0.8) + (max(0.0, (feat.noise_ratio_db + 30) / 30.0) * 0.6), 0, 1))

    notes_parts: List[str] = []
    if fam == "formant":
        notes_parts.append("Picos tipo formante en el envelope (carácter vocal).")
    if feat.odd_even_ratio > 2.0:
        notes_parts.append("Armónicos impares fuertes (timbre tipo square/clarinet).")
    if feat.brightness > 0.65:
        notes_parts.append("Brillo alto (posible necesidad de lowpass al apilar).")
    if feat.noise_ratio_db > -10:
        notes_parts.append("Componente ruidosa notable.")
    if std is not None:
        env_std_mean = float(np.mean(np.abs(np.array(std.spectral_env_db, dtype=np.float32))))
        if std.brightness > 0.08 or env_std_mean > 3.0:
            notes_parts.append("Multi-frame con movimiento tímbrico apreciable (bueno para pads/morph).")

    if not notes_parts:
        notes_parts.append("Armónicos ordenados, color estable.")

    return WTDiagnosis(
        family=fam,
        best_for=best_for_tags(feat, std),
        pitch_friendly=pitch_friendly,
        needs_filtering=needs_filter,
        notes=" ".join(notes_parts),
    )

# ----------------------------
# Descriptor JSON (spec + compat + aliases de movimiento)
# ----------------------------

def _how_to_use_from_diagnosis(diag: WTDiagnosis, feat: WTFeatures) -> Dict[str, Any]:
    role: List[str] = []
    if "bass" in diag.best_for and feat.tonalness > 0.8 and feat.brightness < 0.45:
        role += ["sub_layer", "main_tone"]
    if "lead" in diag.best_for:
        role += ["main_tone", "unison_layer"]
    if "pad" in diag.best_for:
        role += ["pad_morph", "harmonic_bed"]
    if "air_noise" in diag.best_for or diag.family == "noise":
        role += ["texture_layer", "attack_residual"]

    if not role:
        role = ["main_tone"]

    if feat.brightness > 0.7:
        rec_filter = "lowpass gentle if stacking"
    elif feat.noise_ratio_db > -10:
        rec_filter = "bandpass/hi-shelf depending on layer"
    else:
        rec_filter = "none"

    if "sub_layer" in role:
        gain = [-9, -3]
    elif "main_tone" in role:
        gain = [-12, -3]
    else:
        gain = [-18, -6]

    return {
        "layer_role": sorted(list(set(role))),
        "recommended_filter": rec_filter,
        "recommended_gain_db_range": gain,
    }

def descriptor_to_spec(desc: WTDescriptor) -> Dict[str, Any]:
    feat = desc.features_mean
    spec: Dict[str, Any] = {
        "path": desc.path,
        "type": desc.type,  # single_cycle | multi_frame | sample
        "tableSize": desc.tableSize,
        "numFrames": desc.numFrames,
        "sampleRate": desc.sampleRate,

        # NUEVO: auditable (el SR canónico con el que se mapearon bandas/LUFS)
        "feature_sr": FEATURE_SR,

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
            "lufs": feat.lufs,
        },
        "diagnosis": asdict(desc.diagnosis),
        "how_to_use": _how_to_use_from_diagnosis(desc.diagnosis, feat),

        # compat:
        "features_mean": asdict(desc.features_mean),
        "features_std": asdict(desc.features_std) if desc.features_std is not None else None,
    }

    # Aliases EXACTOS para movimiento (multi-frame)
    if desc.features_std is not None:
        std = desc.features_std
        spec["features"]["frame_brightness_mean"] = feat.brightness
        spec["features"]["frame_brightness_std"] = std.brightness

        spec["features"]["harmonics_mean"] = feat.harmonics
        spec["features"]["harmonics_std"] = std.harmonics

        spec["features"]["spectral_env_db_mean"] = feat.spectral_env_db
        spec["features"]["spectral_env_db_std"] = std.spectral_env_db

        spec["features_std_spec"] = {
            "harmonics_std": std.harmonics,
            "spectral_env_db_std": std.spectral_env_db,
            "brightness_std": std.brightness,
        }

    return spec

def get_desc_features(desc: Dict[str, Any]) -> Tuple[List[float], List[float], float, float]:
    """
    Retorna: harmonics_64, spectral_env_db_128, rms_dbfs, lufs
    """
    if "features" in desc:
        h = desc["features"].get("harmonics_64")
        e = desc["features"].get("spectral_env_db_128")
        r = desc["features"].get("rms_dbfs")
        l = desc["features"].get("lufs")
        if h is not None and e is not None and r is not None:
            return h, e, float(r), float(l if l is not None else r)

    fm = desc.get("features_mean", {})
    h = fm.get("harmonics")
    e = fm.get("spectral_env_db")
    r = fm.get("rms_dbfs")
    l = fm.get("lufs")
    if h is None or e is None or r is None:
        raise KeyError("Descriptor sin features esperadas (ni spec ni compat).")
    return h, e, float(r), float(l if l is not None else r)

# ----------------------------
# Indexado / procesamiento
# ----------------------------

def process_wavetable(path: Path, table_size: int, n_harm: int, n_bands: int) -> WTDescriptor:
    x, sr = safe_read_wav(path)
    x = standardize_audio(x)

    wtype, in_table, num_frames = infer_layout(len(x), table_size)

    if wtype == "single_cycle":
        frame = standardize_frame(x[:in_table], table_size, loop_fix=True)
        frames = frame.reshape(1, -1)
        num_frames = 1

    elif wtype == "multi_frame":
        x = ensure_len(x, in_table * num_frames)
        raw_frames = x.reshape(num_frames, in_table)
        frames = np.stack(
            [standardize_frame(raw_frames[i], table_size, loop_fix=True) for i in range(num_frames)],
            axis=0
        )

    else:
        # type="sample": se sigue extrayendo features (útil para diagnosticar),
        # pero el index/match deberían decidir si incluirlo o no.
        frame = standardize_frame(x[:min(len(x), table_size)], table_size, loop_fix=True)
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
    return np.clip(d, -limit_db, limit_db)

def _interp_center_freq(centers: np.ndarray, x: float) -> float:
    # Interp simple index-fraccional -> Hz usando centers
    idx = np.arange(len(centers), dtype=np.float32)
    return float(np.interp(np.float32(x), idx, centers.astype(np.float32)))

def delta_to_peq(delta_db: np.ndarray, sr: int, max_filters: int = 6, min_sep_bands: int = 6) -> List[Dict[str, float]]:
    """
    Convierte delta_env_db (en bandas log) a una lista de filtros paramétricos aproximados:
      - low_shelf / high_shelf (para "tilt" global)
      - peaking con Q variable (según ancho del pico/valle)

    Robusto ante datos patológicos: NaN/Inf, pocos puntos válidos, polyfit inestable.
    """
    d = np.asarray(delta_db, dtype=np.float32).copy()
    n = int(d.size)
    if n < 8 or max_filters <= 0:
        return []

    # Si viene basura (NaN/Inf), usamos solo los puntos finitos.
    finite = np.isfinite(d)
    if not bool(np.any(finite)):
        return []

    centers = bands_centers_hz(sr, n)
    logf = np.log10(np.maximum(centers, 1.0)).astype(np.float32)

    filters: List[Dict[str, float]] = []

    # 1) Tilt global: fit delta ~ a*logf + b (solo puntos válidos)
    #    Si no hay suficientes puntos o polyfit falla, se salta el tilt.
    TILT_THR_DB = 3.0
    MAX_SHELF_GAIN_DB = 12.0  # límite extra (además del clip previo del delta)
    residual = d.copy()

    valid_idx = np.where(finite)[0]
    if valid_idx.size >= 6:
        try:
            a, b = np.polyfit(logf[valid_idx].astype(np.float64), d[valid_idx].astype(np.float64), 1)
            a = float(a)
            b = float(b)
            pred = (a * logf + b).astype(np.float32)

            # dB high - low (aprox). Usa extremos finitos si existen, si no extremos del vector.
            lo_i = int(valid_idx[0])
            hi_i = int(valid_idx[-1])
            tilt_gain = float(pred[hi_i] - pred[lo_i])

            # Evita valores absurdos por ajustes raros
            tilt_gain = float(np.clip(tilt_gain, -MAX_SHELF_GAIN_DB, MAX_SHELF_GAIN_DB))

            if abs(tilt_gain) >= TILT_THR_DB and max_filters >= 1:
                if tilt_gain > 0:
                    f0 = float(centers[int(np.clip(int(0.60 * (n - 1)), 0, n - 1))])
                    filters.append({"type": "high_shelf", "f0_hz": f0, "gain_db": tilt_gain, "slope": 1.0})
                else:
                    f0 = float(centers[int(np.clip(int(0.40 * (n - 1)), 0, n - 1))])
                    filters.append({"type": "low_shelf", "f0_hz": f0, "gain_db": float(abs(tilt_gain)), "slope": 1.0})

                residual = (d - pred).astype(np.float32)
        except Exception:
            # polyfit puede fallar si hay degeneración numérica; no pasa nada, seguimos sin tilt.
            pass

    remaining = max(0, max_filters - len(filters))
    if remaining <= 0:
        return filters

    # 2) Picos/valleys en residual
    MIN_PROM_DB = 1.0
    MIN_GAIN_DB = 0.8
    MAX_PEAK_GAIN_DB = 12.0  # límite extra

    # Asegura finitud en residual para find_peaks
    residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    peaks, props_p = find_peaks(residual, prominence=MIN_PROM_DB, distance=max(1, int(min_sep_bands)))
    valleys, props_v = find_peaks(-residual, prominence=MIN_PROM_DB, distance=max(1, int(min_sep_bands)))

    candidates: List[Tuple[float, int, str, float]] = []
    for i, p in enumerate(peaks):
        prom = float(props_p["prominences"][i]) if "prominences" in props_p else float(abs(residual[p]))
        gain = float(residual[p])
        candidates.append((abs(gain) + 0.25 * prom, int(p), "peaking", gain))

    for i, v in enumerate(valleys):
        prom = float(props_v["prominences"][i]) if "prominences" in props_v else float(abs(residual[v]))
        gain = float(residual[v])  # negativo
        candidates.append((abs(gain) + 0.25 * prom, int(v), "peaking", gain))

    candidates.sort(key=lambda t: t[0], reverse=True)

    chosen_idx: List[int] = []
    for _score, idx, _typ, gain in candidates:
        if len(chosen_idx) >= remaining:
            break
        if abs(gain) < MIN_GAIN_DB:
            continue
        if any(abs(idx - c) < min_sep_bands for c in chosen_idx):
            continue
        chosen_idx.append(idx)

    if not chosen_idx:
        return filters

    for idx in chosen_idx:
        gain = float(residual[idx])
        if abs(gain) < MIN_GAIN_DB:
            continue

        # límite adicional
        gain = float(np.clip(gain, -MAX_PEAK_GAIN_DB, MAX_PEAK_GAIN_DB))

        try:
            if gain >= 0:
                w_res = peak_widths(residual, peaks=np.array([idx]), rel_height=0.5)
            else:
                w_res = peak_widths(-residual, peaks=np.array([idx]), rel_height=0.5)

            left_ip = float(w_res[2][0])   # left_ips
            right_ip = float(w_res[3][0])  # right_ips

            f0 = float(centers[idx])
            f_left = _interp_center_freq(centers, left_ip)
            f_right = _interp_center_freq(centers, right_ip)
            bw = max(1e-3, float(abs(f_right - f_left)))

            q = float(np.clip(f0 / bw, 0.3, 12.0))
        except Exception:
            # Fallback razonable
            f0 = float(centers[idx])
            q = 1.2

        filters.append({"type": "peaking", "f0_hz": f0, "gain_db": gain, "q": q})

        if len(filters) >= max_filters:
            break

    return filters[:max_filters]

def load_descriptor(path: Path) -> Dict[str, Any]:
    """
    Carga descriptor JSON con validaciones mínimas + checks de compatibilidad.

    - Valida que el JSON sea parseable y tenga un schema mínimo.
    - Verifica feature_sr (si existe) contra FEATURE_SR del motor:
        si no coincide -> se ignora (lanza ValueError) para evitar matches raros.
    - Valida que harmonics/env sean listas numéricas y finitas.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe descriptor: {p}")
    if not p.is_file():
        raise IsADirectoryError(f"No es un archivo descriptor: {p}")

    try:
        with open(p, "r", encoding="utf-8") as f:
            desc = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Descriptor JSON inválido '{p}': {e}") from e
    except Exception as e:
        raise IOError(f"No pude leer descriptor '{p}': {e}") from e

    if not isinstance(desc, dict):
        raise ValueError(f"Descriptor '{p}': el JSON raíz debe ser un objeto/dict.")

    # --- schema mínimo ---
    for k in ("type", "tableSize"):
        if k not in desc:
            raise ValueError(f"Descriptor '{p}': falta key obligatoria '{k}'.")

    # Compat-check: SR canónico usado para mapear bandas
    fs = desc.get("feature_sr", None)
    if fs is not None:
        try:
            fs_i = int(fs)
        except Exception:
            raise ValueError(f"Descriptor '{p}': feature_sr inválido: {fs!r}")
        if fs_i != int(FEATURE_SR):
            raise ValueError(
                f"Descriptor '{p}': feature_sr={fs_i} != FEATURE_SR={FEATURE_SR}. "
                "Reindexa la DB con los mismos parámetros."
            )

    # Extrae features (soporta spec nuevo y compat viejo)
    try:
        h, e, r, l = get_desc_features(desc)
    except Exception as ex:
        raise ValueError(f"Descriptor '{p}': no pude extraer features: {ex}") from ex

    # Validaciones de tipo/longitud
    if not isinstance(h, list) or not isinstance(e, list):
        raise ValueError(f"Descriptor '{p}': harmonics/env deben ser listas.")
    if len(h) < 8 or len(e) < 8:
        raise ValueError(f"Descriptor '{p}': harmonics/env demasiado cortos (h={len(h)}, env={len(e)}).")

    # Validaciones numéricas
    h_np = np.asarray(h, dtype=np.float32)
    e_np = np.asarray(e, dtype=np.float32)

    if not np.isfinite(h_np).all() or not np.isfinite(e_np).all():
        raise ValueError(f"Descriptor '{p}': harmonics/env contienen NaN/Inf.")

    # Validación ligera de ganancia (evita strings/None)
    try:
        _ = float(r)
        _ = float(l)
    except Exception:
        raise ValueError(f"Descriptor '{p}': rms_dbfs/lufs inválidos (no numéricos).")

    return desc

