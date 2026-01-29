# wtgui.py
import threading
import queue
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

# ✅ NUEVO: capturar stdout/stderr del motor (print())
import io
import contextlib

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Importa el motor (tu wtdiag.py)
import wtdiag


class TkTextHandler(logging.Handler):
    """Handler de logging que manda logs a un Text via cola (thread-safe)."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        try:
            msg = self.format(record)
            self.q.put(msg)
        except Exception:
            pass


def open_in_file_manager(path: Path):
    """Abre carpeta/archivo en el explorador del SO."""
    p = str(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(p)  # noqa
        elif sys.platform == "darwin":
            subprocess.run(["open", p], check=False)
        else:
            subprocess.run(["xdg-open", p], check=False)
    except Exception:
        parent = str(path if path.is_dir() else path.parent)
        try:
            if sys.platform.startswith("win"):
                os.startfile(parent)  # noqa
            elif sys.platform == "darwin":
                subprocess.run(["open", parent], check=False)
            else:
                subprocess.run(["xdg-open", parent], check=False)
        except Exception:
            pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WT Diagnoser - Offline")
        self.geometry("980x740")

        # Flags de cierre/cancelación
        self._closing = False
        self._poll_after_id = None
        self.cancel_requested = False

        self.log_queue = queue.Queue()

        # Vars comunes
        # ✅ PRO estandarizado SIEMPRE: tableSize fijo 2048
        self.table_size = tk.IntVar(value=2048)
        self.harmonics = tk.IntVar(value=64)
        self.bands = tk.IntVar(value=128)

        # Vars Index
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()

        # Vars Diag
        self.diag_wav = tk.StringVar()
        self.diag_out = tk.StringVar()

        # Vars Match
        self.match_db = tk.StringVar()
        self.match_target = tk.StringVar()
        self.match_out = tk.StringVar()
        self.match_topk = tk.IntVar(value=15)
        self.match_eq_limit = tk.DoubleVar(value=6.0)
        self.match_eq_smooth = tk.DoubleVar(value=1.5)
        self.match_include_samples = tk.BooleanVar(value=False)

        # parámetros match (UI)
        self.match_topn = tk.IntVar(value=10)
        self.match_gain_mode = tk.StringVar(value="rms")  # "rms" | "lufs"
        self.match_max_filters = tk.IntVar(value=6)
        self.match_min_sep_bands = tk.IntVar(value=6)
        self.match_w_harm = tk.DoubleVar(value=1.0)
        self.match_w_eq = tk.DoubleVar(value=0.8)

        # refs UI que necesitamos habilitar/deshabilitar
        self._spin_table_size: Optional[ttk.Spinbox] = None

        self._build_ui()
        self._setup_logging()

        # ✅ cierre limpio
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ✅ PRO fijo: bloquear tableSize en 2048
        self._apply_pro_ui_state()

        # Arranca polling de logs
        self._poll_logs()

    # ---------------- UI ----------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Opciones comunes (arriba)
        opt = ttk.LabelFrame(self, text="Parámetros (comunes)")
        opt.pack(fill="x", **pad)

        ttk.Label(opt, text="tableSize").grid(row=0, column=0, sticky="w")
        self._spin_table_size = ttk.Spinbox(
            opt, from_=256, to=8192, increment=256, textvariable=self.table_size, width=10
        )
        self._spin_table_size.grid(row=0, column=1, padx=6)

        ttk.Label(opt, text="harmonics").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(opt, from_=16, to=256, increment=16, textvariable=self.harmonics, width=10)\
            .grid(row=0, column=3, padx=6)

        ttk.Label(opt, text="bands").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(opt, from_=32, to=256, increment=32, textvariable=self.bands, width=10)\
            .grid(row=0, column=5, padx=6)

        # ✅ Nota visible: PRO fijo
        ttk.Label(opt, text="PRO estandarizado: Index exporta __CAN__2048x64 (WAV+JSON) en el Output.").grid(
            row=1, column=0, columnspan=6, sticky="w", pady=(6, 0)
        )

        opt.columnconfigure(6, weight=1)

        # Notebook con 3 tabs
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="x", **pad)

        self.tab_index = ttk.Frame(self.nb)
        self.tab_diag = ttk.Frame(self.nb)
        self.tab_match = ttk.Frame(self.nb)

        self.nb.add(self.tab_index, text="Indexar DB")
        self.nb.add(self.tab_diag, text="Diag 1 WAV")
        self.nb.add(self.tab_match, text="Match target")

        self._build_tab_index(self.tab_index)
        self._build_tab_diag(self.tab_diag)
        self._build_tab_match(self.tab_match)

        # Barra acciones + progreso (global)
        act = ttk.Frame(self)
        act.pack(fill="x", **pad)

        self.btn_open_output = ttk.Button(act, text="Abrir salida (según tab)", command=self._open_current_output)
        self.btn_open_output.pack(side="left")

        # botón Cancel
        self.btn_cancel = ttk.Button(act, text="Cancelar", command=self._request_cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=10)

        self.progress = ttk.Progressbar(act, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True)

        # Log box
        logfrm = ttk.Frame(self)
        logfrm.pack(fill="both", expand=True, **pad)

        ttk.Label(logfrm, text="Logs (detallados):").pack(anchor="w")
        self.txt = tk.Text(logfrm, wrap="word")
        self.txt.pack(fill="both", expand=True)

    def _build_tab_index(self, parent: ttk.Frame):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(parent)
        frm.pack(fill="x", **pad)

        ttk.Label(frm, text="Carpeta wavetables (entrada):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.in_dir, width=72).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir...", command=self._pick_in).grid(row=0, column=2)

        # ✅ Un solo Output (DB + WAV CAN + JSON CAN + _INDEX.json)
        ttk.Label(frm, text="Output (DB + WAV __CAN + JSON __CAN + _INDEX.json):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_dir, width=72).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir...", command=self._pick_out).grid(row=1, column=2)

        frm.columnconfigure(1, weight=1)

        info = ttk.LabelFrame(parent, text="Modo PRO (siempre activo)")
        info.pack(fill="x", **pad)
        ttk.Label(
            info,
            text="Index siempre exporta: base__CAN__2048x64.wav + base__CAN__2048x64.json en el Output.\n"
                 "No se usa carpeta separada para WAV; todo queda junto."
        ).pack(anchor="w", padx=8, pady=6)

        runfrm = ttk.Frame(parent)
        runfrm.pack(fill="x", **pad)

        self.btn_run_index = ttk.Button(runfrm, text="RUN (Indexar)", command=self._run_index)
        self.btn_run_index.pack(side="left")

        ttk.Label(runfrm, text="Genera: _INDEX.json + WAV/JSON canónicos (__CAN) en el Output.").pack(side="left", padx=12)

    def _build_tab_diag(self, parent: ttk.Frame):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(parent)
        frm.pack(fill="x", **pad)

        ttk.Label(frm, text="WAV (wavetable):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.diag_wav, width=72).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir...", command=self._pick_diag_wav).grid(row=0, column=2)

        ttk.Label(frm, text="Salida JSON (descriptor):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.diag_out, width=72).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Guardar como...", command=self._pick_diag_out).grid(row=1, column=2)

        frm.columnconfigure(1, weight=1)

        runfrm = ttk.Frame(parent)
        runfrm.pack(fill="x", **pad)

        self.btn_run_diag = ttk.Button(runfrm, text="RUN (Diag)", command=self._run_diag)
        self.btn_run_diag.pack(side="left")

        ttk.Label(runfrm, text="Diagnostica 1 wav y exporta JSON spec.").pack(side="left", padx=12)

    def _build_tab_match(self, parent: ttk.Frame):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(parent)
        frm.pack(fill="x", **pad)

        ttk.Label(frm, text="DB (carpeta con _INDEX.json):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.match_db, width=72).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir...", command=self._pick_match_db).grid(row=0, column=2)

        ttk.Label(frm, text="Target WAV (layer):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.match_target, width=72).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir...", command=self._pick_match_target).grid(row=1, column=2)

        ttk.Label(frm, text="Salida Report (JSON):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.match_out, width=72).grid(row=2, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Guardar como...", command=self._pick_match_out).grid(row=2, column=2)

        frm.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(parent, text="Parámetros Match")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="topK").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(opts, from_=3, to=200, increment=1, textvariable=self.match_topk, width=10)\
            .grid(row=0, column=1, padx=6)

        ttk.Label(opts, text="topN").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(opts, from_=1, to=50, increment=1, textvariable=self.match_topn, width=10)\
            .grid(row=0, column=3, padx=6)

        ttk.Label(opts, text="Gain mode").grid(row=0, column=4, sticky="w")
        ttk.Combobox(opts, textvariable=self.match_gain_mode, values=["rms", "lufs"], width=8, state="readonly")\
            .grid(row=0, column=5, padx=6)

        ttk.Checkbutton(opts, text="Include samples (type=sample)", variable=self.match_include_samples)\
            .grid(row=0, column=6, padx=10, sticky="w")

        ttk.Label(opts, text="EQ limit (dB)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(opts, from_=1.0, to=18.0, increment=0.5, textvariable=self.match_eq_limit, width=10)\
            .grid(row=1, column=1, padx=6)

        ttk.Label(opts, text="EQ smooth").grid(row=1, column=2, sticky="w")
        ttk.Spinbox(opts, from_=0.2, to=6.0, increment=0.1, textvariable=self.match_eq_smooth, width=10)\
            .grid(row=1, column=3, padx=6)

        ttk.Label(opts, text="maxFilters").grid(row=1, column=4, sticky="w")
        ttk.Spinbox(opts, from_=0, to=20, increment=1, textvariable=self.match_max_filters, width=10)\
            .grid(row=1, column=5, padx=6)

        ttk.Label(opts, text="minSepBands").grid(row=1, column=6, sticky="w")
        ttk.Spinbox(opts, from_=1, to=32, increment=1, textvariable=self.match_min_sep_bands, width=10)\
            .grid(row=1, column=7, padx=6)

        ttk.Label(opts, text="w_harm").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(opts, from_=0.0, to=5.0, increment=0.1, textvariable=self.match_w_harm, width=10)\
            .grid(row=2, column=1, padx=6)

        ttk.Label(opts, text="w_eq").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(opts, from_=0.0, to=5.0, increment=0.1, textvariable=self.match_w_eq, width=10)\
            .grid(row=2, column=3, padx=6)

        opts.columnconfigure(8, weight=1)

        runfrm = ttk.Frame(parent)
        runfrm.pack(fill="x", **pad)

        self.btn_run_match = ttk.Button(runfrm, text="RUN (Match)", command=self._run_match)
        self.btn_run_match.pack(side="left")

        ttk.Label(runfrm, text="Genera report JSON + .txt con filtros PEQ aproximados.").pack(side="left", padx=12)

    # ✅ PRO fijo: tableSize = 2048 y spinbox disabled
    def _apply_pro_ui_state(self):
        try:
            self.table_size.set(2048)
        except Exception:
            pass
        if self._spin_table_size is not None:
            try:
                self._spin_table_size.config(state="disabled")
            except Exception:
                pass

    # ---------------- Logging ----------------

    def _setup_logging(self):
        self.logger = logging.getLogger("wtgui")
        # ✅ CAMBIO: DEBUG global
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        ui_handler = TkTextHandler(self.log_queue)
        ui_handler.setFormatter(fmt)
        # (no fijamos nivel del handler -> hereda DEBUG del logger)
        self.logger.addHandler(ui_handler)

    def _ensure_file_logger(self, out_dir: Path):
        """Crea (o recrea) un logger a archivo dentro del output folder."""
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)

        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "wt_gui.log", encoding="utf-8")
        # ✅ CAMBIO: DEBUG a archivo
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(fh)

    # ✅ NUEVO: capturar print() del motor y mandarlo a la GUI + log file
    def _run_with_captured_stdio(self, fn, *args, **kwargs):
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            rc = fn(*args, **kwargs)

        out = buf_out.getvalue().splitlines()
        err = buf_err.getvalue().splitlines()

        for line in out:
            self.logger.info(f"[ENGINE] {line}")
        for line in err:
            self.logger.error(f"[ENGINE:stderr] {line}")

        return rc

    # ---------------- Pickers ----------------

    def _pick_in(self):
        p = filedialog.askdirectory(title="Elige carpeta de wavetables")
        if p:
            self.in_dir.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory(title="Elige Output (DB + WAV/JSON __CAN + _INDEX.json)")
        if p:
            self.out_dir.set(p)

    def _pick_diag_wav(self):
        p = filedialog.askopenfilename(
            title="Elige WAV (wavetable)",
            filetypes=[("WAV", "*.wav"), ("All files", "*.*")]
        )
        if p:
            self.diag_wav.set(p)
            suggested = Path(p).with_suffix(".descriptor.json")
            self.diag_out.set(str(suggested))

    def _pick_diag_out(self):
        p = filedialog.asksaveasfilename(
            title="Guardar descriptor JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if p:
            self.diag_out.set(p)

    def _pick_match_db(self):
        p = filedialog.askdirectory(title="Elige carpeta DB (con _INDEX.json)")
        if p:
            self.match_db.set(p)
            suggested = Path(p) / "match_report.json"
            self.match_out.set(str(suggested))

    def _pick_match_target(self):
        p = filedialog.askopenfilename(
            title="Elige target WAV (layer)",
            filetypes=[("WAV", "*.wav"), ("All files", "*.*")]
        )
        if p:
            self.match_target.set(p)

    def _pick_match_out(self):
        p = filedialog.asksaveasfilename(
            title="Guardar report JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if p:
            self.match_out.set(p)

    # ---------------- Helpers ----------------

    def _on_close(self):
        """Cierre limpio: cancela after() del polling y destruye la ventana."""
        self._closing = True
        try:
            if self._poll_after_id is not None:
                self.after_cancel(self._poll_after_id)
                self._poll_after_id = None
        except Exception:
            pass

        if getattr(self.btn_cancel, "state", None) != "disabled":
            self.cancel_requested = True

        try:
            self.destroy()
        except Exception:
            pass

    def _request_cancel(self):
        """Cancelación suave: marca flag (no mata thread)."""
        if not self.cancel_requested:
            self.cancel_requested = True
            self.logger.warning("CANCEL REQUESTED: se intentará detener al terminar la etapa actual.")

    def _current_output_path(self) -> Optional[Path]:
        tab = self.nb.index("current")
        if tab == 0:
            p = self.out_dir.get().strip()
            return Path(p) if p else None
        if tab == 1:
            p = self.diag_out.get().strip()
            return Path(p) if p else None
        if tab == 2:
            p = self.match_out.get().strip()
            return Path(p) if p else None
        return None

    def _open_current_output(self):
        p = self._current_output_path()
        if not p:
            return
        try:
            if p.suffix.lower() in (".json", ".txt", ".wav"):
                open_in_file_manager(p.parent)
            else:
                open_in_file_manager(p)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _set_running(self, running: bool):
        state = "disabled" if running else "normal"
        self.btn_run_index.config(state=state)
        self.btn_run_diag.config(state=state)
        self.btn_run_match.config(state=state)

        self.btn_cancel.config(state=("normal" if running else "disabled"))

        if running:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _common_args(self):
        # table_size siempre 2048 (PRO fijo)
        return int(self.table_size.get()), int(self.harmonics.get()), int(self.bands.get())

    # ---------------- Run: Index ----------------

    def _run_index(self):
        in_p = Path(self.in_dir.get().strip())
        out_p = Path(self.out_dir.get().strip())

        if not in_p.exists():
            messagebox.showerror("Falta entrada", "Elige una carpeta válida de wavetables.")
            return
        if not str(out_p).strip():
            messagebox.showerror("Falta salida", "Elige un Output válido.")
            return

        self.cancel_requested = False
        self._ensure_file_logger(out_p)
        self._set_running(True)

        # ✅ PRO siempre activo
        self.table_size.set(2048)
        self._apply_pro_ui_state()

        ts, harm, bands = self._common_args()
        self.logger.info("== INDEX START ==")
        self.logger.info(f"Entrada: {in_p}")
        self.logger.info(f"Salida:  {out_p}")
        self.logger.info(f"Params: tableSize={ts} harmonics={harm} bands={bands}")
        self.logger.info("PRO: True (export __CAN__2048x64 PCM16, WAV+JSON en Output)")

        th = threading.Thread(target=self._worker_index, args=(in_p, out_p, ts, harm, bands), daemon=True)
        th.start()

    def _worker_index(self, in_p: Path, out_p: Path, ts: int, harm: int, bands: int):
        try:
            if self.cancel_requested:
                self.logger.warning("Index cancelado antes de iniciar.")
                return

            # ✅ forzar por seguridad
            ts = 2048

            args = SimpleNamespace(
                in_dir=str(in_p),
                out_dir=str(out_p),
                table_size=int(ts),
                harmonics=int(harm),
                bands=int(bands),
                # PRO fijo:
                pro=True,
                # NO pro_wav_dir (ya no existe)
            )

            # ✅ CAMBIO: capturar print() del motor
            rc = self._run_with_captured_stdio(wtdiag.cmd_index, args)

            if self.cancel_requested:
                self.logger.warning("Index terminó, pero hubo CANCEL REQUESTED (no se pudo interrumpir a mitad).")

            # ✅ NUEVO: resumen real de outputs en carpeta
            try:
                jsons = list(out_p.glob("*.json"))
                wavs = list(out_p.glob("*.wav"))
                self.logger.info(f"Outputs en {out_p}: json={len(jsons)} wav={len(wavs)}")
            except Exception as e:
                self.logger.debug(f"No pude contar outputs: {type(e).__name__}: {e}")

            # ✅ NUEVO (opcional): leer _INDEX.json y reportar kept/failed/skipped
            try:
                idx_path = out_p / "_INDEX.json"
                if idx_path.exists():
                    import json as _json_stdlib
                    with open(idx_path, "r", encoding="utf-8") as f:
                        idx = _json_stdlib.load(f)
                    self.logger.info(
                        "INDEX summary: "
                        f"kept={idx.get('kept')} failed_files={idx.get('failed_files')} "
                        f"skipped_samples={idx.get('skipped_samples')} processed={idx.get('processed')}"
                    )
            except Exception as e:
                self.logger.debug(f"No pude leer _INDEX.json: {type(e).__name__}: {e}")

            if rc == 0:
                self.logger.info("Indexado OK. Se generó _INDEX.json y WAV/JSON __CAN.")
                self.logger.info(f"Log guardado en: {out_p / 'wt_gui.log'}")
            else:
                self.logger.error(f"Indexado terminó con código {rc}. Revisa logs.")
        except Exception as e:
            self.logger.exception(f"Fallo inesperado: {e}")
        finally:
            self.logger.info("== INDEX END ==")
            if not self._closing:
                self.after(0, lambda: self._set_running(False))

    # ---------------- Run: Diag ----------------

    def _run_diag(self):
        wav_p = Path(self.diag_wav.get().strip())
        out_p = Path(self.diag_out.get().strip())

        if not wav_p.exists():
            messagebox.showerror("Falta WAV", "Elige un archivo WAV válido.")
            return
        if not str(out_p).strip():
            messagebox.showerror("Falta salida", "Elige dónde guardar el JSON del descriptor.")
            return

        self.cancel_requested = False
        self._ensure_file_logger(out_p.parent)
        self._set_running(True)

        # tableSize fijo
        self.table_size.set(2048)
        self._apply_pro_ui_state()

        ts, harm, bands = self._common_args()
        self.logger.info("== DIAG START ==")
        self.logger.info(f"WAV:   {wav_p}")
        self.logger.info(f"Salida:{out_p}")
        self.logger.info(f"Params: tableSize={ts} harmonics={harm} bands={bands}")

        th = threading.Thread(target=self._worker_diag, args=(wav_p, out_p, ts, harm, bands), daemon=True)
        th.start()

    def _worker_diag(self, wav_p: Path, out_p: Path, ts: int, harm: int, bands: int):
        try:
            if self.cancel_requested:
                self.logger.warning("Diag cancelado antes de iniciar.")
                return

            args = SimpleNamespace(
                wav=str(wav_p),
                out=str(out_p),
                table_size=int(ts),
                harmonics=int(harm),
                bands=int(bands),
            )

            # ✅ CAMBIO: capturar print() del motor
            rc = self._run_with_captured_stdio(wtdiag.cmd_diag, args)

            if self.cancel_requested:
                self.logger.warning("Diag terminó, pero hubo CANCEL REQUESTED (no se pudo interrumpir a mitad).")

            if rc == 0:
                self.logger.info("Diag OK. Se generó descriptor JSON.")
                self.logger.info(f"Log guardado en: {out_p.parent / 'wt_gui.log'}")
            else:
                self.logger.error(f"Diag terminó con código {rc}. Revisa logs.")
        except Exception as e:
            self.logger.exception(f"Fallo inesperado: {e}")
        finally:
            self.logger.info("== DIAG END ==")
            if not self._closing:
                self.after(0, lambda: self._set_running(False))

    # ---------------- Run: Match ----------------

    def _run_match(self):
        db_p = Path(self.match_db.get().strip())
        target_p = Path(self.match_target.get().strip())
        out_p = Path(self.match_out.get().strip())

        if not db_p.exists():
            messagebox.showerror("Falta DB", "Elige una carpeta DB válida (debe contener _INDEX.json).")
            return
        if not (db_p / "_INDEX.json").exists():
            messagebox.showerror("DB inválida", "No se encontró _INDEX.json en la carpeta DB.")
            return
        if not target_p.exists():
            messagebox.showerror("Falta target", "Elige un WAV target válido.")
            return
        if not str(out_p).strip():
            messagebox.showerror("Falta salida", "Elige dónde guardar el report JSON.")
            return

        self.cancel_requested = False
        self._ensure_file_logger(out_p.parent)
        self._set_running(True)

        # tableSize fijo
        self.table_size.set(2048)
        self._apply_pro_ui_state()

        ts, harm, bands = self._common_args()
        self.logger.info("== MATCH START ==")
        self.logger.info(f"DB:     {db_p}")
        self.logger.info(f"Target: {target_p}")
        self.logger.info(f"Salida: {out_p}")
        self.logger.info(
            "Params: "
            f"tableSize={ts} harmonics={harm} bands={bands} "
            f"topk={self.match_topk.get()} topn={self.match_topn.get()} "
            f"gain_mode={self.match_gain_mode.get()} "
            f"eq_limit_db={self.match_eq_limit.get()} eq_smooth={self.match_eq_smooth.get()} "
            f"max_filters={self.match_max_filters.get()} min_sep_bands={self.match_min_sep_bands.get()} "
            f"w_harm={self.match_w_harm.get()} w_eq={self.match_w_eq.get()} "
            f"include_samples={bool(self.match_include_samples.get())}"
        )

        th = threading.Thread(
            target=self._worker_match,
            args=(db_p, target_p, out_p, ts, harm, bands),
            daemon=True
        )
        th.start()

    def _worker_match(self, db_p: Path, target_p: Path, out_p: Path, ts: int, harm: int, bands: int):
        try:
            if self.cancel_requested:
                self.logger.warning("Match cancelado antes de iniciar.")
                return

            args = SimpleNamespace(
                db_dir=str(db_p),
                target=str(target_p),
                out=str(out_p),
                table_size=int(ts),
                harmonics=int(harm),
                bands=int(bands),

                topk=int(self.match_topk.get()),
                topn=int(self.match_topn.get()),
                gain_mode=str(self.match_gain_mode.get()),

                eq_limit_db=float(self.match_eq_limit.get()),
                eq_smooth=float(self.match_eq_smooth.get()),
                include_samples=bool(self.match_include_samples.get()),

                max_filters=int(self.match_max_filters.get()),
                min_sep_bands=int(self.match_min_sep_bands.get()),
                w_harm=float(self.match_w_harm.get()),
                w_eq=float(self.match_w_eq.get()),
            )

            # ✅ CAMBIO: capturar print() del motor
            rc = self._run_with_captured_stdio(wtdiag.cmd_match, args)

            if self.cancel_requested:
                self.logger.warning("Match terminó, pero hubo CANCEL REQUESTED (no se pudo interrumpir a mitad).")

            if rc == 0:
                self.logger.info("Match OK. Se generó report JSON y .txt.")
                self.logger.info(f"Report: {out_p}")
                self.logger.info(f"TXT:    {out_p.with_suffix('.txt')}")
                self.logger.info(f"Log guardado en: {out_p.parent / 'wt_gui.log'}")
            else:
                self.logger.error(f"Match terminó con código {rc}. Revisa logs.")
        except Exception as e:
            self.logger.exception(f"Fallo inesperado: {e}")
        finally:
            self.logger.info("== MATCH END ==")
            if not self._closing:
                self.after(0, lambda: self._set_running(False))

    # ---------------- Log poll ----------------

    def _poll_logs(self):
        if self._closing:
            return

        try:
            while True:
                msg = self.log_queue.get_nowait()
                try:
                    self.txt.insert("end", msg + "\n")
                    self.txt.see("end")
                except (tk.TclError, RuntimeError):
                    return
        except queue.Empty:
            pass

        try:
            self._poll_after_id = self.after(100, self._poll_logs)
        except (tk.TclError, RuntimeError):
            return


if __name__ == "__main__":
    App().mainloop()

