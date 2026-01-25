# wtgui.py
import threading
import queue
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional  # <-- FIX
from types import SimpleNamespace  # <-- FIX: para crear args tipo argparse

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
        # fallback: abrir carpeta contenedora
        parent = str(path if path.is_dir() else path.parent)
        if sys.platform.startswith("win"):
            os.startfile(parent)  # noqa
        elif sys.platform == "darwin":
            subprocess.run(["open", parent], check=False)
        else:
            subprocess.run(["xdg-open", parent], check=False)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WT Diagnoser - Offline")
        self.geometry("980x700")

        self.log_queue = queue.Queue()

        # Vars comunes
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

        self._build_ui()
        self._setup_logging()
        self._poll_logs()

    # ---------------- UI ----------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Opciones comunes (arriba)
        opt = ttk.LabelFrame(self, text="Parámetros (comunes)")
        opt.pack(fill="x", **pad)

        ttk.Label(opt, text="tableSize").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(opt, from_=256, to=8192, increment=256, textvariable=self.table_size, width=10)\
            .grid(row=0, column=1, padx=6)

        ttk.Label(opt, text="harmonics").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(opt, from_=16, to=256, increment=16, textvariable=self.harmonics, width=10)\
            .grid(row=0, column=3, padx=6)

        ttk.Label(opt, text="bands").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(opt, from_=32, to=256, increment=32, textvariable=self.bands, width=10)\
            .grid(row=0, column=5, padx=6)

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

        ttk.Label(frm, text="Carpeta DB (salida descriptores):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_dir, width=72).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir...", command=self._pick_out).grid(row=1, column=2)

        frm.columnconfigure(1, weight=1)

        runfrm = ttk.Frame(parent)
        runfrm.pack(fill="x", **pad)

        self.btn_run_index = ttk.Button(runfrm, text="RUN (Indexar)", command=self._run_index)
        self.btn_run_index.pack(side="left")

        ttk.Label(runfrm, text="Genera: descriptores JSON + _INDEX.json").pack(side="left", padx=12)

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

        # Parámetros match
        opts = ttk.LabelFrame(parent, text="Parámetros Match")
        opts.pack(fill="x", **pad)

        ttk.Label(opts, text="topK").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(opts, from_=3, to=100, increment=1, textvariable=self.match_topk, width=10)\
            .grid(row=0, column=1, padx=6)

        ttk.Label(opts, text="EQ limit (dB)").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(opts, from_=1.0, to=18.0, increment=0.5, textvariable=self.match_eq_limit, width=10)\
            .grid(row=0, column=3, padx=6)

        ttk.Label(opts, text="EQ smooth").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(opts, from_=0.2, to=6.0, increment=0.1, textvariable=self.match_eq_smooth, width=10)\
            .grid(row=0, column=5, padx=6)

        # Label CONSISTENTE con el motor/CLI:
        ttk.Checkbutton(opts, text="Include samples (type=sample)", variable=self.match_include_samples)\
            .grid(row=0, column=6, padx=10, sticky="w")

        opts.columnconfigure(7, weight=1)

        runfrm = ttk.Frame(parent)
        runfrm.pack(fill="x", **pad)

        self.btn_run_match = ttk.Button(runfrm, text="RUN (Match)", command=self._run_match)
        self.btn_run_match.pack(side="left")

        ttk.Label(runfrm, text="Genera report JSON + .txt con filtros PEQ aproximados.").pack(side="left", padx=12)

    # ---------------- Logging ----------------

    def _setup_logging(self):
        self.logger = logging.getLogger("wtgui")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        ui_handler = TkTextHandler(self.log_queue)
        ui_handler.setFormatter(fmt)
        self.logger.addHandler(ui_handler)

    def _ensure_file_logger(self, out_dir: Path):
        """Crea (o recrea) un logger a archivo dentro del output folder."""
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)

        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "wt_gui.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(fh)

    # ---------------- Pickers ----------------

    def _pick_in(self):
        p = filedialog.askdirectory(title="Elige carpeta de wavetables")
        if p:
            self.in_dir.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory(title="Elige carpeta DB de salida (descriptores)")
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
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _common_args(self):
        return int(self.table_size.get()), int(self.harmonics.get()), int(self.bands.get())

    # ---------------- Run: Index ----------------

    def _run_index(self):
        in_p = Path(self.in_dir.get().strip())
        out_p = Path(self.out_dir.get().strip())

        if not in_p.exists():
            messagebox.showerror("Falta entrada", "Elige una carpeta válida de wavetables.")
            return
        if not str(out_p).strip():
            messagebox.showerror("Falta salida", "Elige una carpeta DB de salida.")
            return

        self._ensure_file_logger(out_p)
        self._set_running(True)

        ts, harm, bands = self._common_args()
        self.logger.info("== INDEX START ==")
        self.logger.info(f"Entrada: {in_p}")
        self.logger.info(f"Salida:  {out_p}")
        self.logger.info(f"Params: tableSize={ts} harmonics={harm} bands={bands}")

        th = threading.Thread(target=self._worker_index, args=(in_p, out_p, ts, harm, bands), daemon=True)
        th.start()

    def _worker_index(self, in_p: Path, out_p: Path, ts: int, harm: int, bands: int):
        try:
            args = SimpleNamespace(
                in_dir=str(in_p),
                out_dir=str(out_p),
                table_size=int(ts),
                harmonics=int(harm),
                bands=int(bands),
                # Nota: no exponemos include_samples en UI Index (por defecto False).
                # cmd_index usa getattr(..., False), así que está OK.
            )

            rc = wtdiag.cmd_index(args)
            if rc == 0:
                self.logger.info("Indexado OK. Se generó _INDEX.json y descriptores.")
                self.logger.info(f"Log guardado en: {out_p / 'wt_gui.log'}")
            else:
                self.logger.error(f"Indexado terminó con código {rc}. Revisa logs.")
        except Exception as e:
            self.logger.exception(f"Fallo inesperado: {e}")
        finally:
            self.logger.info("== INDEX END ==")
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

        self._ensure_file_logger(out_p.parent)
        self._set_running(True)

        ts, harm, bands = self._common_args()
        self.logger.info("== DIAG START ==")
        self.logger.info(f"WAV:   {wav_p}")
        self.logger.info(f"Salida:{out_p}")
        self.logger.info(f"Params: tableSize={ts} harmonics={harm} bands={bands}")

        th = threading.Thread(target=self._worker_diag, args=(wav_p, out_p, ts, harm, bands), daemon=True)
        th.start()

    def _worker_diag(self, wav_p: Path, out_p: Path, ts: int, harm: int, bands: int):
        try:
            args = SimpleNamespace(
                wav=str(wav_p),
                out=str(out_p),
                table_size=int(ts),
                harmonics=int(harm),
                bands=int(bands),
            )

            rc = wtdiag.cmd_diag(args)
            if rc == 0:
                self.logger.info("Diag OK. Se generó descriptor JSON.")
                self.logger.info(f"Log guardado en: {out_p.parent / 'wt_gui.log'}")
            else:
                self.logger.error(f"Diag terminó con código {rc}. Revisa logs.")
        except Exception as e:
            self.logger.exception(f"Fallo inesperado: {e}")
        finally:
            self.logger.info("== DIAG END ==")
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

        self._ensure_file_logger(out_p.parent)
        self._set_running(True)

        ts, harm, bands = self._common_args()
        self.logger.info("== MATCH START ==")
        self.logger.info(f"DB:     {db_p}")
        self.logger.info(f"Target: {target_p}")
        self.logger.info(f"Salida: {out_p}")
        self.logger.info(
            f"Params: tableSize={ts} harmonics={harm} bands={bands} "
            f"topk={self.match_topk.get()} eq_limit_db={self.match_eq_limit.get()} eq_smooth={self.match_eq_smooth.get()} "
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
            args = SimpleNamespace(
                db_dir=str(db_p),
                target=str(target_p),
                out=str(out_p),
                table_size=int(ts),
                harmonics=int(harm),
                bands=int(bands),
                topk=int(self.match_topk.get()),
                eq_limit_db=float(self.match_eq_limit.get()),
                eq_smooth=float(self.match_eq_smooth.get()),
                include_samples=bool(self.match_include_samples.get()),
            )

            rc = wtdiag.cmd_match(args)
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
            self.after(0, lambda: self._set_running(False))

    # ---------------- Log poll ----------------

    def _poll_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.txt.insert("end", msg + "\n")
                self.txt.see("end")
        except queue.Empty:
            pass
        self.after(100, self._poll_logs)


if __name__ == "__main__":
    App().mainloop()

