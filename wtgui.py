# wtgui.py
import threading
import queue
import logging
from pathlib import Path
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


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WT Diagnoser - Offline")
        self.geometry("920x640")

        self.log_queue = queue.Queue()

        # Vars UI
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.table_size = tk.IntVar(value=2048)
        self.harmonics = tk.IntVar(value=64)
        self.bands = tk.IntVar(value=128)

        self._build_ui()
        self._setup_logging()
        self._poll_logs()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="x", **pad)

        ttk.Label(frm, text="Carpeta wavetables (entrada):").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.in_dir, width=70).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir...", command=self._pick_in).grid(row=0, column=2)

        ttk.Label(frm, text="Carpeta diagnóstico (salida):").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_dir, width=70).grid(row=1, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Elegir...", command=self._pick_out).grid(row=1, column=2)

        frm.columnconfigure(1, weight=1)

        opt = ttk.Frame(self)
        opt.pack(fill="x", **pad)

        ttk.Label(opt, text="tableSize").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(opt, from_=256, to=8192, increment=256, textvariable=self.table_size, width=10).grid(row=0, column=1, padx=6)

        ttk.Label(opt, text="harmonics").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(opt, from_=16, to=256, increment=16, textvariable=self.harmonics, width=10).grid(row=0, column=3, padx=6)

        ttk.Label(opt, text="bands").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(opt, from_=32, to=256, increment=32, textvariable=self.bands, width=10).grid(row=0, column=5, padx=6)

        act = ttk.Frame(self)
        act.pack(fill="x", **pad)

        self.btn_run = ttk.Button(act, text="RUN (Indexar)", command=self._run_index)
        self.btn_run.pack(side="left")

        ttk.Button(act, text="Abrir carpeta salida", command=self._open_out).pack(side="left", padx=8)

        self.progress = ttk.Progressbar(act, mode="indeterminate")
        self.progress.pack(side="right", fill="x", expand=True)

        # Log box
        logfrm = ttk.Frame(self)
        logfrm.pack(fill="both", expand=True, **pad)

        ttk.Label(logfrm, text="Logs (detallados):").pack(anchor="w")
        self.txt = tk.Text(logfrm, wrap="word")
        self.txt.pack(fill="both", expand=True)

    def _setup_logging(self):
        self.logger = logging.getLogger("wtgui")
        self.logger.setLevel(logging.INFO)

        # Limpia handlers previos (útil en dev)
        self.logger.handlers.clear()

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        # Handler a UI
        ui_handler = TkTextHandler(self.log_queue)
        ui_handler.setFormatter(fmt)
        self.logger.addHandler(ui_handler)

    def _ensure_file_logger(self, out_dir: Path):
        """Crea (o recrea) un logger a archivo dentro del output."""
        # remover handlers File anteriores
        for h in list(self.logger.handlers):
            if isinstance(h, logging.FileHandler):
                self.logger.removeHandler(h)

        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "wt_diagnoser.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(fh)

    def _pick_in(self):
        p = filedialog.askdirectory(title="Elige carpeta de wavetables")
        if p:
            self.in_dir.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory(title="Elige carpeta de salida (diagnóstico)")
        if p:
            self.out_dir.set(p)

    def _open_out(self):
        p = self.out_dir.get().strip()
        if not p:
            return
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            # Windows explorer
            import os
            os.startfile(p)  # noqa
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _set_running(self, running: bool):
        self.btn_run.config(state=("disabled" if running else "normal"))
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _run_index(self):
        in_p = Path(self.in_dir.get().strip())
        out_p = Path(self.out_dir.get().strip())
        if not in_p.exists():
            messagebox.showerror("Falta entrada", "Elige una carpeta válida de wavetables.")
            return
        if not out_p:
            messagebox.showerror("Falta salida", "Elige una carpeta de salida.")
            return

        self._ensure_file_logger(out_p)
        self._set_running(True)
        self.logger.info("== WT DIAG START ==")
        self.logger.info(f"Entrada: {in_p}")
        self.logger.info(f"Salida:  {out_p}")
        self.logger.info(f"Params: tableSize={self.table_size.get()} harmonics={self.harmonics.get()} bands={self.bands.get()}")

        # Corre en thread para no congelar UI
        th = threading.Thread(target=self._worker_index, args=(in_p, out_p), daemon=True)
        th.start()

    def _worker_index(self, in_p: Path, out_p: Path):
        try:
            # En vez de invocar CLI, llamamos la lógica usando "args" fake:
            class Args:
                in_dir = str(in_p)
                out_dir = str(out_p)
                table_size = int(self.table_size.get())
                harmonics = int(self.harmonics.get())
                bands = int(self.bands.get())

            rc = wtdiag.cmd_index(Args())
            if rc == 0:
                self.logger.info("Indexado OK. Se generó _INDEX.json y descriptores.")
                self.logger.info(f"Log guardado en: {out_p / 'wt_diagnoser.log'}")
            else:
                self.logger.error(f"Indexado terminó con código {rc}. Revisa logs.")
        except Exception as e:
            self.logger.exception(f"Fallo inesperado: {e}")
        finally:
            self.logger.info("== WT DIAG END ==")
            self.after(0, lambda: self._set_running(False))

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
