import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
import sys
import glob
import signal
import json
import hparams


class MusicTransformerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Transformer")
        self.root.geometry("720x600")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_generate_tab()
        self._build_train_tab()
        self._build_backend_tab()

        self.current_process = None
        self._load_backend_config()

        # Shared output console (visible from both tabs)
        self.output_frame = ttk.LabelFrame(root, text="Output", padding=5)
        self.output_frame.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        output_toolbar = ttk.Frame(self.output_frame)
        output_toolbar.pack(fill="x")
        ttk.Button(output_toolbar, text="Copy", command=self._copy_output).pack(side="right")
        self.output_text_widget = tk.Text(self.output_frame, height=10, wrap="word", state="disabled")
        scrollbar = ttk.Scrollbar(self.output_frame, orient="vertical", command=self.output_text_widget.yview)
        self.output_text_widget.configure(yscrollcommand=scrollbar.set)
        self.output_text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.output_text("System ready.")

    # ──────────────────────────── Generate Tab ────────────────────────────

    def _build_generate_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Create MIDI")

        # Model selection
        row = 0
        ttk.Label(frame, text="Model (.pt):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(frame, textvariable=self.model_var, width=50)
        self.model_combo.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(frame, text="Browse...", command=self._browse_model).grid(row=row, column=2, padx=5, pady=5)
        self._refresh_models()
        frame.columnconfigure(1, weight=1)

        # Output path
        row += 1
        ttk.Label(frame, text="Output MIDI:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.output_path_var = tk.StringVar(value="./outputs/gen_audio.mid")
        ttk.Entry(frame, textvariable=self.output_path_var, width=50).grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(frame, text="Browse...", command=self._browse_output).grid(row=row, column=2, padx=5, pady=5)

        # Advanced settings in a LabelFrame
        row += 1
        adv = ttk.LabelFrame(frame, text="Generation Settings", padding=10)
        adv.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        adv.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(adv, text="Duration (seconds):").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.duration_var = tk.IntVar(value=30)
        ttk.Entry(adv, textvariable=self.duration_var, width=15).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Tempo (BPM):").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.tempo_var = tk.IntVar(value=120)
        ttk.Entry(adv, textvariable=self.tempo_var, width=15).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Temperature:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.temperature_var = tk.DoubleVar(value=1.0)
        ttk.Entry(adv, textvariable=self.temperature_var, width=15).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Top-K:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.topk_var = tk.StringVar(value="")
        ttk.Entry(adv, textvariable=self.topk_var, width=15).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Mode:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.mode_var = tk.StringVar(value="categorical")
        ttk.Combobox(adv, textvariable=self.mode_var, values=["categorical", "argmax"], width=13, state="readonly").grid(
            row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="MIDI Prompt (optional):").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.prompt_var = tk.StringVar(value="")
        ttk.Entry(adv, textvariable=self.prompt_var, width=40).grid(row=r, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(adv, text="Browse...", command=self._browse_prompt).grid(row=r, column=2, padx=5, pady=2)

        # Generate + Stop buttons
        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=5)
        self.gen_btn = ttk.Button(btn_frame, text="Generate MIDI", command=self._generate)
        self.gen_btn.pack(side="left", padx=5)
        self.gen_stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_process, state="disabled")
        self.gen_stop_btn.pack(side="left", padx=5)

    # ──────────────────────────── Train Tab ───────────────────────────────

    def _build_train_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Train")

        # Sample directory
        row = 0
        ttk.Label(frame, text="MIDI Samples Dir:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.samples_var = tk.StringVar(value="./samples/")
        ttk.Entry(frame, textvariable=self.samples_var, width=50).grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(frame, text="Browse...", command=self._browse_samples).grid(row=row, column=2, padx=5, pady=5)
        frame.columnconfigure(1, weight=1)

        # Output data path
        row += 1
        ttk.Label(frame, text="Processed Data (.pt):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.data_path_var = tk.StringVar(value="./data/processed_data.pt")
        ttk.Entry(frame, textvariable=self.data_path_var, width=50).grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(frame, text="Browse...", command=self._browse_data_path).grid(row=row, column=2, padx=5, pady=5)

        # Checkpoint path
        row += 1
        ttk.Label(frame, text="Checkpoint (.pt):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.ckpt_var = tk.StringVar(value="./checkpoints/ckpt_path.pt")
        ttk.Entry(frame, textvariable=self.ckpt_var, width=50).grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(frame, text="Browse...", command=self._browse_ckpt).grid(row=row, column=2, padx=5, pady=5)

        # Save path
        row += 1
        ttk.Label(frame, text="Save Model (.pt):").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.save_model_var = tk.StringVar(value="./models/save_path.pt")
        ttk.Entry(frame, textvariable=self.save_model_var, width=50).grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(frame, text="Browse...", command=self._browse_save_model).grid(row=row, column=2, padx=5, pady=5)

        # Training settings
        row += 1
        adv = ttk.LabelFrame(frame, text="Training Settings", padding=10)
        adv.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        adv.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(adv, text="Sequence Length:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.seq_len_var = tk.IntVar(value=512)
        ttk.Entry(adv, textvariable=self.seq_len_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Epochs:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.epochs_var = tk.IntVar(value=10)
        ttk.Entry(adv, textvariable=self.epochs_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Batch Size:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Entry(adv, textvariable=self.batch_size_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="d_model:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.d_model_var = tk.IntVar(value=128)
        ttk.Entry(adv, textvariable=self.d_model_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Num Layers:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.num_layers_var = tk.IntVar(value=3)
        ttk.Entry(adv, textvariable=self.num_layers_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Num Heads:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.num_heads_var = tk.IntVar(value=8)
        ttk.Entry(adv, textvariable=self.num_heads_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="d_ff:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.d_ff_var = tk.IntVar(value=512)
        ttk.Entry(adv, textvariable=self.d_ff_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Load Checkpoint:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.load_ckpt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv, variable=self.load_ckpt_var).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        r += 1
        ttk.Label(adv, text="Use AMP:").grid(row=r, column=0, sticky="w", padx=5, pady=2)
        self.use_amp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv, variable=self.use_amp_var).grid(row=r, column=1, sticky="w", padx=5, pady=2)

        # Preprocess + Train + Stop buttons
        row += 1
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=15)
        self.train_btn = ttk.Button(btn_frame, text="Preprocess & Train", command=self._train)
        self.train_btn.pack(side="left", padx=5)
        self.train_stop_btn = ttk.Button(btn_frame, text="Stop", command=self._stop_process, state="disabled")
        self.train_stop_btn.pack(side="left", padx=5)

    # ──────────────────────────── Backend Tab ────────────────────────────

    def _build_backend_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Backend")

        row = 0
        ttk.Label(frame, text="Device:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.backend_var = tk.StringVar(value="auto")
        backend_combo = ttk.Combobox(frame, textvariable=self.backend_var,
                                     values=["auto", "cpu", "cuda", "rocm", "xpu", "directml"],
                                     width=10, state="readonly")
        backend_combo.grid(row=row, column=1, sticky="w", padx=5, pady=5)

        row += 1
        ttk.Label(frame, text="Device ID:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.device_id_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.device_id_var, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=5)

        row += 1
        ttk.Label(frame, text=f"Active: {hparams.device_type}").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.active_device_label = ttk.Label(frame, text=str(hparams.device))
        self.active_device_label.grid(row=row, column=1, sticky="w", padx=5, pady=5)

        row += 1
        ttk.Button(frame, text="Reload Backend", command=self._reload_backend).grid(
            row=row, column=0, columnspan=2, pady=15)

    def _backend_config_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_config.json")

    def _load_backend_config(self):
        path = self._backend_config_path()
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    cfg = json.load(f)
                self.backend_var.set(cfg.get("backend", "auto"))
                self.device_id_var.set(str(cfg.get("device_id", 0)))
            except Exception:
                pass

    def _reload_backend(self):
        backend = self.backend_var.get()
        device_id = self.device_id_var.get().strip()
        try:
            device_id_int = int(device_id) if device_id else 0
        except ValueError:
            messagebox.showerror("Error", "Device ID must be an integer.")
            return

        cfg = {"backend": backend, "device_id": device_id_int}
        with open(self._backend_config_path(), "w") as f:
            json.dump(cfg, f)

        import importlib
        importlib.reload(hparams)

        self.active_device_label.configure(text=str(hparams.device))
        self.output_text(f"Backend reloaded: {backend}, device_type={hparams.device_type}, device={hparams.device}")

    # ──────────────── Helpers ────────────────

    def output_text(self, msg):
        self.output_text_widget.configure(state="normal")
        self.output_text_widget.insert("end", msg + "\n")
        self.output_text_widget.see("end")
        self.output_text_widget.configure(state="disabled")

    def _refresh_models(self):
        models = glob.glob("./models/*.pt") + glob.glob("./models/*.pth")
        self.model_combo["values"] = models
        if models and not self.model_var.get():
            self.model_var.set(models[0])

    def _browse_model(self):
        path = filedialog.askopenfilename(initialdir="./models", title="Select Model",
                                          filetypes=[("PyTorch model", "*.pt *.pth")])
        if path:
            self.model_var.set(path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(initialdir="./outputs", title="Save MIDI As",
                                            defaultextension=".mid",
                                            filetypes=[("MIDI", "*.mid"), ("All files", "*.*")])
        if path:
            self.output_path_var.set(path)

    def _browse_prompt(self):
        path = filedialog.askopenfilename(initialdir="./samples", title="Select MIDI Prompt",
                                          filetypes=[("MIDI", "*.mid *.midi"), ("All files", "*.*")])
        if path:
            self.prompt_var.set(path)

    def _browse_samples(self):
        path = filedialog.askdirectory(initialdir="./samples", title="Select MIDI Samples Directory")
        if path:
            self.samples_var.set(path)

    def _browse_data_path(self):
        path = filedialog.asksaveasfilename(initialdir="./data", title="Save Processed Data As",
                                            defaultextension=".pt",
                                            filetypes=[("PyTorch data", "*.pt"), ("All files", "*.*")])
        if path:
            self.data_path_var.set(path)

    def _browse_ckpt(self):
        path = filedialog.asksaveasfilename(initialdir="./checkpoints", title="Checkpoint Path",
                                            defaultextension=".pt",
                                            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def _browse_save_model(self):
        path = filedialog.asksaveasfilename(initialdir="./models", title="Save Model As",
                                            defaultextension=".pt",
                                            filetypes=[("PyTorch model", "*.pt *.pth"), ("All files", "*.*")])
        if path:
            self.save_model_var.set(path)

    # ──────────────── Generate ────────────────

    def _generate(self):
        model_path = self.model_var.get()
        if not model_path or not os.path.isfile(model_path):
            messagebox.showerror("Error", "Please select a valid model file.")
            return

        save_path = self.output_path_var.get()
        if not save_path:
            messagebox.showerror("Error", "Please specify an output path.")
            return

        self.gen_btn.configure(state="disabled")
        self.gen_stop_btn.configure(state="normal")
        threading.Thread(target=self._run_generate, args=(model_path, save_path), daemon=True).start()

    def _run_generate(self, model_path, save_path):
        self.output_text("\n--- Generating MIDI ---")
        cmd = [sys.executable, "generate.py", model_path, save_path, "-v"]

        temp = self.temperature_var.get()
        if temp != 1.0:
            cmd.extend(["-t", str(temp)])
        k = self.topk_var.get().strip()
        if k:
            try:
                k_int = int(k)
                if k_int > 0:
                    cmd.extend(["-k", str(k_int)])
            except ValueError:
                self.output_text(f"Warning: top-k must be a positive integer, got '{k}'. Skipping.")
        tempo = self.tempo_var.get()
        if tempo != 120:
            cmd.extend(["-tm", str(tempo)])
        if self.mode_var.get() != "categorical":
            cmd.extend(["-m", self.mode_var.get()])
        cmd.extend(["-d", str(self.duration_var.get())])
        prompt = self.prompt_var.get().strip()
        if prompt and os.path.isfile(prompt):
            cmd.extend(["-i", prompt])

        self._run_cmd(cmd, self.gen_btn, self.gen_stop_btn)

    # ──────────────── Train ────────────────

    def _train(self):
        samples_dir = self.samples_var.get()
        if not samples_dir or not os.path.isdir(samples_dir):
            messagebox.showerror("Error", "Please select a valid MIDI samples directory.")
            return

        self.train_btn.configure(state="disabled")
        self.train_stop_btn.configure(state="normal")
        threading.Thread(target=self._run_train_pipeline, daemon=True).start()

    def _run_train_pipeline(self):
        samples_dir = self.samples_var.get()
        data_path = self.data_path_var.get()
        seq_len = self.seq_len_var.get()
        epochs = self.epochs_var.get()
        ckpt_path = self.ckpt_var.get()
        save_path = self.save_model_var.get()

        # Step 1: Preprocessing
        self.output_text("\n--- Preprocessing MIDI files ---")
        pre_cmd = [
            sys.executable, "preprocessing.py",
            samples_dir,
            data_path,
            str(seq_len),
        ]

        exit_code = self._run_cmd(pre_cmd, None, self.train_stop_btn)
        if exit_code != 0:
            self._enable_btn(self.train_btn)
            self.output_text("Preprocessing failed. Aborting.")
            return

        # Step 2: Train
        self.root.after(0, lambda: self.train_stop_btn.configure(state="normal"))
        self.output_text("\n--- Training ---")
        train_cmd = [
            sys.executable, "train.py",
            data_path,
            ckpt_path,
            save_path,
            str(epochs),
            "-bs", str(self.batch_size_var.get()),
        ]

        if self.load_ckpt_var.get():
            train_cmd.append("-l")

        if self.use_amp_var.get():
            train_cmd.append("--use-amp")

        d_model = self.d_model_var.get()
        if d_model != 128:
            train_cmd.extend(["-d", str(d_model)])
        num_layers = self.num_layers_var.get()
        if num_layers != 3:
            train_cmd.extend(["-nl", str(num_layers)])
        num_heads = self.num_heads_var.get()
        if num_heads != 8:
            train_cmd.extend(["-nh", str(num_heads)])
        d_ff = self.d_ff_var.get()
        if d_ff != 512:
            train_cmd.extend(["-dff", str(d_ff)])

        self._run_cmd(train_cmd, self.train_btn, self.train_stop_btn)

    # ──────────────── Command Execution ────────────────

    def _run_cmd(self, cmd, btn, stop_btn=None, output_fn=None):
        if output_fn is None:
            output_fn = self.output_text
        output_fn(f"Running: {' '.join(cmd)}")
        try:
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", **kwargs)
            self.current_process = proc
            for line in proc.stdout:
                output_fn(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                output_fn("Completed successfully.")
            else:
                output_fn(f"Process exited with code {proc.returncode}.")
            return proc.returncode
        except Exception as e:
            output_fn(f"Error: {e}")
            return -1
        finally:
            self.current_process = None
            if btn is not None:
                self.root.after(0, lambda b=btn: b.configure(state="normal"))
            if stop_btn is not None:
                self.root.after(0, lambda b=stop_btn: b.configure(state="disabled"))

    def _enable_btn(self, btn):
        self.root.after(0, lambda: btn.configure(state="normal"))

    def _stop_process(self):
        if self.current_process is not None:
            self.output_text("\n--- Stopping process ---")
            if sys.platform == "win32":
                self.current_process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.current_process.send_signal(signal.SIGINT)

    def _copy_output(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output_text_widget.get("1.0", "end-1c"))
