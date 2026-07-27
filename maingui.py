import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
import sys
import glob
import signal
import json
import queue
import hparams


def apply_custom_theme(root):
    """
    Applies a modern, polished light theme using ttk.Style.
    Ensures dynamic sizing, clean visual hierarchy, and high contrast.
    """
    style = ttk.Style(root)
    
    # Base theme configuration using 'clam' for reliable cross-platform widget styling
    if "clam" in style.theme_names():
        style.theme_use("clam")

    # Light Theme Design Palette
    BG_LIGHT = "#f8fafc"        # Light slate background
    CARD_BG = "#ffffff"         # Clean white container card
    INNER_BG = "#e2e8f0"        # Sub-frame / button background
    ACCENT_PRIMARY = "#4f46e5"  # Indigo accent
    ACCENT_HOVER = "#4338ca"    # Darker indigo hover
    SUCCESS_COLOR = "#10b981"   # Emerald green
    SUCCESS_HOVER = "#059669" 
    DANGER_COLOR = "#f43f5e"    # Rose red
    DANGER_HOVER = "#e11d48"
    TEXT_MAIN = "#0f172a"       # Dark slate text for high contrast
    TEXT_MUTED = "#64748b"      # Slate muted text
    BORDER_COLOR = "#cbd5e1"    # Light clean border

    root.configure(bg=BG_LIGHT)

    # General style definitions
    style.configure(".", background=BG_LIGHT, foreground=TEXT_MAIN, font=("Segoe UI", 9))
    
    # Frame Styles
    style.configure("TFrame", background=BG_LIGHT)
    style.configure("Card.TFrame", background=CARD_BG, relief="flat", borderwidth=1)
    
    # Label Styles
    style.configure("TLabel", background=BG_LIGHT, foreground=TEXT_MAIN, font=("Segoe UI", 9))
    style.configure("Card.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=("Segoe UI", 9))
    style.configure("Header.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=("Segoe UI", 13, "bold"))
    style.configure("Subheader.TLabel", background=CARD_BG, foreground=TEXT_MUTED, font=("Segoe UI", 8, "italic"))
    style.configure("Section.TLabel", background=CARD_BG, foreground=TEXT_MAIN, font=("Segoe UI", 10, "bold"))

    # LabelFrame Styles
    style.configure("TLabelframe", background=CARD_BG, foreground=TEXT_MAIN, relief="solid", borderwidth=1, lightcolor=BORDER_COLOR, darkcolor=BORDER_COLOR)
    style.configure("TLabelframe.Label", background=CARD_BG, foreground=TEXT_MAIN, font=("Segoe UI", 9, "bold"))

    # Button Styles
    style.configure("TButton", font=("Segoe UI", 9, "bold"), background=INNER_BG, foreground=TEXT_MAIN, borderwidth=0, focuscolor="none", padding=(12, 6))
    style.map("TButton", background=[("active", BORDER_COLOR), ("disabled", "#e2e8f0")], foreground=[("disabled", "#94a3b8")])

    style.configure("Accent.TButton", background=ACCENT_PRIMARY, foreground="#ffffff", borderwidth=0, padding=(16, 7))
    style.map("Accent.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#cbd5e1")], foreground=[("disabled", "#94a3b8")])

    style.configure("Success.TButton", background=SUCCESS_COLOR, foreground="#ffffff", borderwidth=0, padding=(16, 7))
    style.map("Success.TButton", background=[("active", SUCCESS_HOVER), ("disabled", "#cbd5e1")], foreground=[("disabled", "#94a3b8")])

    style.configure("Danger.TButton", background=DANGER_COLOR, foreground="#ffffff", borderwidth=0, padding=(16, 7))
    style.map("Danger.TButton", background=[("active", DANGER_HOVER), ("disabled", "#cbd5e1")], foreground=[("disabled", "#94a3b8")])

    # Entry & Combobox Styles
    style.configure("TEntry", fieldbackground="#ffffff", foreground=TEXT_MAIN, insertcolor=TEXT_MAIN, borderwidth=1, relief="solid", lightcolor=BORDER_COLOR, darkcolor=BORDER_COLOR, padding=5)
    style.configure("TCombobox", fieldbackground="#ffffff", background=INNER_BG, foreground=TEXT_MAIN, arrowcolor=TEXT_MAIN, borderwidth=1, relief="solid", lightcolor=BORDER_COLOR, darkcolor=BORDER_COLOR, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", "#ffffff")], selectbackground=[("readonly", ACCENT_PRIMARY)], selectforeground=[("readonly", "#ffffff")])

    # Checkbutton Style
    style.configure("TCheckbutton", background=CARD_BG, foreground=TEXT_MAIN, font=("Segoe UI", 9), focuscolor="none")
    style.map("TCheckbutton", background=[("active", CARD_BG)], indicatorcolor=[("selected", ACCENT_PRIMARY), ("!selected", INNER_BG)])

    # Notebook (Tab Bar) Styles
    style.configure("TNotebook", background=BG_LIGHT, borderwidth=0, tabmargins=[0, 4, 0, 0])
    style.configure("TNotebook.Tab", background=CARD_BG, foreground=TEXT_MUTED, padding=(16, 8), font=("Segoe UI", 9, "bold"), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", ACCENT_PRIMARY), ("active", INNER_BG)], foreground=[("selected", "#ffffff"), ("active", TEXT_MAIN)])

    # Scrollbar Style
    style.configure("TScrollbar", background=CARD_BG, troughcolor=BG_LIGHT, borderwidth=0, arrowcolor=TEXT_MUTED)
    style.map("TScrollbar", background=[("active", BORDER_COLOR)])

    # PanedWindow Style
    style.configure("TPanedwindow", background=BG_LIGHT)


class MusicTransformerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Transformer Studio")
        self.root.geometry("840x680")
        self.root.minsize(720, 520)

        # Apply dark modern design theme
        apply_custom_theme(self.root)

        # Configure root flex layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Vertical PanedWindow for flexible resizing between controls and console output
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.main_paned.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        # Notebook for feature tabs
        self.notebook = ttk.Notebook(self.main_paned)
        self.main_paned.add(self.notebook, weight=3)

        self.current_process = None
        self._msg_queue = queue.Queue()

        self._build_generate_tab()
        self._build_train_tab()
        self._build_backend_tab()

        self._build_shared_console()
        self.root.after(100, self._poll_msg_queue)
        self._load_backend_config()

        self.output_text("System initialized. Welcome to Music Transformer Studio.")


    def _build_shared_console(self):
        """Constructs a modern terminal-like console widget at the bottom."""
        self.output_frame = ttk.LabelFrame(self.main_paned, text=" Console Output ", padding=8)
        self.main_paned.add(self.output_frame, weight=1)

        self.output_frame.grid_rowconfigure(1, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)

        # Console toolbar
        output_toolbar = ttk.Frame(self.output_frame, style="Card.TFrame")
        output_toolbar.grid(row=0, column=0, sticky="ew", columnspan=2, pady=(0, 6))

        title_lbl = ttk.Label(output_toolbar, text="Execution Logs & Status", style="Section.TLabel")
        title_lbl.pack(side="left", padx=4)

        copy_btn = ttk.Button(output_toolbar, text="Copy Logs", command=self._copy_output)
        copy_btn.pack(side="right")

        # Text Console Widget (Styled for Light Mode)
        self.output_text_widget = tk.Text(
            self.output_frame,
            height=6,
            wrap="word",
            state="disabled",
            bg="#f1f5f9",
            fg="#0f172a",
            insertbackground="#0f172a",
            selectbackground="#cbd5e1",
            selectforeground="#0f172a",
            font=("Consolas", 9),
            relief="flat",
            padx=8,
            pady=6
        )
        scrollbar = ttk.Scrollbar(self.output_frame, orient="vertical", command=self.output_text_widget.yview)
        self.output_text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.output_text_widget.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")


    def _build_generate_tab(self):
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text=" Create MIDI ")
        frame = self._create_scrollable_content(outer)

        # File IO Card
        io_card = ttk.LabelFrame(frame, text=" Model & Paths ", padding=12)
        io_card.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        io_card.columnconfigure(1, weight=1)

        # Model selection
        ttk.Label(io_card, text="Model Checkpoint:").grid(row=0, column=0, sticky="w", padx=5, pady=6)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(io_card, textvariable=self.model_var)
        self.model_combo.grid(row=0, column=1, padx=5, pady=6, sticky="ew")
        ttk.Button(io_card, text="Browse...", command=self._browse_model).grid(row=0, column=2, padx=5, pady=6)
        self._refresh_models()

        # Output path
        ttk.Label(io_card, text="Output MIDI File:").grid(row=1, column=0, sticky="w", padx=5, pady=6)
        self.output_path_var = tk.StringVar(value="./outputs/gen_audio.mid")
        ttk.Entry(io_card, textvariable=self.output_path_var).grid(row=1, column=1, padx=5, pady=6, sticky="ew")
        ttk.Button(io_card, text="Browse...", command=self._browse_output).grid(row=1, column=2, padx=5, pady=6)

        # MIDI Prompt
        ttk.Label(io_card, text="Prompt MIDI (Optional):").grid(row=2, column=0, sticky="w", padx=5, pady=6)
        self.prompt_var = tk.StringVar(value="")
        ttk.Entry(io_card, textvariable=self.prompt_var).grid(row=2, column=1, padx=5, pady=6, sticky="ew")
        ttk.Button(io_card, text="Browse...", command=self._browse_prompt).grid(row=2, column=2, padx=5, pady=6)


        # Advanced Generation Settings Card
        adv = ttk.LabelFrame(frame, text=" Generation Parameters ", padding=12)
        adv.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        adv.columnconfigure((1, 3), weight=1)

        # Column 1
        r = 0
        ttk.Label(adv, text="Target Duration (sec):").grid(row=r, column=0, sticky="w", padx=5, pady=6)
        self.duration_var = tk.IntVar(value=30)
        ttk.Entry(adv, textvariable=self.duration_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv, text="Tempo (BPM):").grid(row=r, column=2, sticky="w", padx=(15, 5), pady=6)
        self.tempo_var = tk.IntVar(value=120)
        ttk.Entry(adv, textvariable=self.tempo_var, width=12).grid(row=r, column=3, sticky="w", padx=5, pady=6)

        r += 1
        ttk.Label(adv, text="Temperature:").grid(row=r, column=0, sticky="w", padx=5, pady=6)
        self.temperature_var = tk.DoubleVar(value=1.0)
        ttk.Entry(adv, textvariable=self.temperature_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv, text="Sampling Mode:").grid(row=r, column=2, sticky="w", padx=(15, 5), pady=6)
        self.mode_var = tk.StringVar(value="categorical")
        ttk.Combobox(adv, textvariable=self.mode_var, values=["categorical", "argmax"], width=11, state="readonly").grid(
            row=r, column=3, sticky="w", padx=5, pady=6)

        r += 1
        ttk.Label(adv, text="Top-K:").grid(row=r, column=0, sticky="w", padx=5, pady=6)
        self.topk_var = tk.StringVar(value="")
        ttk.Entry(adv, textvariable=self.topk_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv, text="Top-P (Nucleus):").grid(row=r, column=2, sticky="w", padx=(15, 5), pady=6)
        self.topp_var = tk.DoubleVar(value=0.9)
        ttk.Entry(adv, textvariable=self.topp_var, width=12).grid(row=r, column=3, sticky="w", padx=5, pady=6)

        r += 1
        ttk.Label(adv, text="Repetition Penalty:").grid(row=r, column=0, sticky="w", padx=5, pady=6)
        self.rep_penalty_var = tk.DoubleVar(value=1.05)
        ttk.Entry(adv, textvariable=self.rep_penalty_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv, text="Beam Width:").grid(row=r, column=2, sticky="w", padx=(15, 5), pady=6)
        self.beam_var = tk.IntVar(value=1)
        ttk.Entry(adv, textvariable=self.beam_var, width=12).grid(row=r, column=3, sticky="w", padx=5, pady=6)

        r += 1
        self.compile_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv, text="Compile Model (torch.compile — experimental, may cause crashes)", variable=self.compile_var).grid(
            row=r, column=0, columnspan=4, sticky="w", padx=5, pady=6)

        # Action Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=15)
        
        self.gen_btn = ttk.Button(btn_frame, text="⚡ Generate MIDI", command=self._generate, style="Success.TButton")
        self.gen_btn.pack(side="left", padx=8)
        self.gen_stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self._stop_process, state="disabled", style="Danger.TButton")
        self.gen_stop_btn.pack(side="left", padx=8)


    def _build_train_tab(self):
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text=" Train Model ")
        frame = self._create_scrollable_content(outer)

        # Dataset & Checkpoint Paths Card
        paths_card = ttk.LabelFrame(frame, text=" Dataset & Artifact Locations ", padding=12)
        paths_card.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        paths_card.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(paths_card, text="MIDI Samples Dir:").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.samples_var = tk.StringVar(value="./samples/")
        ttk.Entry(paths_card, textvariable=self.samples_var).grid(row=row, column=1, padx=5, pady=6, sticky="ew")
        ttk.Button(paths_card, text="Browse...", command=self._browse_samples).grid(row=row, column=2, padx=5, pady=6)

        row += 1
        ttk.Label(paths_card, text="Processed Data (.pt):").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.data_path_var = tk.StringVar(value="./data/processed_data.pt")
        ttk.Entry(paths_card, textvariable=self.data_path_var).grid(row=row, column=1, padx=5, pady=6, sticky="ew")
        ttk.Button(paths_card, text="Browse...", command=self._browse_data_path).grid(row=row, column=2, padx=5, pady=6)

        row += 1
        ttk.Label(paths_card, text="Checkpoint (.pt):").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.ckpt_var = tk.StringVar(value="./checkpoints/ckpt_path.pt")
        ttk.Entry(paths_card, textvariable=self.ckpt_var).grid(row=row, column=1, padx=5, pady=6, sticky="ew")
        ttk.Button(paths_card, text="Browse...", command=self._browse_ckpt).grid(row=row, column=2, padx=5, pady=6)

        row += 1
        ttk.Label(paths_card, text="Save Model (.pt):").grid(row=row, column=0, sticky="w", padx=5, pady=6)
        self.save_model_var = tk.StringVar(value="./models/save_path.pt")
        ttk.Entry(paths_card, textvariable=self.save_model_var).grid(row=row, column=1, padx=5, pady=6, sticky="ew")
        ttk.Button(paths_card, text="Browse...", command=self._browse_save_model).grid(row=row, column=2, padx=5, pady=6)


        # Hyperparameters Card
        adv = ttk.LabelFrame(frame, text=" Core Hyperparameters ", padding=12)
        adv.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)
        adv.columnconfigure((1, 3), weight=1)

        r = 0
        ttk.Label(adv, text="Sequence Length:").grid(row=r, column=0, sticky="w", padx=5, pady=6)
        self.seq_len_var = tk.IntVar(value=512)
        ttk.Entry(adv, textvariable=self.seq_len_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv, text="Epochs:").grid(row=r, column=2, sticky="w", padx=(15, 5), pady=6)
        self.epochs_var = tk.IntVar(value=10)
        ttk.Entry(adv, textvariable=self.epochs_var, width=12).grid(row=r, column=3, sticky="w", padx=5, pady=6)

        r += 1
        ttk.Label(adv, text="Batch Size:").grid(row=r, column=0, sticky="w", padx=5, pady=6)
        self.batch_size_var = tk.IntVar(value=16)
        ttk.Entry(adv, textvariable=self.batch_size_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv, text="d_model:").grid(row=r, column=2, sticky="w", padx=(15, 5), pady=6)
        self.d_model_var = tk.IntVar(value=128)
        ttk.Entry(adv, textvariable=self.d_model_var, width=12).grid(row=r, column=3, sticky="w", padx=5, pady=6)

        r += 1
        ttk.Label(adv, text="Num Layers:").grid(row=r, column=0, sticky="w", padx=5, pady=6)
        self.num_layers_var = tk.IntVar(value=3)
        ttk.Entry(adv, textvariable=self.num_layers_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv, text="Num Heads:").grid(row=r, column=2, sticky="w", padx=(15, 5), pady=6)
        self.num_heads_var = tk.IntVar(value=8)
        ttk.Entry(adv, textvariable=self.num_heads_var, width=12).grid(row=r, column=3, sticky="w", padx=5, pady=6)

        r += 1
        ttk.Label(adv, text="d_ff:").grid(row=r, column=0, sticky="w", padx=5, pady=6)
        self.d_ff_var = tk.IntVar(value=512)
        ttk.Entry(adv, textvariable=self.d_ff_var, width=12).grid(row=r, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv, text="Context Window:").grid(row=r, column=2, sticky="w", padx=(15, 5), pady=6)
        self.max_rel_dist_var = tk.IntVar(value=1024)
        ttk.Entry(adv, textvariable=self.max_rel_dist_var, width=12).grid(row=r, column=3, sticky="w", padx=5, pady=6)

        r += 1
        # Toggle Options
        flags_frame = ttk.Frame(adv, style="Card.TFrame")
        flags_frame.grid(row=r, column=0, columnspan=4, sticky="w", padx=5, pady=6)

        self.load_ckpt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(flags_frame, text="Load Checkpoint", variable=self.load_ckpt_var).pack(side="left", padx=5)

        self.use_amp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(flags_frame, text="Use AMP", variable=self.use_amp_var).pack(side="left", padx=5)

        # Architecture Tweaks Card
        arch_frame = ttk.LabelFrame(frame, text=" Model Architecture Options ", padding=10)
        arch_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        arch_frame.columnconfigure((0, 1, 2, 3), weight=1)

        self.swiglu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(arch_frame, text="SwiGLU FFN", variable=self.swiglu_var).grid(row=0, column=0, sticky="w", padx=8, pady=4)

        self.qknorm_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(arch_frame, text="QK-Norm", variable=self.qknorm_var).grid(row=0, column=1, sticky="w", padx=8, pady=4)

        self.sdpa_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(arch_frame, text="SDPA (Flash Attention)", variable=self.sdpa_var).grid(row=0, column=2, sticky="w", padx=8, pady=4)

        self.ema_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(arch_frame, text="EMA Optimization", variable=self.ema_var).grid(row=0, column=3, sticky="w", padx=8, pady=4)

        # Advanced Training Options Card
        adv2 = ttk.LabelFrame(frame, text=" Optimization & Scheduler ", padding=10)
        adv2.grid(row=3, column=0, columnspan=3, sticky="ew", padx=10, pady=5)
        adv2.columnconfigure((1, 3), weight=1)

        ttk.Label(adv2, text="LR Schedule:").grid(row=0, column=0, sticky="w", padx=5, pady=6)
        self.lr_sched_var = tk.StringVar(value="cosine")
        ttk.Combobox(adv2, textvariable=self.lr_sched_var, values=["cosine", "transformer"], width=11, state="readonly").grid(
            row=0, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv2, text="Weight Decay:").grid(row=0, column=2, sticky="w", padx=(15, 5), pady=6)
        self.wd_var = tk.DoubleVar(value=0.01)
        ttk.Entry(adv2, textvariable=self.wd_var, width=12).grid(row=0, column=3, sticky="w", padx=5, pady=6)

        ttk.Label(adv2, text="Max Grad Norm:").grid(row=1, column=0, sticky="w", padx=5, pady=6)
        self.grad_norm_var = tk.DoubleVar(value=1.0)
        ttk.Entry(adv2, textvariable=self.grad_norm_var, width=12).grid(row=1, column=1, sticky="w", padx=5, pady=6)

        ttk.Label(adv2, text="Label Smoothing:").grid(row=1, column=2, sticky="w", padx=(15, 5), pady=6)
        self.label_smooth_var = tk.DoubleVar(value=0.05)
        ttk.Entry(adv2, textvariable=self.label_smooth_var, width=12).grid(row=1, column=3, sticky="w", padx=5, pady=6)

        ttk.Label(adv2, text="EMA Decay:").grid(row=2, column=0, sticky="w", padx=5, pady=6)
        self.ema_decay_var = tk.DoubleVar(value=0.999)
        ttk.Entry(adv2, textvariable=self.ema_decay_var, width=12).grid(row=2, column=1, sticky="w", padx=5, pady=6)

        # Train Pipeline Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=15)
        self.train_btn = ttk.Button(btn_frame, text="🚀 Preprocess & Train", command=self._train, style="Accent.TButton")
        self.train_btn.pack(side="left", padx=8)
        self.train_stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self._stop_process, state="disabled", style="Danger.TButton")
        self.train_stop_btn.pack(side="left", padx=8)


    def _build_backend_tab(self):
        outer = ttk.Frame(self.notebook)
        self.notebook.add(outer, text=" Backend & Hardware ")
        frame = self._create_scrollable_content(outer)

        card = ttk.LabelFrame(frame, text=" Target Execution Device ", padding=16)
        card.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        card.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(card, text="Backend Type:").grid(row=row, column=0, sticky="w", padx=5, pady=8)
        self.backend_var = tk.StringVar(value="auto")
        backend_combo = ttk.Combobox(card, textvariable=self.backend_var,
                                     values=["auto", "cpu", "cuda", "rocm", "xpu", "directml"],
                                     width=14, state="readonly")
        backend_combo.grid(row=row, column=1, sticky="w", padx=5, pady=8)

        row += 1
        ttk.Label(card, text="Device ID:").grid(row=row, column=0, sticky="w", padx=5, pady=8)
        self.device_id_var = tk.StringVar(value="0")
        ttk.Entry(card, textvariable=self.device_id_var, width=14).grid(row=row, column=1, sticky="w", padx=5, pady=8)

        row += 1
        ttk.Label(card, text=f"Active System Driver: {hparams.device_type}", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=8)
        self.active_device_label = ttk.Label(card, text=str(hparams.device), font=("Segoe UI", 9, "bold"), foreground="#0284c7")
        self.active_device_label.grid(row=row, column=1, sticky="w", padx=5, pady=8)

        row += 1
        ttk.Button(card, text="🔄 Reload Backend", command=self._reload_backend, style="Accent.TButton").grid(
            row=row, column=0, columnspan=2, pady=(15, 5))


    def _create_scrollable_content(self, parent):
        """Creates a smooth, scrollable canvas frame that responds to dynamic resizing."""
        canvas = tk.Canvas(parent, borderwidth=0, highlightthickness=0, bg="#f8fafc")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)

        inner = ttk.Frame(canvas, style="TFrame")
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event):
            canvas.itemconfig(inner_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            if sys.platform == "darwin":
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_up(event):
            canvas.yview_scroll(-1, "units")
        def _on_mousewheel_down(event):
            canvas.yview_scroll(1, "units")

        def _bind_mousewheel_recursive(widget):
            widget.bind("<MouseWheel>", _on_mousewheel, add="+")
            widget.bind("<Button-4>", _on_mousewheel_up, add="+")
            widget.bind("<Button-5>", _on_mousewheel_down, add="+")
            for child in widget.winfo_children():
                _bind_mousewheel_recursive(child)

        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel_up)
        canvas.bind("<Button-5>", _on_mousewheel_down)
        _bind_mousewheel_recursive(inner)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return inner


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

        # Full reload: subprocess scripts reimport on their own,
        # but for the GUI we need to refresh hparams
        import importlib
        import sys
        for mod_name in list(sys.modules):
            if mod_name in ("hparams", "vocabulary", "masking", "layers", "model", "tokenizer"):
                del sys.modules[mod_name]
        for mod_name in ("vocabulary", "masking", "layers", "model", "hparams"):
            importlib.import_module(mod_name)
        import hparams as reloaded_hparams

        self.active_device_label.configure(text=str(reloaded_hparams.device))
        self.output_text(f"Backend reloaded: {backend}, device_type={reloaded_hparams.device_type}, device={reloaded_hparams.device}")


    def output_text(self, msg):
        self._msg_queue.put(msg)

    def _poll_msg_queue(self):
        try:
            while True:
                msg = self._msg_queue.get_nowait()
                self.output_text_widget.configure(state="normal")
                self.output_text_widget.insert("end", msg + "\n")
                self.output_text_widget.see("end")
                self.output_text_widget.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_msg_queue)

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

        p = self.topp_var.get()
        if p != 0.9:
            cmd.extend(["-p", str(p)])
        rp = self.rep_penalty_var.get()
        if rp != 1.05:
            cmd.extend(["-rp", str(rp)])
        bw = self.beam_var.get()
        if bw > 1:
            cmd.extend(["-bw", str(bw)])
        if self.compile_var.get():
            cmd.append("-c")

        prompt = self.prompt_var.get().strip()
        if prompt and os.path.isfile(prompt):
            cmd.extend(["-i", prompt])

        self._run_cmd(cmd, self.gen_btn, self.gen_stop_btn)


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
        max_rel_dist = self.max_rel_dist_var.get()
        if max_rel_dist != 1024:
            train_cmd.extend(["-mrd", str(max_rel_dist)])

        # Architecture toggles
        if not self.swiglu_var.get():
            train_cmd.append("--no-swiglu")
        if not self.qknorm_var.get():
            train_cmd.append("--no-qk-norm")
        if not self.sdpa_var.get():
            train_cmd.append("--no-sdpa")
        if self.ema_var.get():
            train_cmd.append("--use-ema")

        # Advanced training args
        lr_sched = self.lr_sched_var.get()
        if lr_sched != "cosine":
            train_cmd.extend(["--lr-schedule", lr_sched])
        wd = self.wd_var.get()
        if wd != 0.01:
            train_cmd.extend(["-wd", str(wd)])
        gn = self.grad_norm_var.get()
        if gn != 1.0:
            train_cmd.extend(["-gn", str(gn)])
        ls = self.label_smooth_var.get()
        if ls != 0.05:
            train_cmd.extend(["-ls", str(ls)])
        ed = self.ema_decay_var.get()
        if ed != 0.999:
            train_cmd.extend(["--ema-decay", str(ed)])

        self._run_cmd(train_cmd, self.train_btn, self.train_stop_btn)


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
            self.current_process.terminate()
            try:
                self.current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                self.current_process.wait()

    def _copy_output(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.output_text_widget.get("1.0", "end-1c"))


if __name__ == "__main__":
    root = tk.Tk()
    app = MusicTransformerGUI(root)
    root.mainloop()