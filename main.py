import sys
import os

for dir in ["./checkpoints", "./models", "./outputs", "./data", "./samples"]:
    if not os.path.exists(dir):
        os.makedirs(dir)

from maingui import MusicTransformerGUI
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = MusicTransformerGUI(root)
    root.mainloop()
