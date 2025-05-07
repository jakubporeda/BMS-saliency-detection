# main.py

import tkinter as tk
from gui.gui import BMS_GUI

if __name__ == "__main__":
    root = tk.Tk()
    app = BMS_GUI(root)
    root.mainloop()