# gui.py

import tkinter as tk
import os
import random
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from bms.bms import resize_and_convert_to_lab, generate_boolean_maps, compute_attention_map, post_process


class BMS_GUI:
    def __init__(self, root):
        self.boolean_maps = []
        self.attention_map = None
        self.root = root
        self.root.title("BMS - Binary Map Saliency")
        self.root.geometry("1600x800")
        self.root.resizable(True, True)

        self.original_image = None
        self.processed_image = None
        self.fixation_image = None

        # --- Górny pasek przycisków ---
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.load_btn = tk.Button(button_frame, text="Załaduj obraz", command=self.load_image)
        self.load_btn.pack(side="left", padx=5)

        self.process_btn = tk.Button(button_frame, text="Uruchom BMS", command=self.run_bms)
        self.process_btn.pack(side="left", padx=5)

        self.save_btn = tk.Button(button_frame, text="Zapisz wynik", command=self.save_image)
        self.save_btn.pack(side="left", padx=5)

        self.view_boolean_btn = tk.Button(button_frame, text="Pokaż boolean maps", command=self.show_boolean_maps)
        self.view_boolean_btn.pack(side="left", padx=5)

        self.view_attention_btn = tk.Button(button_frame, text="Pokaż attention map", command=self.show_attention_map)
        self.view_attention_btn.pack(side="left", padx=5)

        self.random_cat_btn = tk.Button(button_frame, text="Wylosuj z CAT2000", command=self.load_random_cat2000)
        self.random_cat_btn.pack(side="left", padx=5)

        # --- Dolna część: obrazy i podpisy ---
        image_frame = tk.Frame(root)
        image_frame.pack(pady=20)

        # Oryginał
        original_frame = tk.Frame(image_frame)
        original_frame.pack(side="left", padx=10)
        tk.Label(original_frame, text="Oryginał", font=("Arial", 12)).pack()
        self.canvas_original = tk.Label(original_frame)
        self.canvas_original.pack()

        # PIONOWA LINIA
        separator1 = tk.Frame(image_frame, width=2, bg="gray", height=420)
        separator1.pack(side="left", padx=5)

        # Fixation map (z CAT2000)
        fixation_frame = tk.Frame(image_frame)
        fixation_frame.pack(side="left", padx=10)
        tk.Label(fixation_frame, text="Fixation map (dataset)", font=("Arial", 12)).pack()
        self.canvas_fixation = tk.Label(fixation_frame)
        self.canvas_fixation.pack()

        # PIONOWA LINIA
        separator2 = tk.Frame(image_frame, width=2, bg="gray", height=420)
        separator2.pack(side="left", padx=5)

        # BMS wynik
        bms_frame = tk.Frame(image_frame)
        bms_frame.pack(side="left", padx=10)
        tk.Label(bms_frame, text="BMS (Twój wynik)", font=("Arial", 12)).pack()
        self.canvas_result = tk.Label(bms_frame)
        self.canvas_result.pack()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.png *.jpeg")])
        if path:
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                messagebox.showerror("Błąd", "Nie udało się wczytać obrazu.")
                return
            self.display_image(self.original_image, self.canvas_original)

    def run_bms(self):
        if self.original_image is None:
            messagebox.showwarning("Uwaga", "Najpierw załaduj obraz.")
            return
        lab = resize_and_convert_to_lab(self.original_image)
        self.boolean_maps = generate_boolean_maps(lab, thresholds_per_channel=20)
        self.attention_map = compute_attention_map(self.boolean_maps, lab.shape)
        attention_norm = cv2.normalize(self.attention_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # po wykonaniu BMS
        self.processed_image = post_process(attention_norm)
        self.display_image(self.processed_image, self.canvas_result, is_gray=True)

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Uwaga", "Brak przetworzonego obrazu do zapisania.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            cv2.imwrite(path, self.processed_image)
            messagebox.showinfo("Sukces", "Obraz zapisany pomyślnie.")

    def display_image(self, image, canvas, is_gray=False):
        if is_gray:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_pil = img_pil.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.configure(image=img_tk)
        canvas.image = img_tk

    def show_boolean_maps(self):
        if not self.boolean_maps:
            messagebox.showwarning("Brak danych", "Najpierw uruchom BMS.")
            return

        viewer = tk.Toplevel(self.root)
        viewer.title("Boolean Maps")
        img_label = tk.Label(viewer)
        img_label.pack()

        index = {"i": 0}

        def update_image():
            _, threshold, binary = self.boolean_maps[index["i"]]
            title = f"Mapa {index['i'] + 1}/{len(self.boolean_maps)}  próg: {'>' if threshold > 0 else '<='} {abs(int(threshold))}"
            viewer.title(title)
            img = cv2.resize(binary, (400, 400))
            pil = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(pil)
            img_label.configure(image=imgtk)
            img_label.image = imgtk

        def next_img():
            index["i"] = (index["i"] + 1) % len(self.boolean_maps)
            update_image()

        def prev_img():
            index["i"] = (index["i"] - 1) % len(self.boolean_maps)
            update_image()

        tk.Button(viewer, text="<<", command=prev_img).pack(side="left", padx=10)
        tk.Button(viewer, text=">>", command=next_img).pack(side="right", padx=10)

        update_image()

    def show_attention_map(self):
        if self.attention_map is None:
            messagebox.showwarning("Brak danych", "Najpierw uruchom BMS.")
            return

        attn = cv2.normalize(self.attention_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        viewer = tk.Toplevel(self.root)
        viewer.title("Attention Map")
        img = cv2.resize(attn, (400, 400))
        pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(pil)
        lbl = tk.Label(viewer, image=imgtk)
        lbl.image = imgtk
        lbl.pack()

    def load_random_cat2000(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(current_dir, "..", "pictures", "CAT2000")
        stim_dir = os.path.join(base_path, "Stimuli")
        fix_dir = os.path.join(base_path, "FIXATIONMAPS")

        # Sprawdzenie czy foldery istnieją
        if not os.path.exists(stim_dir) or not os.path.exists(fix_dir):
            messagebox.showerror("Błąd", f"Nie znaleziono folderu CAT2000.\nSprawdź, czy dane są w:\n{base_path}")
            return

        try:
            categories = [c for c in os.listdir(stim_dir) if os.path.isdir(os.path.join(stim_dir, c))]
            if not categories:
                messagebox.showerror("Błąd", "Brak kategorii w katalogu Stimuli.")
                return

            chosen_cat = random.choice(categories)

            cat_stim_path = os.path.join(stim_dir, chosen_cat)
            cat_fix_path = os.path.join(fix_dir, chosen_cat)

            image_files = [f for f in os.listdir(cat_stim_path) if f.lower().endswith(".jpg")]
            if not image_files:
                messagebox.showerror("Błąd", f"Brak obrazów w kategorii {chosen_cat}")
                return

            chosen_img = random.choice(image_files)
            stim_img_path = os.path.join(cat_stim_path, chosen_img)
            fix_img_path = os.path.join(cat_fix_path, chosen_img)

            stim = cv2.imread(stim_img_path)
            fix = cv2.imread(fix_img_path, cv2.IMREAD_GRAYSCALE)

            if stim is None or fix is None:
                messagebox.showerror("Błąd", f"Nie udało się wczytać:\n{chosen_img}")
                return

            self.original_image = stim
            self.fixation_image = fix
            self.processed_image = None

            self.display_image(self.original_image, self.canvas_original)
            self.display_image(self.fixation_image, self.canvas_fixation, is_gray=True)
            self.canvas_result.configure(image="")  # Wyczyść canvas BMS

        except Exception as e:
            messagebox.showerror("Błąd krytyczny", f"Wystąpił błąd:\n{str(e)}")

