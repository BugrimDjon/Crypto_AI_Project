import os
import json
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

HISTORY_FOLDER = "results/history"

class HistoryViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 Просмотр истории обучения")

        self.file_list = []
        self.current_index = 0
        self.history = {}

        # === UI ===
        self.combo = ttk.Combobox(root, state="readonly", width=120)
        self.combo.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.combo.bind("<<ComboboxSelected>>", self.on_select)

        ttk.Button(root, text="🔄 Обновить", command=self.refresh_files).grid(row=0, column=3, padx=5)

        ttk.Button(root, text="⏮️ Назад", command=self.prev_file).grid(row=1, column=0)
        ttk.Button(root, text="📊 Показать", command=self.show_plot).grid(row=1, column=1)
        ttk.Button(root, text="⏭️ Вперёд", command=self.next_file).grid(row=1, column=2)

        self.fig, self.ax = plt.subplots(figsize=(12, 6))  # Увеличенный размер
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=4, pady=10)

        self.refresh_files()

    def refresh_files(self):
        self.file_list = [f for f in os.listdir(HISTORY_FOLDER) if f.endswith(".json")]
        self.file_list.sort()
        self.combo["values"] = self.file_list
        if self.file_list:
            self.combo.current(0)
            self.load_history(self.file_list[0])

    def on_select(self, event):
        file = self.combo.get()
        self.current_index = self.file_list.index(file)
        self.load_history(file)

    def load_history(self, filename):
        path = os.path.join(HISTORY_FOLDER, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.history = json.load(f)
            self.show_plot()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")

    def show_plot(self):
        self.ax.clear()
        has_plot = False

        if "loss" in self.history:
            self.ax.plot(self.history["loss"], label="Train Loss")
            has_plot = True

        if "val_loss" in self.history:
            self.ax.plot(self.history["val_loss"], label="Val Loss")
            has_plot = True

        if "learning_rate" in self.history:
            self.ax.plot(self.history["learning_rate"], label="Learning Rate", color="orange")
            has_plot = True

        if has_plot:
            self.ax.set_title("Графики обучения")
            self.ax.set_xlabel("Эпоха")
            self.ax.grid(True)
            self.ax.legend()
        else:
            self.ax.text(0.5, 0.5, "Нет графиков для отображения", ha='center', va='center')

        self.canvas.draw()

    def next_file(self):
        if not self.file_list:
            return
        self.current_index = (self.current_index + 1) % len(self.file_list)
        self.combo.current(self.current_index)
        self.load_history(self.file_list[self.current_index])

    def prev_file(self):
        if not self.file_list:
            return
        self.current_index = (self.current_index - 1) % len(self.file_list)
        self.combo.current(self.current_index)
        self.load_history(self.file_list[self.current_index])

# === Запуск ===
if __name__ == "__main__":
    root = tk.Tk()
    app = HistoryViewer(root)
    root.mainloop()
