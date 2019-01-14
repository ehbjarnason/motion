import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np


class Tktest(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        frame = StartPage(container, self)

        self.frames[StartPage] = frame

        print(self.frames)

        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 1, 3, 8, 9, 3, 5])

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


class Window(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.toggle1 = ttk.Button(text="Off",
                command=lambda *args: self.toggle(self.toggle1))
        self.toggle1.pack(padx=5, pady=5)

        self.toggle2 = ttk.Button(text="Off", style="Toolbutton",
                command=lambda *args: self.toggle(self.toggle2))
        self.toggle2.pack(padx=5, pady=5)

        self.toggle3 = ttk.Button(text="Off",
                command=lambda *args: self.toggle(self.toggle3))
        self.toggle3.pack(padx=5, pady=5)
        style = ttk.Style()
        style.configure("Toggle.TButton")
        style.map("Toggle.TButton", relief=[("pressed", "sunken"),
                ("selected", "sunken"), ("!selected", "raised")])
        self.toggle3.config(style="Toggle.TButton")

        self.toggle4 = ttk.Checkbutton(self,
                style="Toggle.TButton", text="Off (4)",
                command=lambda *args: self.toggle_style(self.toggle4, 4))
        self.toggle4.pack(padx=5, pady=5)

        ttk.Button(text="Quit",
                command=self.master.destroy).pack(padx=5, pady=5)
        self.pack()

    def toggle_style(self, button, number):
        if button.instate(("selected",)):
            button.config(text="Off ({})".format(number))
            button.config(style="Toolbutton")
        else:
            button.config(text="On ({})".format(number))
            button.config(style="Toggle.TButton")

    def toggle(self, button):
        if button.instate(("!selected",)):
            button.state(("selected",))
            button.config(text="On")
        else:
            button.state(("!selected",))
            button.config(text="Off")



if __name__ == '__main__':
    # LARGE_FONT = ("Verdana", 12)
    # app = Tktest2()
    # app.mainloop()

    # root = tk.Tk()
    #
    # def toggle(button, variable):
    #     if variable.get():
    #         button.config(text='On')
    #     else:
    #         button.config(text='Off')
    #
    # v1 = tk.BooleanVar()
    # v1.set(False)
    # b1 = ttk.Checkbutton(root, text='Off', variable=v1, indicatoron=False,
    #                      selectcolor='', command=lambda: toggle(b1, v1))
    # b1.pack(padx=50, pady=50)
    #
    # root.mainloop()

    window = Window()
    window.master.title("Toggle")
    window.master.mainloop()
