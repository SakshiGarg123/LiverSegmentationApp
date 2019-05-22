import tkinter as tk
import os
from PIL import Image, ImageTk

root = tk.Tk()
tk.Label(root, text="this is the root window").pack()
root.geometry("200x200")

for i in range(1, 6):
    loc = r"C:/Users/sakshigarg/PycharmProjects/Brats/sample_data/Brats18_2013_2_1/Brats18_2013_2_1_seg/Brats18_2013_2_1_seg_z_label1_090.png"
    img = Image.open(loc)
    img.load()
    photoimg = ImageTk.PhotoImage(img)
    window = tk.Toplevel()
    window.geometry("200x200")
    tk.Label(window, text="this is window %s" % i).pack()
    tk.Label(window, image=photoimg).pack()


root.mainloop()