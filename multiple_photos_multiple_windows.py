import tkinter as tk
import os
from PIL import Image, ImageTk

root = tk.Tk()
tk.Label(root, text="this is the root window").pack()
root.geometry("200x200")

photoimage_list = [] # Create a list to hold the PhotoImages!

for i in range(1, 6):
    loc = os.getcwd() + '\pictures\pic%s.jpg'%(i)
    img = Image.open(loc)
    img.load()
    photoimg = ImageTk.PhotoImage(img)
    photoimg.append(photoimage_list) # Add it to a list so it isn't garbage collected!!
    window = tk.Toplevel()
    window.geometry("200x200")
    tk.Label(window, text="this is window %s" % i).pack()
    tk.Label(window, image=photoimg).pack()

root.mainloop()