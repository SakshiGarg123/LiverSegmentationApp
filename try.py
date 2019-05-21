from tkinter import *
from PIL import ImageTk,Image
root = Tk()
canvas = Canvas(root, width = 300, height = 300)
canvas.pack()
img = ImageTk.PhotoImage(Image.open(r"C:/Users/sakshigarg/PycharmProjects/Brats/sample_data/Brats18_2013_2_1/Brats18_2013_2_1_seg/Brats18_2013_2_1_seg_z_label1_090.png"))
canvas.create_image(20, 20, anchor=NW, image=img)
root.mainloop()