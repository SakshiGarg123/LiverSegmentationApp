import tkinter as tk
from PIL import ImageTk
from tkinter import *
from tkinter import ttk
from tkinter import Label
from PIL import Image
from tkinter import filedialog
from functools import partial

class SampleApp(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.initialize_user_interface()

    def initialize_user_interface(self):
        self.parent.geometry("200x200")
        self.parent.title("Password checker")
        self.entry = tk.Entry(self.parent)
        self.entry.pack()
        self.button = tk.Button(self.parent, text="Enter", command=self.PassCheck)
        self.button.pack()
        self.label = tk.Label(self.parent, text="Please a password")
        self.label.pack()

    def clicked_chooseme(self,window):
        file = filedialog.askopenfilename(title="Choose a file",
                                          filetypes=[('image files', '.png'), ('image files', '.jpg'), ])
        self.file_loader(file,window)

    def predict_output(self,file):
        img = Image.open(file)

        #save image
        filepath=""
        return filepath
        #call file loader to plot the image
    

    def file_loader(self,file,window):
        print(file)
        img = ImageTk.PhotoImage(Image.open(file))
        print(img)
        canvas = Canvas(window, width=300, height=300)
        canvas.grid(column=1, row=1)
        canvas.create_image(20, 20, anchor=NW, image=img)
        window.mainloop()

    def InputUploader(self):
        window = tk.Toplevel(root)
        window.title("Input Uploader")
        chooseme_with_arg = partial(self.clicked_chooseme, window)
        btn = Button(window, text="Choose Files", command=chooseme_with_arg)
        btn.grid(column=1, row=0)

    def PassCheck(self):
        password = self.entry.get()
        if len(password) >= 9 and len(password) <= 12:
            self.label.config(text="Password is correct")
            self.InputUploader()
        else:
            self.label.config(text="Password is incorrect")

if __name__ == "__main__":
    root = tk.Tk()
    run = SampleApp(root)
    root.mainloop()


