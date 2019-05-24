import tkinter as tk
from PIL import ImageTk
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import Label
from PIL import Image
from tkinter import filedialog
from functools import partial
import Stage_1_U_net as stage1
import Stage_2 as stage_2
import Stage_3_Segcaps as stage3
import cropping as crop
import numpy as np
class SampleApp(tk.Frame):
    photolist=[]
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

    def predict_output_helper_stage2(self,file,window):
        model = stage_2.get_model()
        x=stage_2.main_predict(model,file)
        print(x)
        if x>0.7 :
            return 1
        else:
            return 0;

    def predict_output_helper_stage3(self, file, window,truemaskfile):
        model = stage3.get_model()
        arr = stage3.main_predict(model, file,truemaskfile)
        print(arr.shape)
        arr = arr * 255
        outputfilestage3 = r"C:\Users\sakshigarg\Desktop\Liver_disease_demo\a_output\outputafterlesionsegmentation.png"
        print(np.unique(arr))
        image = Image.fromarray(arr.astype('uint8'), 'RGB')
        image.save(outputfilestage3)
        return outputfilestage3

    def predict_output_helper_stage1(self,file,window):
        model = stage1.get_model()
        arr=stage1.main_predict(model,file)
        print(arr.shape)
        #'C:/Users/Soumya/Desktop/EnlargedWithoutBorder_Dataset/enlarged_vol4a.png'
        # save image
        arr=arr*255
        filepath = r"C:\Users\sakshigarg\Desktop\Liver_disease_demo\a_output\output.png"
        #plt.imshow(arr)
        #plt.show()
        print(np.unique(arr))
        image = Image.fromarray(arr.astype('uint8'), 'RGB')
        image.save(filepath)
        return filepath


    def predict_output(self):
        window = tk.Toplevel(root)
        window.title("Input Uploader")
        window.geometry("1000x1200")
        self.InputUploader(window)
        window.mainloop()

        #filepath=self.predict_output_helper()
        #call file loader to plot the image


    def file_loader(self,file,window,ip_column,ip_row):
        print("inside file loader",file)
        raw=Image.open(file)
        raw = raw.resize((128, 128))
        img = ImageTk.PhotoImage(raw)
        self.photolist.append(img)
        print(img)
        canvas = Canvas(window, width=256, height=256)
        canvas.grid(column=ip_column, row=ip_row)
        canvas.create_image(20, 20, anchor=NW, image=img)
    def label_output(self,window):
        label_in = Label(window, text="Input Image", relief=RAISED)
        label_in.grid(column=1, row=2)
        label_in = Label(window, text="Segmented Liver", relief=RAISED)
        label_in.grid(column=2, row=2)
        label_in = Label(window, text="Cropped Image", relief=RAISED)
        label_in.grid(column=3, row=2)
        label_in = Label(window, text="Segmented Lesion", relief=RAISED)
        label_in.grid(column=4, row=2)


    def clicked_chooseme(self, window):
        file = filedialog.askopenfilename(title="Choose a file",
                                          filetypes=[('image files', '.png'), ('image files', '.jpg'), ])
        self.file_loader(file, window, 1, 1)
        filepath_stage1 = self.predict_output_helper_stage1(file, window)
        print("end", filepath_stage1)
        self.file_loader(filepath_stage1, window, 2, 1)
        outputfile = r"C:\Users\sakshigarg\Desktop\Liver_disease_demo\a_output\outputaftercropping.png"
        crop.crop_file(filepath_stage1,file,outputfile)
        self.file_loader(outputfile, window, 3, 1)
        x=self.predict_output_helper_stage2(outputfile,window)
        print("x = ",x)
        if x==0:
            print("Lesion is absent")
            self.label_output(window)
        else:
            stage3output=self.predict_output_helper_stage3(outputfile,window,truemaskfile=file)
            self.file_loader(stage3output, window, 4, 1)
            self.label_output(window)

    def InputUploader(self,window):
        chooseme_with_arg = partial(self.clicked_chooseme, window)
        btn = Button(window, text="Choose Files", command=chooseme_with_arg)
        btn.grid(column=0, row=0)

    def PassCheck(self):
        password = self.entry.get()
        if len(password) >= 9 and len(password) <= 12 and password=='qwertyuiop':
            self.label.config(text="Password is correct")
            #self.InputUploader()
            #self.preprocessing_pipeline() #-> intial dataset
            self.predict_output()
        else:
            self.label.config(text="Password is incorrect")

    # def preprocessing_pipeline(self):
    #     window = tk.Toplevel(root)
    #     window.title("Intial Dataset")
    #     filepath_axial=r"C:\Users\sakshigarg\Desktop\Liver_disease_demo\Dataset\Enlarged_Dataset\vol0a.png"
    #     filepath_coronnal = r"C:\Users\sakshigarg\Desktop\Liver_disease_demo\Dataset\Enlarged_Dataset\vol0c.png"
    #     filepath_sagittal = r"C:\Users\sakshigarg\Desktop\Liver_disease_demo\Dataset\Enlarged_Dataset\vol0s.png"
    #
    #     img_axial = ImageTk.PhotoImage(Image.open(filepath_axial))
    #     img_coronnal = ImageTk.PhotoImage(Image.open(filepath_coronnal))
    #     img_sagittal = ImageTk.PhotoImage(Image.open(filepath_sagittal))
    #
    #     canvas = Canvas(window, width=300, height=300)
    #     canvas.grid(column=0, row=1)
    #     canvas.create_image(20, 20, anchor=NW, image=img_axial)
    #
    #     canvas = Canvas(window, width=300, height=300)
    #     canvas.grid(column=1, row=1)
    #     canvas.create_image(20, 20, anchor=NW, image=img_coronnal)
    #
    #     canvas = Canvas(window, width=300, height=300)
    #     canvas.grid(column=2, row=1)
    #     canvas.create_image(20, 20, anchor=NW, image=img_sagittal)
    #     window.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    run = SampleApp(root)
    root.mainloop()


