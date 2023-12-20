
import customtkinter
from PIL import Image
from tkinter import filedialog

import numpy as np
import cv2
#model
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

#kích cỡ ảnh hiển thị
HEIGHT=350
WIDTH=400

def change_scaling_event(new_scaling: str):
    new_scaling_float = int(new_scaling.replace("%", "")) / 100
    customtkinter.set_widget_scaling(new_scaling_float)

# pre_model=None

def apply_grayscale(image,height,width,key=0):#model nhận hình đen trắng
    new_image=image.copy()
    if image.mode=='RGB':
        new_image=new_image.convert("L")
    new_image=new_image.resize((height,width),resample=Image.BILINEAR)
    new_image=np.array(new_image)
    new_image=np.uint8(new_image)
    if key==1:
        new_image=new_image.reshape(1,height,width,1)
    else:
        new_image=new_image.reshape(1,height,width,1).repeat(3,axis=-1)
    new_image=new_image/255.0
    print('image_shape: ',new_image.shape)
    return new_image

def apply_color(image,height,width):#model nhận hình màu
    new_image=image.copy()
    new_image=new_image.resize((height,width),resample=Image.BILINEAR)
    new_image=np.array(new_image)
    new_image=np.uint8(new_image)
    if image.mode=='L':
        new_image=new_image.reshape(1,height,width,1).repeat(3,axis=-1)
    else:
        new_image=new_image.reshape(1,height,width,3)
    new_image=new_image/255.0
    print('image_shape: ',new_image.shape)
    return new_image

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        #create model
        self.model=None
        self.image=None
        self.old_image=None
        self.classes=None
        self.mode=None

        # configure window
        self.test_label = None
        self.title("Xử lý ảnh")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure(0, weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Phân loại giống chó",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_load_img = customtkinter.CTkButton(self.sidebar_frame, command=self.load_image, text="Load image")
        self.btn_load_img.grid(row=1, column=0, padx=20, pady=10)
        self.btn_load_model = customtkinter.CTkButton(self.sidebar_frame, command=self.load_model, text="Load model")
        self.btn_load_model.grid(row=2, column=0, padx=20, pady=10)

        self.model_status=customtkinter.CTkLabel(self.sidebar_frame,text="Empty",
                                                 font=customtkinter.CTkFont(size=15, weight="bold"))
        self.model_status.grid(row=3,column=0,padx=20,pady=10)

        self.btn_predict = customtkinter.CTkButton(self.sidebar_frame, command=self.predict, text="Predict")
        self.btn_predict.grid(row=4, column=0, padx=20, pady=10)

        self.btn_change = customtkinter.CTkButton(self.sidebar_frame, command=self.change, text="Color")
        self.btn_change.grid(row=5, column=0, padx=20, pady=10)

        self.btn_equal = customtkinter.CTkButton(self.sidebar_frame, command=self.apply_clahe_16, text="CLAHE16(Gray)")
        self.btn_equal.grid(row=6, column=0, padx=20, pady=10)

        self.btn_median = customtkinter.CTkButton(self.sidebar_frame, command=self.apply_median, text="Median")
        self.btn_median.grid(row=7, column=0, padx=20, pady=10)

        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_option_menu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                                       values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_option_menu.grid(row=9, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=10, column=0, padx=20, pady=(10, 0))
        self.scaling_o_menu = customtkinter.CTkOptionMenu(self.sidebar_frame,
                                                          values=["80%", "90%", "100%", "110%", "120%"],
                                                          command=change_scaling_event)
        self.scaling_o_menu.grid(row=11, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="Output")
        self.entry.grid(row=3, column=1, columnspan=2, padx=20, pady=20, sticky="nsew")

        self.scaling_o_menu.set("100%")
        self.frame = customtkinter.CTkFrame(self)
        self.frame.grid(row=0, column=1, padx=20, pady=20)
        self.forecast_image = customtkinter.CTkLabel(self.frame, text="")
        self.forecast_image.grid(row=0, column=0, sticky="nsew")

        self.label_image = customtkinter.CTkLabel(self.frame, text="")
        self.label_image.grid(row=0, column=2, sticky="nsew")

    def change(self):
        if self.image is None:
            return
        if self.image.mode=="RGB":
            pure_image = self.image.copy()
            self.old_image=self.image.copy()
            pure_image=pure_image.convert("L")
        else:
            pure_image=self.old_image.copy()
        w_image = WIDTH
        h_image = HEIGHT
        image = customtkinter.CTkImage(light_image=pure_image, dark_image=pure_image,size=(w_image, h_image))
        self.forecast_image.configure(image=image)
        self.image=pure_image
        print('old',self.old_image.mode)
        print('image',self.image.mode)

    def apply_clahe_16(self):
        temp_image=self.image.copy()
        if temp_image.mode=="L":
            temp_image=np.array(temp_image)
            clahe_16=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(16,16))
            temp_image=clahe_16.apply(temp_image)
            temp_image=Image.fromarray(temp_image)
            w_image = WIDTH
            h_image = HEIGHT
            image = customtkinter.CTkImage(light_image=temp_image, dark_image=temp_image,size=(w_image, h_image))
            self.forecast_image.configure(image=image)
            self.image=temp_image

    def apply_median(self):
        temp_image=self.image.copy()
        if temp_image.mode=="L":
            temp_image=np.array(temp_image)
            temp_image=cv2.medianBlur(temp_image,3)
            temp_image=Image.fromarray(temp_image)
            w_image = WIDTH
            h_image = HEIGHT
            image = customtkinter.CTkImage(light_image=temp_image, dark_image=temp_image,size=(w_image, h_image))
            self.forecast_image.configure(image=image)
            self.image=temp_image

    def up_label_image(self,label):
        try:
            file_path_image = 'image/'+label+'.jpg'
            pure_image = Image.open(file_path_image)
            w_image = WIDTH
            h_image = HEIGHT
            image = customtkinter.CTkImage(light_image=pure_image, dark_image=pure_image,size=(w_image, h_image))
            self.label_image.configure(image=image)
            self.label_image.grid()
        except Exception as e:
            print(e)
            self.entry.insert("end","==> error! ")
            self.entry.insert("end",e)

    def load_image(self):
        try:
            
            file_path_image = filedialog.askopenfilename()
            pure_image = Image.open(file_path_image)
            pure_image = pure_image.convert("RGB")
            w_image = WIDTH
            h_image = HEIGHT
            image = customtkinter.CTkImage(light_image=pure_image, dark_image=pure_image,size=(w_image, h_image))
            self.forecast_image.configure(image=image)
            self.forecast_image.grid()
            self.label_image.grid_remove()

            self.image=pure_image#lưu lại ảnh
            
            
        except Exception as e:
            print(e)
            # self.image = None
            # self.forecast_image.configure(image=None)

    def load_model(self):
        try:
            file_path_model=filedialog.askopenfilename()
            if 'resnetv2' in file_path_model:
                self.model=keras.models.load_model(file_path_model,custom_objects={'KerasLayer':hub.KerasLayer})
            else:
                self.model=keras.models.load_model(file_path_model)
            if 'Gray' in file_path_model:
                self.classes=['beagle', 'chihuahua', 'chow', 'cocker_spaniel', 'french_bulldog', 'german_shepherd', 'giant_schnauzer', 'golden_retriever', 'great_dane', 'labrador_retriever', 'malamute', 'miniature_poodle', 'miniature_schnauzer', 'other', 'pembroke', 'pomeranian', 'pug', 'rottweiler', 'samoyed', 'shih-tzu', 'siberian_husky', 'standard_poodle', 'standard_schnauzer', 'toy_poodle']
                self.mode='Gray'
            elif 'Color' in file_path_model:
                self.classes=['chihuahua','miniature_schnauzer','golden_retriever','rottweiler','great_dane','siberian_husky','pug','pembroke','other']
                self.mode='Color'
            name_file=file_path_model.split('/')[-1]
            self.model_status.configure(text=name_file)
        except Exception as e:
            print(e)
            self.model = None
            self.model_status.configure(text="Empty")
        # pre_model=self.model
    
    def predict(self):
        if self.model is None or self.image is None:
            self.entry.delete(0,customtkinter.END)
            self.entry.insert("end","please load image and model!!")
        else:
            if self.mode=='Gray':
                if 'cnn' in self.model_status._text:
                    pred_image=apply_grayscale(self.image,224,224,1)
                else:
                    pred_image=apply_grayscale(self.image,224,224)
            else:
                pred_image=apply_color(self.image,224,224)
            predict=self.model.predict(pred_image)
            y_pred=np.argmax(predict,axis=1)[0]
            print(predict)
            max=np.round(np.max(predict)*100,2)
            self.entry.delete(0,customtkinter.END)
            self.entry.insert("end","==>"+self.classes[y_pred]+'    '+str(max)+'%')
            self.up_label_image(self.classes[y_pred])

    @staticmethod
    def change_appearance_mode_event(new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)


if __name__ == "__main__":
    app = App()
    app.mainloop()
