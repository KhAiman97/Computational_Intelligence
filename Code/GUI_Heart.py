import customtkinter
import tkinter as tk
from tkinter import *

class HeartRatePredictionApp:
    def __init__(self, master):
        customtkinter.set_appearance_mode("system")
        customtkinter.set_default_color_theme("dark-blue")
        
        self.master = master
        self.master.title('Heart Rate Prediction')

        self.text_var = StringVar()
        self.Stress_var = StringVar()
        self.prediction = 1
        self.self_info = []

        self.create_widgets()

    def create_widgets(self):
        self.frame = customtkinter.CTkFrame(master=self.master)
        self.frame.pack(pady=0, padx=20, fill="both", expand=True)

        self.label = customtkinter.CTkLabel(master=self.frame, text="Heart Attack Prediction", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.label.pack(pady=12, padx=10)

        self.entry1 = customtkinter.CTkEntry(master=self.frame, placeholder_text="BMI")
        self.entry1.pack(pady=12, padx=10)

        self.entry2 = customtkinter.CTkEntry(master=self.frame, placeholder_text="Cholestrol Level")
        self.entry2.pack(pady=12, padx=10)

        self.entry3 = customtkinter.CTkEntry(master=self.frame, placeholder_text="HeartRate")
        self.entry3.pack(pady=12, padx=10)

        self.entry4 = customtkinter.CTkEntry(master=self.frame, placeholder_text="Tryglcerin")
        self.entry4.pack(pady=12, padx=10)
        
        self.slider_frame = customtkinter.CTkFrame(master=self.frame)
        self.slider_frame.pack(pady=12, padx=10)

        self.slider_label = customtkinter.CTkLabel(master=self.slider_frame, text="Stress Level:")
        self.slider_label.pack(pady=5, padx=10)

        self.stress_level = IntVar()
        self.stress_slider = customtkinter.CTkSlider(master=self.slider_frame, variable=self.stress_level, from_=0, to=10)
        self.stress_slider.pack(pady=5, padx=10)

        self.stress_var = StringVar()
        self.stress_var.set("NO")  # Default selection
        
        self.label = customtkinter.CTkLabel(master=self.frame , text = "Previous Heart Attack",width=12,height=25,#fg_color=("White","White"),
                                corner_radius=8)
        self.label.pack(pady=12, padx=10)
        
        self.radio_yes = customtkinter.CTkRadioButton(master=self.frame, text="YES", variable=self.stress_var, value=1)
        self.radio_yes.pack(pady=12, padx=10)

        self.radio_no = customtkinter.CTkRadioButton(master=self.frame, text="NO", variable=self.stress_var, value=0)
        self.radio_no.pack(pady=12, padx=10)


        self.button = customtkinter.CTkButton(master=self.frame, text="Submit", command=self.button_function)
        self.button.pack(pady=12, padx=10)

        self.label1 = customtkinter.CTkLabel(master=self.frame, textvariable=self.text_var, width=12, height=25, corner_radius=8)
        self.label1.pack(pady=12, padx=10)

    def button_function(self):
        bmi = float(self.entry1.get())
        Cholestrol = float(self.entry2.get())
        HeartRate = float(self.entry3.get())
        Tryglycerin = float(self.entry4.get())
        StressLevel = float(self.stress_level.get())
        Prev =float(self.stress_var.get())
        self.self_info = [bmi, Cholestrol, HeartRate,StressLevel,Tryglycerin, Prev]
        self.master.quit()
    
if __name__ == "__main__":
    root = tk.Tk()
    app = HeartRatePredictionApp(root)
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Calculate the x and y coordinates for the top-right corner
    x_coordinate = screen_width - 500  # Adjust 500 according to your window width
    y_coordinate = 0
    
    # Set the geometry of the window
    root.geometry(f"500x600+{x_coordinate}+{y_coordinate}")
    
    root.mainloop()
