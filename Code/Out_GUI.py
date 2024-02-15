import customtkinter as ct
import tkinter as tk
import tkinter.messagebox
from tkinter import *

class HeartRatePredictor:

    def __init__(self):
        self.app = ct.CTk()
        self.app.geometry("500x350")
        self.app.title('Heart Rate Prediction')
        #self.pred = Outputs #Placeholder prediction value

    def set_prediction(self, pred):
        self.pred = pred  # Update prediction value

    def create_ui(self):
        self.frame = ct.CTkFrame(master=self.app)
        self.frame.pack(pady=20, padx=60, fill="x", expand=True)

        label_text = ""
        if self.pred >= 0.5:
            label_text = "You have a risk for having a Heart Attack."
            #predictor.set_prediction(1) 
        else:
            label_text = "You're Healthy!"
            #predictor.set_prediction(0)

        self.label = ct.CTkLabel(master=self.frame, text=label_text, width=30, height=5,
                                 corner_radius=8, justify=tk.CENTER)
        self.label.pack(pady=12, padx=10)

        #self.create_ui()  # Recreate UI with updated text

    def main(self):
        self.create_ui()
        self.app.mainloop()

#Instantiate and run the application
#predictor = HeartRatePredictor()
# Assign your prediction value to pred. Example:
#Set pred to 1 (indicating high risk)
#predictor.set_prediction(1)
# or predictor.set_prediction(0)  # Set pred to 0 (indicating healthy)
#predictor.main()