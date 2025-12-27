import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2 as cv
from frames import Frames
from displayTumor import DisplayTumor
from predictTumor import predictTumor


class Gui:
    def __init__(self):
        self.MainWindow = tk.Tk()
        self.MainWindow.geometry('1200x720')
        self.MainWindow.resizable(width=False, height=False)
        


        self.mriImage = None
        self.DT = DisplayTumor()
        self.fileName = tk.StringVar()

        self.FirstFrame = Frames(self, self.MainWindow, 1180, 700, 0, 0)
        self.FirstFrame.btnView['state'] = 'disable'

        self.listOfWinFrame = [self.FirstFrame]

        WindowLabel = tk.Label(self.FirstFrame.getFrames(), text="Brain Tumor Detection", height=1, width=40)
        WindowLabel.place(x=320, y=30)
        WindowLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"))

        self.val = tk.IntVar()
        RB1 = tk.Radiobutton(self.FirstFrame.getFrames(), text="Detect Tumor", variable=self.val,
                             value=1, command=self.check)
        RB1.place(x=250, y=200)
        RB2 = tk.Radiobutton(self.FirstFrame.getFrames(), text="View Tumor Region",
                             variable=self.val, value=2, command=self.check)
        RB2.place(x=250, y=250)

        browseBtn = tk.Button(self.FirstFrame.getFrames(), text="Browse", width=8, command=self.browseWindow)
        browseBtn.place(x=800, y=550)

        self.MainWindow.mainloop()

    def browseWindow(self):
        FILEOPENOPTIONS = dict(defaultextension='*.*',
                               filetypes=[('jpg', '*.jpg'), ('png', '*.png'), ('jpeg', '*.jpeg'), ('All Files', '*.*')])
        self.fileName = filedialog.askopenfilename(**FILEOPENOPTIONS)
        if self.fileName:
            image = Image.open(self.fileName)
            self.mriImage = cv.imread(self.fileName, 1)
            self.listOfWinFrame[0].readImage(image)
            self.listOfWinFrame[0].displayImage()
            self.DT.readImage(image)
            self.FirstFrame.btnView['state'] = 'active'  # Enable the "View Tumor Region" button

    def check(self):
        if self.val.get() == 1:
            self.listOfWinFrame = [self.FirstFrame]
            self.listOfWinFrame[0].setCallObject(self.DT)

            if self.mriImage is not None:
                res = predictTumor(self.mriImage)

                if res > 0.5:
                    resLabel = tk.Label(self.FirstFrame.getFrames(), text="Tumor Detected", height=1, width=20)
                    resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="red")
                else:
                    resLabel = tk.Label(self.FirstFrame.getFrames(), text="No Tumor", height=1, width=20)
                    resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="green")

                resLabel.place(x=700, y=450)
            else:
                print("No MRI image loaded")

        elif self.val.get() == 2:
            self.listOfWinFrame = [self.FirstFrame]
            self.listOfWinFrame[0].setCallObject(self.DT)
            self.listOfWinFrame[0].setMethod(self.DT.removeNoise)
            secFrame = Frames(self, self.MainWindow, 1180, 700, self.DT.displayTumor, self.DT)
            self.listOfWinFrame.append(secFrame)

            for i in range(len(self.listOfWinFrame)):
                if i != 0:
                    self.listOfWinFrame[i].hide()
            self.listOfWinFrame[0].unhide()

            if len(self.listOfWinFrame) > 1:
                self.listOfWinFrame[0].btnView['state'] = 'active'

        else:
            print("Not Working")


if __name__ == "__main__":
    mainObj = Gui()
