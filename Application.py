#Import libary
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from UISetup import Ui_MainWindow

import cv2
import numpy as np
import tensorflow as tf
from model import build_model

#Define labels:
labels = ['cats', 'dogs']

# Store the result in the Worker class so we can get it in the Main class
res = None 

class PredictWorker(QObject):
    finished = pyqtSignal()
    # progress = pyqtSignal(int)

    def __init__(self, img):
        super(QObject, self).__init__()
        self.img = img

    def run(self):
        #Buiild the model
        model = build_model()
        model.load_weights("model_weight.h5")

        #Try predicting
        y_pred = model.predict(np.array([self.img])) # Predict the img's label
        y_pred = np.where(y_pred > 0.1, 1,0)   #Get the numeric label

        # Get the result 
        global res
        res = labels[y_pred[0][0]]

        self.finished.emit()

class Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        #Setting up UI
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.uploadButton.clicked.connect(self.browseImg)

    def browseImg(self):
        # Let user browse image:
        fname = QFileDialog.getOpenFileName(self, "OpenFile", 'C\\', 'Image files (*.jpg *.jpeg *.png)')

        imgPath = fname[0]
        if (imgPath != ""): #Check whether an image selected
            # Display the image on app's screen
            pixmap = QPixmap(imgPath)
            self.ui.uploadedImg.setPixmap(QPixmap(pixmap))

            # Read image with opencv to convert it to grayscale
            img = cv2.imread(imgPath)   # Read image from the imgPath
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert it to grayscale
            img = cv2.resize(img, [224, 224])   # Resize it so we can let the model predict 
            img = cv2.merge((img,img,img))  # Make it a 3-channels img so the model can work
            img = img / 255 # Rescale the image

            # Let the user know they have to wait
            self.ui.uploadButton.setText("Please wait while we predicting")

            # Using thread in order to pretend the app freeze
            self.thread = QThread()
            self.worker = PredictWorker(img)
            self.worker.moveToThread(self.thread)
            
            # Connect things
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            # Start the thread
            self.thread.start()

            # Disable the button while we predicting
            self.ui.uploadButton.setEnabled(False)
            self.thread.finished.connect(
                lambda: self.ui.uploadButton.setEnabled(True)
            )
            self.thread.finished.connect(
                lambda: self.ui.uploadButton.setText("Browse image")
            )

            # Popup the result message box
            self.thread.finished.connect(self.resultPopUp)
    
    def resultPopUp(self):
        global res

        msg = QMessageBox()
        msg.setWindowTitle("RESULT")
        msg.setText("Our prediction is: " + str(res))

        x = msg.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
