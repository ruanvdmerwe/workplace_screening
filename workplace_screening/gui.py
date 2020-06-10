from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
from imutils import resize
import cv2
import os
from imutils.video import VideoStream
import argparse
import time
import math
import numpy as np

def add_text_to_image(image, text):
    """
    Will ad text to image to display in GUI. Depending on length of text,
    the location of text might need to change.

    Arguments:
        image {opencv image instance}:
            Image on which the text will be put
        text {str}:
            Text to be displayed on the image

    Returns:
        opencv image instance
    """

    font = cv2.FONT_HERSHEY_SIMPLEX 
    org = (20, 400) 
    fontScale = 0.7
    color = (255, 0, 0) 
    thickness = 1
    height, width, dim = image.shape
    number_of_characters = len(text)

    
    if number_of_characters<42:
       return cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA) 
    else:
        number_of_sentences = math.ceil(number_of_characters/42)
        words_split = text.split()
        sentences = np.array_split(words_split, number_of_sentences)
        image = cv2.rectangle(image, (0, height), (0 + width, height-50-len(sentences)*25 - 25), (0,0,0), -1)

        for i, sentence in enumerate(reversed(sentences)):
            sentence = " ".join(sentence)
            org = (20, height-50-i*25)
            image = cv2.putText(image, sentence, org, font, fontScale, color, thickness, cv2.LINE_AA) 
    return image

class WorkPlaceScreening(object):
    def __init__(self, vs, outputPath):

        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # start a thread that constantly pools the video sensor for the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("Workplace Screening")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)


    def videoLoop(self):
        #try:
            while not self.stopEvent.is_set():
                self.frame = self.vs.read()
                self.frame = resize(self.frame, width=900)

                text = "STOP! We need to check your mask, temperature and symptoms before you enter."
                restart = False
                
                while not restart:
                    self.frame = self.vs.read()
                    self.frame = resize(self.frame, width=900)
                    image = cv2.flip(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), 1)
                    image = add_text_to_image(image, text)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    if self.panel is None:
                        self.panel = tki.Label(image=image)
                        self.panel.image = image
                        self.panel.pack(side="left", padx=1, pady=1)
                    else:
                        self.panel.configure(image=image)
                        self.panel.image = image
                    
                    with open ('predictions.txt', 'rt') as myfile:  # Open lorem.txt for reading text
                        text = myfile.read()
                    print(text)
        # except:
        #     print("There was an error")
            
    def onClose(self):
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()


vs = VideoStream(src=0).start()
time.sleep(2.0)
wps = WorkPlaceScreening(vs, './faces/')
#wps.root.attributes("-fullscreen", True)
wps.root.mainloop()

