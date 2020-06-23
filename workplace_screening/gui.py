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
import pickle
from voice_recognition.voice_recognition import SpeechToText


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
    fontScale = 1
    lineheight = 35
    boxheight = 130
    color = (255, 255, 255)
    thickness = 2
    height, width, dim = image.shape
    org = (lineheight, height-140)
    number_of_characters = len(text)
    print(number_of_characters)
    if number_of_characters < 45:
        image = cv2.rectangle(image, (0, height), (0 + width, height-boxheight-1*lineheight ), (0, 0, 0), -1)
        return cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        #number_of_sentences = math.ceil(number_of_characters/45)
        #words_split = text.split()
        #sentences = np.array_split(words_split, number_of_sentences)
        sentences = text.splitlines()
        image = cv2.rectangle(image, (0, height), (0 + width, height-boxheight-len(sentences)*lineheight ), (0, 0, 0), -1)

        for i, sentence in enumerate(reversed(sentences)):
            #sentence = " ".join(sentence)
            org = (lineheight, height-boxheight-i*lineheight)
            image = cv2.putText(image, sentence, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return image


class WorkPlaceScreening(object):
    def __init__(self, vs):

        self.vs = vs

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
        frame_counter = 0
        text = "We need to check your mask, temperature and symptoms before you enter."

        while not self.stopEvent.is_set():

            try:
                with open('./workplace_screening/state.pkl', 'rb') as myfile:
                    text = pickle.load(myfile)
            except:
                pass

            if frame_counter == 5:
                try:
                    with open('./workplace_screening/frame.pkl', 'wb') as filetowrite:
                        pickle.dump(self.frame, filetowrite)
                except:
                    pass
                frame_counter = 0

            self.frame = self.vs.read()
            self.frame = resize(self.frame, width=800, height = 600)
            image = cv2.flip(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB), 1)
            image = add_text_to_image(image, text)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            if self.panel is None:
                self.panel = tki.Label(image=image)
                self.panel.image = image
                self.panel.pack(fill='both')
            else:
                self.panel.configure(image=image)
                self.panel.image = image

            frame_counter = frame_counter + 1

    def onClose(self):
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()


if __name__ == "__main__":

    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    wps = WorkPlaceScreening(vs)
    wps.root.attributes("-fullscreen", True)
    wps.root.mainloop()
