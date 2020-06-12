#!/usr/bin/env python3
import sys

# uncomment for local development:
# # Replace RPi library with a mock (if you're rnot running on a Pi)
# import fake_rpi
# sys.modules['RPi'] = fake_rpi.RPi     # Fake RPi
# sys.modules['RPi.GPIO'] = fake_rpi.RPi.GPIO # Fake GPIO

import RPi.GPIO as GPIO
from detect_facemask.detect_facemask import FaceMaskDetector
from detect_faces.recognize import FaceIdentifier
from voice_recognition.voice_recognition import SpeechToText
from imutils import resize
from imutils.video import VideoStream
import pickle
import time
from collections import Counter
import serial
from datetime import datetime
from pathlib import Path

BUTTON_GPIO = 16
SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 57600


class WorkPlaceScreening(object):

    def __init__(self):
        self.face_mask_detector = FaceMaskDetector(mask_detect_model='./workplace_screening/facemask_detection_model.tflite')
        self.face_recognizer = FaceIdentifier(encodings_location='./workplace_screening/encodings.pkl',
                                              embeding_model_location='./workplace_screening/face_embedding_model.tflite')
        self.speech_to_text = SpeechToText()
        self.sequence_count = 0
        self.start_time = datetime.now().replace(microsecond=0)  # ignore microseconds for the sake of brevity
        # start a new log file
        self.log_file_name = f"log_{self.start_time.isoformat('-')}.txt"
        Path(self.log_file_name).touch()
        print(f"STARTING UP\n{self.start_time.isoformat(' ')}")
        print("---------------------------------\n")

    def log(self, text):
        # print to console
        print(text)
        # print to log file
        with open(self.log_file_name, "a") as log_file:
            text = f"{datetime.now().replace(microsecond=0).isoformat(' ')}\t{text}"
            print(text, file=log_file)

    def fail(self, reason="unspecified", message="Restarting sequence..."):
        self.save_text_to_file(message)
        self.log("FAIL: screening sequence failed")
        self.log(f"reason: {reason}")
        self.log("")
        time.sleep(5)
        # restart the sequence
        self.wait_for_face()

    def button_pressed_callback(self, channel):
        self.fail("foot-pedal-interrupt")

    def load_image(self):
        try:
            with open('./workplace_screening/frame.pkl', 'rb') as myfile: 
                self.frame = pickle.load(myfile)
        except:
            time.sleep(0.1)
            with open('./workplace_screening/frame.pkl', 'rb') as myfile: 
                self.frame = pickle.load(myfile)

    def wait_for_face(self):
        self.log("RESTARTING: waiting for a face...")
        self.save_text_to_file("STOP! We need to check your mask, temperature and symptoms before you enter.")
        
        # keep looping unitl a face is detected
        number_of_faces = 0
        while number_of_faces == 0:
            self.load_image()
            self.face_mask_detector.load_image_from_frame(self.frame)
            number_of_faces = self.face_mask_detector.detect_faces(probability=0.8, face_size=(160,160))
            time.sleep(0.25)
        self.sequence_count += 1
        self.log(f"FACE DETECTED: starting sequence #{self.sequence_count}")
        self.start_time = datetime.now().replace(microsecond=0)
 
        self.save_text_to_file("Look directly at the screen. Make sure you can see your whole head.")
        self.check_for_mask()
        
    def check_for_mask(self):
        self.face_mask_detector.load_image_from_frame(self.frame)
        number_of_faces =  self.face_mask_detector.detect_faces(probability=0.8, face_size=(224,224))
        wearing_facemask =  self.face_mask_detector.detect_facemask(mask_probability=0.97, verbose=True)

        self.log(f'Wearing facemask: {wearing_facemask}')
        if number_of_faces>=1 and wearing_facemask:
            self.recognize_person()
        else:
            self.save_text_to_file("You are not allowed in without a mask. Please wear your mask.")
            time.sleep(4)
            self.fail("no-mask")
    
    def recognize_person(self):
        
        time.sleep(0.5)
        self.save_text_to_file("Please wait, recognising...")
        names = []
        for i in range(10):
            try:
                time.sleep(0.2)
                self.load_image()
                self.face_recognizer.load_image_from_frame(self.frame)
                number_of_faces = self.face_recognizer.detect_faces(probability=0.8, face_size=(160,160))
                recognized_names = self.face_recognizer.recognize_faces(tolerance=0.41, verbose=False, method = 'distance')
                recognized_names.append('Unkown')
                names.append(recognized_names[0])
            except:
                time.sleep(0.2)
                self.load_image()
                self.face_recognizer.load_image_from_frame(self.frame)
                number_of_faces = self.face_recognizer.detect_faces(probability=0.8, face_size=(160,160))
                recognized_names = self.face_recognizer.recognize_faces(tolerance=0.41, verbose=False, method = 'distance')
                recognized_names.append('Unkown')
                names.append(recognized_names[0])
        
        name = Counter(names)
        person = name.most_common(1)[0][0]

        self.log(f'Recognized {person}')
        if number_of_faces>=1:
            if person != 'Unkown':
                self.recognized_name = person
                self.save_text_to_file(f"Hi {str(self.recognized_name).capitalize()}.")
                time.sleep(3)

                self.save_text_to_file(f"Thanks for wearing your mask. Going to take your temperature now.")
                time.sleep(3)
                self.measure_temperature()
            else:
                self.save_text_to_file(f"Welcome Visitor.")
                time.sleep(3)

                self.save_text_to_file(f"Thanks for wearing your mask. Going to take your temperature now.")
                time.sleep(3)
                self.recognized_name = 'Unkown'
                self.measure_temperature()
        else:
            self.check_for_mask()

    
    def measure_temperature(self):

        temperature = None
        # uncomment the next line to skip temperature reading (e.g. for developing locally)
        # temperature = 36.3

        text = '<--  Please move to the Temperature Box'
        self.save_text_to_file(text)
        
        if temperature is None:
            # connect to serial port
            try:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                ser.flush()
            except ConnectionError:
                self.log(f"Cannot connect to Serial {SERIAL_PORT}")
                self.fail("serial-port-error", "There was an error connecting to the temperature sensor. Please try again.")

            # try reading temperature
            count = 0
            sleep_interval = 0.2 
            seconds_to_keep_trying = 300  # fail if we couldn't read a temperature on the serial port after this time
            while count < (seconds_to_keep_trying / sleep_interval):
                data_left = ser.inWaiting()  # check for remaining bytes
                input = ser.read(data_left)
                if input:
                    self.log(f"serial input: {input}")
                    try:
                        temperature = float(input)
                        break
                    except Exception:
                        # We saw something, but it wasn't a float, so keep going
                        pass
                time.sleep(0.2)
                count += 1

            if temperature is None:
                self.fail("temperature-reading-timeout", "Couldn't read temperature. Please try again")
            self.log(f"TEMPERATURE MEASURED: {temperature}")
        else:
            # this is only called in DEV
            time.sleep(4)

        text = f'{temperature} degrees. Thank you.'
        if temperature > 38:
            text = f'You are not allowed in because your temperature ({temperature}) is over 38 degrees. You might have a fever.'
            self.save_text_to_file(text) 
            time.sleep(4)
            self.fail("temperature-too-high", "We recommend you self-isolate. Contact the health department if you have any concerns. Thanks for keeping us safe.")
        else:
            text = f'Your temperature was {temperature} degrees.'
            time.sleep(5)
            self.save_text_to_file(text)
            self.question_1()

    def question_1(self):
        self.speech_to_text.fine_tune(duration=3)
        self.save_text_to_file("Do you have any of the following?\na persistent cough? \ndifficulty breathing? \na sore throat? \nWait for the instruction to say your answer.")
        time.sleep(5)
        self.save_text_to_file("Answer YES or NO and wait for response") 
        answer = self.speech_to_text.listen_and_predict(online=True, verbose=True)
        self.log(f'QUESTION 1 ANSWERED: {answer}')
        if answer == 'yes':
            text = f'You are not allowed in because you might have covid-19 symptoms. We recommend you self-isolate. Contact the health department if you have any concerns. Thanks for keeping us safe!'
            self.save_text_to_file(text) 
            time.sleep(5)
            self.fail("question-1-symptoms", text)
        elif answer == 'no':
            self.question_2()
        else:
            # try again
            text = f'Sorry, but we could not understand you. You need to speak clearly when prompted.'
            self.save_text_to_file(text) 
            time.sleep(2)
            self.question_1()

    def question_2(self):
        self.speech_to_text.fine_tune(duration=2)
        self.save_text_to_file("Have you been in contact with anyone who tested positive for covid-19 in the last 2 weeks? Wait for the instruction to say your answer.")
        time.sleep(5)
        self.save_text_to_file("Answer YES or NO and wait for the response")
        answer = self.speech_to_text.listen_and_predict(online=True, verbose=True)
        self.log(f'QUESTION 2 ANSWERED: {answer}')

        if answer == 'yes':
            text = f'You are not allowed in because you might have covid-19 symptoms. We recommend you self-isolate. Contact the health department if you have any concerns. Thanks for keeping us safe!'
            self.save_text_to_file(text) 
            time.sleep(5)
            self.fail("question-2-contact", text)
        elif answer == 'no':
            self.passed()
        else:
            # try again
            text = f'Sorry, but we could not understand you. Please speak clearly when prompted.'
            self.save_text_to_file(text) 
            time.sleep(2)
            self.question_2()

    def passed(self):
        self.save_text_to_file("All clear! Please sanitise your hands before you enter.")
        duration = datetime.now().replace(microsecond=0) - self.start_time
        self.log(f"SUCCESS: screening passed (duration {duration})")
        self.log("")
        time.sleep(15)
        self.wait_for_face()
        # TODO: prompt for phone number

    # def passed_unkown(self):
    #     self.save_text_to_file("All clear! Please sanitise your hands before you enter.")
    #     self.ringbell()
    #     time.sleep(2)
    #     self.wait_for_face()

    # def ringbell(self):
    #     pass

    # def get_phone_number(self):
    #     self.speech_to_text.fine_tune(duration=3)
    #     self.save_text_to_file("Please say your contact number in Plain English.")
    #     time.sleep(0.2)
    #     phone_number = self.speech_to_text.listen_and_predict(online=False)

    #     self.speech_to_text.fine_tune(duration=2)
    #     self.save_text_to_file("Answer YES or NO")
    #     time.sleep(2)
    #     self.save_text_to_file(f"Is this your contact number? {phone_number}")
    #     answer = self.speech_to_text.listen_and_predict(online=False)

    #     if answer == 'yes':
    #         self.passed_unkown()
    #     else:
    #        self.get_phone_number()


    def save_text_to_file(self, text):
        self.log(f" -> {text}")
        self.log("")
        with open('./workplace_screening/state.pkl', 'wb') as file:
            pickle.dump(text, file)


if __name__ == "__main__":
    # setup GPIO for foot pedal
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # initialize new state machine instance
    controller = WorkPlaceScreening()

    # reset state from foot pedal
    GPIO.add_event_detect(BUTTON_GPIO, GPIO.FALLING,
            callback=controller.button_pressed_callback, bouncetime=500)

    controller.wait_for_face()
