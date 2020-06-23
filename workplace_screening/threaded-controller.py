#!/usr/bin/env python3
import sys

# uncomment for local development:
#Replace RPi library with a mock (if you're rnot running on a Pi)
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
import requests
from urllib.parse import quote_plus
import serial
from datetime import datetime
from pathlib import Path
from PIL import Image
import os
import cv2
import pytz
import threading 

BUTTON_GPIO = 16
SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 9600
LOCATION_KEY = os.environ.get('LOCATION_KEY', 'dev')
TELEGRAM_API_KEY = os.environ.get('TELEGRAM_API_KEY')
BACKEND_USERNAME = os.environ.get('BACKEND_USERNAME')
BACKEND_PASSWORD = os.environ.get('BACKEND_PASSWORD')
GLOBAL_RESET = False

def load_image():
    counter = 0
    while counter < 10:
        counter += 1
        try:
            with open('./workplace_screening/frame.pkl', 'rb') as myfile:
                frame = pickle.load(myfile)
                return frame
        except:
            time.sleep(0.1)
            pass

def button_pressed_callback(logger):
    global GLOBAL_RESET
    GLOBAL_RESET = True
    fail(logger, "foot-pedal-interrupt")

def fail(logger=None, 
        reason="unspecified", message="Restarting sequence...",
        recognized_name=None, mask=None, symptoms = None, contact = None, temperature = None):
        logger.save_text_to_file(message)
        logger.log("FAIL: screening sequence failed")
        logger.log(f"reason: {reason}")
        logger.log("")
        logger.log_to_backend(recognized_name, mask, symptoms, contact, temperature)
        time.sleep(5)

class Logger(object):

    def __init__(self):
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

    def log_image(self, frame, reason):
        # check if folder exists
        folder = f'image_logs/{str(datetime.now().date())}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        # convert from BGR to RGB
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # save image
        filename = f'{folder}/{datetime.now().replace(microsecond=0)}_{reason}.jpg'.replace(" ", "_")
        image.save(filename)

    def log_telegram(self, message):
        if TELEGRAM_API_KEY is not None:
            url = f'https://api.telegram.org/bot1229071509:{TELEGRAM_API_KEY}/sendMessage?chat_id=-1001380311183&text={quote_plus(message)}'
            response = requests.get(url)
            if response.status_code != 200:
                self.log(f"Couldn't log to Telegram. Status code: {response.status_code}")
            else:
                self.log(f"Telegram sent: {message}")
        else:
            self.log("skip logging to Telegram... TELEGRAM_API_KEY env variable not set")

    def log_to_backend(self, recognized_name, mask, symptoms, contact, temperature):

        url = "https://workplacescreening.herokuapp.com/api"
        try:
            # get access token
            response = requests.post(url + "/authenticate", json = {
                "password": BACKEND_PASSWORD,
                "username": BACKEND_USERNAME
            })

            if response.status_code != 200:
                self.log(f"Error authentication with backend API. Status code: {response.status_code}")
            else:
                id_token = response.json().get("id_token")

                # post a new record
                headers = {'Authorization': f'Bearer {id_token}'}
                payload = {
                    "eventDateTime": datetime.utcnow().replace(tzinfo=pytz.utc).isoformat(),
                    "firstname": recognized_name,
                    "lastname": "-",  # this is expected by the backend, but we don't use it yet
                    "locationId": LOCATION_KEY,
                    "mask": mask,
                    "symptoms": symptoms,
                    "contact": contact,  # this is a new field... not expected by the backend
                    "temperature": temperature
                }
                response = requests.post(url + "/screeningevents", json=payload, headers=headers)
                if response.status_code != 200:
                    self.log(f"Error posting record to backend API. Status code: {response.status_code}")
        except BaseException as e:
            self.log(f"Exception while trying to log record to backend: {e}")

    def save_text_to_file(self, text):
        self.log(f" -> {text}")
        self.log("")
        try:
            with open('./workplace_screening/state.pkl', 'wb') as file:
                pickle.dump(text, file)
        except:
            # TODO: handle retries in a loop, rather than just once-off
            time.sleep(0.1)
            with open('./workplace_screening/state.pkl', 'wb') as file:
                pickle.dump(text, file)


class Facemask(threading.Thread):

    def __init__(self, logger):
        self.face_mask_detector = FaceMaskDetector(
                    mask_detect_model='./workplace_screening/facemask_detection_model.tflite')

        self.logger = logger

    def check_for_mask(self):
        self.frame = load_image()
        self.face_mask_detector.load_image_from_frame(self.frame)
        number_of_faces = self.face_mask_detector.detect_faces(probability=0.8, face_size=(224, 224))
        wearing_facemask = self.face_mask_detector.detect_facemask(mask_probability=0.97, verbose=True)

        self.logger.log(f'Wearing facemask: {wearing_facemask}')
        if number_of_faces >= 1 and wearing_facemask:
            return True
        else:
            return False

    def get_frame(self):
        return self.frame


class Recognize(threading.Thread, ):

    def __init__(self, logger):
        self.face_recognizer = FaceIdentifier(encodings_location='./workplace_screening/encodings.pkl',
                                              embeding_model_location='./workplace_screening/face_embedding_model.tflite')
        self.logger = logger

    def recognize_person(self):
        time.sleep(0.5)
        self.logger.save_text_to_file("Please wait, recognising...")
        names = []
        counter = 0
        while counter < 50 and len(names) < 5:
            counter += 1
            time.sleep(0.2)
            try:
                self.frame = load_image()
                self.face_recognizer.load_image_from_frame(self.frame)
                # check if a face is present
                number_of_faces = self.face_recognizer.detect_faces(probability=0.8, face_size=(160, 160))
                if number_of_faces >= 1:
                    # only try recognizing faces if a face was present
                    recognized_names = self.face_recognizer.recognize_faces(
                        tolerance=0.41, verbose=False, method='distance')
                    names = names + recognized_names  # append names that were recognized, even duplicates
            except:
                # something went wrong, let's ignore the sample
                pass

        if len(names) > 0:
            name = Counter(names)
            person = name.most_common(1)[0][0]

            return str(person).capitalize()
        else:
            return "Visitor"

    def get_frame(self):
        return self.frame()


class Temperature(threading.Thread, ):

    def __init__(self,logger):
       self.logger = logger

    def measure_temperature(self):

        temperature = None
        # uncomment the next line to skip temperature reading (e.g. for developing locally)
        # temperature = 36.3

        text = 'Please move your head towards the red lights \nuntil they turn green.\nThen hold still until the green lights flash'
        self.logger.save_text_to_file(text)

        if temperature is None and not GLOBAL_RESET:
            # connect to serial port
            try:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
                ser.flush()
            except ConnectionError:
                self.logger.log(f"Cannot connect to Serial {SERIAL_PORT}")
                fail(self.logger, 
                     "serial-port-error", 
                     "There was an error connecting to the temperature sensor. Please try again.",
                     recognized_name, mask, symptoms, contact, temperature)

            # try reading temperature
            count = 0
            sleep_interval = 0.2
            seconds_to_keep_trying = 300  # fail if we couldn't read a temperature on the serial port after this time
            while count < (seconds_to_keep_trying / sleep_interval) and not GLOBAL_RESET:
                data_left = ser.inWaiting()  # check for remaining bytes
                input = ser.read(data_left)
                if input:
                    self.logger.log(f"serial input: {input}")
                    try:
                        temperature = float(input)
                        break
                    except Exception:
                        # We saw something, but it wasn't a float, so keep going
                        pass
                time.sleep(0.2)
                count += 1

            if temperature is None and not GLOBAL_RESET:
                fail(self.logger, 
                     "temperature-reading-timeout",
                      "Couldn't read temperature. Please try again",
                      recognized_name, mask, symptoms, contact, temperature)
            self.logger.log(f"TEMPERATURE MEASURED: {temperature}")
            self.temperature = temperature
        else:
            # this is only called in DEV
            if not GLOBAL_RESET:
                time.sleep(4)
            else:
                print("DID SKIP TO DUE TO GLOBAL RESET")

        if not GLOBAL_RESET:
            return temperature


class Question1(threading.Thread):

    def __init__(self, logger):
        self.speech_to_text = SpeechToText()
        self.logger = logger

    def answer_question(self):
        
        if not GLOBAL_RESET:
            self.speech_to_text.fine_tune(duration=3)
       
        if not GLOBAL_RESET:
            self.logger.save_text_to_file(
                "Have you been in contact with anyone \nwho tested positive for covid-19 \nin the last 2 weeks? \nWait for the instruction to say your answer.")
            time.sleep(5)
            self.logger.save_text_to_file("Answer YES or NO and wait for the response")

        if not GLOBAL_RESET:
            answer = self.speech_to_text.listen_and_predict(online=True, verbose=True)

        if GLOBAL_RESET:
            return 'reset' 
        else:
            return answer


class Question2(threading.Thread, ):

    def __init__(self, logger):
        self.speech_to_text = SpeechToText()
        self.logger = logger

    def answer_question(self):

        if not GLOBAL_RESET:
            self.speech_to_text.fine_tune(duration=3)
    
        if not GLOBAL_RESET:
            self.logger.save_text_to_file(
                "Have you been in contact with anyone \nwho tested positive for covid-19 \nin the last 2 weeks? \nWait for the instruction to say your answer.")
            time.sleep(5)
            self.logger.save_text_to_file("Answer YES or NO and wait for the response")
       
        if not GLOBAL_RESET:
            answer = self.speech_to_text.listen_and_predict(online=True, verbose=True)

        if GLOBAL_RESET:
            return 'reset' 
        else:
            return answer


class Idle(object):

    def __init__(self, logger):
        self.face_mask_detector = FaceMaskDetector(
                    mask_detect_model='./workplace_screening/facemask_detection_model.tflite')
        self.logger = logger
        self.sequence_count = 0

    def wait_for_face(self):

        # keep looping unitl a face is detected
        number_of_faces = 0
        while number_of_faces == 0:
            self.frame = load_image()
            self.face_mask_detector.load_image_from_frame(self.frame)
            number_of_faces = self.face_mask_detector.detect_faces(probability=0.8, face_size=(224, 224))
            time.sleep(0.25)
            # ensure no one just walked by
            if number_of_faces >= 1:
                time.sleep(1)
                self.frame = load_image()
                self.face_mask_detector.load_image_from_frame(self.frame)
                number_of_faces = self.face_mask_detector.detect_faces(probability=0.8, face_size=(224, 224))

        self.sequence_count += 1
        self.logger.log(f"FACE DETECTED: starting sequence #{self.sequence_count}")
        self.start_time = datetime.now().replace(microsecond=0)

        self.logger.save_text_to_file("Look directly at the screen. \nMake sure you can see your whole head.")

        return True


if __name__ == "__main__":
    # setup GPIO for foot pedal
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # initialize state machine instances
    logger = Logger()
    idle_state = Idle(logger)
    detect_facemask_state = Facemask(logger)
    recognise_state = Recognize(logger)
    temperature_reading_state = Temperature(logger)
    question1_state = Question1(logger)
    question2_state = Question2(logger)

    # reset state from foot pedal
    GPIO.add_event_detect(BUTTON_GPIO, GPIO.FALLING,
                          callback=lambda x: button_pressed_callback(logger),
                          bouncetime=500)

    logger.log("RESTARTING: waiting for a face...")    

    # main loop
    while True:
        GLOBAL_RESET = False
        do_not_reset=True
        mask = None
        recognized_name = None
        temperature = None
        symptoms = None
        contact = None
        logger.save_text_to_file("We need to check your mask,\ntemperature and symptoms \nbefore you enter.")

        while do_not_reset and not GLOBAL_RESET:
            
            face_detected = idle_state.wait_for_face()

            if face_detected and do_not_reset and not GLOBAL_RESET:
                start_time = datetime.now()
                mask = detect_facemask_state.check_for_mask()
                
                if mask and not GLOBAL_RESET:
                    recognized_name = recognise_state.recognize_person()

                    if recognized_name == 'Visitor':
                        logger.log("could not recognize anyone")
                        logger.log_image(recognise_state.get_frame(),"unkown-person")
                        logger.save_text_to_file(f"Welcome Visitor.")
                        logger.log_telegram("Visitor at screening station.")
                    else:
                        logger.log(f'Recognized {recognized_name}')
                        logger.save_text_to_file(f"Hi {recognized_name}.")

                    time.sleep(3)
                    logger.save_text_to_file(f"Thanks for wearing your mask. \nGoing to take your temperature now.")
                    time.sleep(3)
                    
                    # starting to read temperature
                    if not GLOBAL_RESET:
                        temperature = temperature_reading_state.measure_temperature()

                    try:
                        if temperature > 38:
                            text = f'You are not allowed in \nbecause your temperature ({temperature}) \nis over 38 degrees. \nYou might have a fever.'
                            logger.save_text_to_file(text)
                            time.sleep(4)
                            fail(logger,
                                "temperature-too-high",
                                "We recommend you self-isolate. \nContact the health department \nif you have any concerns. \nThanks for keeping us safe.",
                                recognized_name, mask, symptoms, contact, temperature)
                            do_not_reset = False
                        else:
                            logger.save_text_to_file(f'Your temperature was {temperature} degrees.')
                            
                            # question 1
                            answer = 'Unkown'
                            while answer=='Unkown' and not GLOBAL_RESET:
                                # starting with speech recognition
                                answer = question1_state.answer_question()

                                if answer == 'no':
                                    answer = 'no'
                                    symptoms = 'no'
                                elif answer == 'yes':
                                    text = f'You are not allowed in \nbecause you might have covid-19 symptoms. \nWe recommend you self-isolate. \nContact the health department \nif you have any concerns. Thanks for keeping us safe!'
                                    logger.save_text_to_file(text)
                                    time.sleep(5)
                                    fail(logger, "question-1-symptoms", text,
                                        recognized_name, mask, symptoms, contact, temperature)
                                    do_not_reset = False
                                elif answer == 'reset':
                                    do_not_reset = False
                                else:
                                    # try again
                                    text = f'Sorry, but we could not understand you. \nYou need to speak clearly when prompted.'
                                    logger.save_text_to_file(text)
                                    time.sleep(2)
                                    answer='Unkown'
                            
                                                    # question 1
                            answer = 'Unkown'
                            while answer=='Unkown' and not GLOBAL_RESET and do_not_reset:
                                # starting with speech recognition
                                answer = question2_state.answer_question()

                                if answer == 'no':
                                    answer = 'no'
                                    contact = 'no'
                                elif answer == 'yes':
                                    text = f'You are not allowed in \nbecause you might have covid-19 symptoms. \nWe recommend you self-isolate. \nContact the health department \nif you have any concerns. Thanks for keeping us safe!'
                                    logger.save_text_to_file(text)
                                    time.sleep(5)
                                    fail(logger, "question-2-contact", text,
                                        recognized_name, mask, symptoms, contact, temperature)
                                    do_not_reset = False
                                elif answer == 'reset':
                                    do_not_reset = False
                                else:
                                    # try again
                                    text = f'Sorry, but we could not understand you. \nYou need to speak clearly when prompted.'
                                    logger.save_text_to_file(text)
                                    time.sleep(2)
                                    answer='Unkown'
                            
                            if answer=='no':
                                logger.save_text_to_file("All clear! \nPlease sanitise your hands before you enter.")
                                duration = datetime.now().replace(microsecond=0) - start_time
                                logger.log(f"SUCCESS: screening passed (duration {duration})")
                                logger.log_telegram(f"Succesfull screening for: {recognized_name}")
                                logger.log("")
                                logger.log_to_backend(recognized_name=recognized_name,
                                                        mask=mask,
                                                        symptoms = 'no',
                                                        contact = 'no',
                                                        temperature = temperature)
                                time.sleep(15)
                            # TODO: prompt for phone number
                    except:
                        pass
                    
                elif not GLOBAL_RESET:
                    logger.save_text_to_file("You are not allowed in without a mask. \nPlease wear your mask.")
                    time.sleep(4)
                    logger.log_image(detect_facemask_state.get_frame(), "no-mask")
                    fail(logger, "no-mask", "", recognized_name, mask, symptoms, contact, temperature)
                    do_not_reset = False





               
