from .detect_facemask.detect_facemask import FaceMaskDetector
from .detect_faces.recognize import FaceIdentifier
from .voice_recognition.voice_recognition import SpeechToText
from imutils import resize
from imutils.video import VideoStream
import pickle
import time
from collections import Counter

class WorkPlaceScreening(object):

    def __init__(self):
        self.face_mask_detector = FaceMaskDetector(mask_detect_model='./workplace_screening/facemask_detection_model.tflite')
        self.face_recognizer = FaceIdentifier(encodings_location='./workplace_screening/encodings.pkl',
                                              embeding_model_location='./workplace_screening/face_embedding_model.tflite')
        self.speech_to_text = SpeechToText()

    def fail(self):
        time.sleep(3)
        self.start()

    def load_image(self):
        try:
            with open('./workplace_screening/frame.pkl', 'rb') as myfile: 
                self.frame = pickle.load(myfile)
        except:
            time.sleep(0.1)
            with open('./workplace_screening/frame.pkl', 'rb') as myfile: 
                self.frame = pickle.load(myfile)

    def start(self):
        self.save_text_to_file("STOP! We need to check your mask, temperature and symptoms before you enter.")
        self.load_image()
        self.face_mask_detector.load_image_from_frame(self.frame)
        number_of_faces =  self.face_mask_detector.detect_faces(probability=0.8, face_size=(160,160))
        if number_of_faces>=1:
            self.save_text_to_file("Look directly at the screen. Make sure you can see your whole head.")
            self.wearing_mask()
        else:
            self.fail()
        
    def wearing_mask(self):
        self.face_mask_detector.load_image_from_frame(self.frame)
        number_of_faces =  self.face_mask_detector.detect_faces(probability=0.8, face_size=(224,224))
        wearing_facemask =  self.face_mask_detector.detect_facemask()

        print(f'Wearing facemask: {wearing_facemask}')
        if number_of_faces>=1 and wearing_facemask:
            self.recognize_person()
        else:
            self.save_text_to_file("You are not allowed in without a mask. Please wear your mask.")
            time.sleep(4)
            self.start()
    
    def recognize_person(self):
        
        names = []
        for i in range(10):
            try:
                time.sleep(0.2)
                self.load_image()
                self.face_recognizer.load_image_from_frame(self.frame)
                number_of_faces = self.face_recognizer.detect_faces(probability=0.8, face_size=(160,160))
                recognized_names = self.face_recognizer.recognize_faces(tolerance=0.35, verbose=True, method = 'distance')
                recognized_names.append('Unkown')
                names.append(recognized_names[0])
            except:
                time.sleep(0.2)
                self.load_image()
                self.face_recognizer.load_image_from_frame(self.frame)
                number_of_faces = self.face_recognizer.detect_faces(probability=0.8, face_size=(160,160))
                recognized_names = self.face_recognizer.recognize_faces(tolerance=0.35, verbose=True, method = 'distance')
                recognized_names.append('Unkown')
                names.append(recognized_names[0])
        
        name = Counter(names)
        person = name.most_common(1)[0][0]

        print(f'Recognized {person}')
        if number_of_faces>=1:
            if person != 'Unkown':
                self.recognized_name = person
                self.save_text_to_file(f"Thanks for wearing your mask, {str(self.recognized_name).capitalize()}. Going to take your temperature now.")
                time.sleep(2)
                self.temperature_measure()
            else:
                self.save_text_to_file(f"Thanks for wearing your mask. Going to take your temperature now.")
                time.sleep(2)
                self.recognized_name = 'Unkown'
                self.temperature_measure()
        else:
            self.wearing_mask()

    
    def temperature_measure(self):
        temperature = 37.5

        text = 'Slowly move closer to the box. Keep still until you see the green light and hear a beep. DON’T touch the surface of the box'
        self.save_text_to_file(text)
        time.sleep(2)
        text = f'{temperature} degrees. Thank you.'
        if temperature > 38:
            text = f'You are not allowed in because your temperature ({temperature}) is over 38 degrees. You might have a fever.'
            self.save_text_to_file(text) 
            time.sleep(2)
            text = f'We recommend you self-isolate. Contact the health department if you have any concerns. Thanks for keeping us safe.'
            self.save_text_to_file(text) 
            time.sleep(2)
            self.fail()
        else:
            text = f'Your temperature was {temperature} degrees.'
            self.save_text_to_file(text) 
            time.sleep(2)
            self.question1()

    def question1(self):
        self.save_text_to_file("Answer YES or NO")
        self.speech_to_text.fine_tune(duration=3)
        self.save_text_to_file("""Do you have any of the following: a persistent cough? difficulty breathing? a sore throat?""")
        time.sleep(0.2)
        answer = self.speech_to_text.listen_and_predict(online=False)
        print(f'Answer of question was: {answer}')
        if answer == 'yes':
            text = f'You’re not allowed in because you might have covid-19 symptoms.'
            self.save_text_to_file(text) 
            time.sleep(2)
            text = f'We recommend you self-isolate. Contact the health department if you have any concerns. Thanks for keeping us safe!'
            self.save_text_to_file(text) 
            self.fail()
        elif answer == 'no':
            self.question2()
        else:
            text = f'Sorry, but we could not understand you. Please clearly say YES or NO'
            self.save_text_to_file(text) 
            time.sleep(2)
            self.question1()

    def question2(self):
        self.save_text_to_file("Answer YES or NO")
        self.speech_to_text.fine_tune(duration=2)
        self.save_text_to_file("Have you been in contact with anyone who tested positive for covid-19 in the last 2 weeks?")
        time.sleep(0.2)
        answer = self.speech_to_text.listen_and_predict(online=False)
        print(f'Answer of question was: {answer}')

        if answer == 'yes':
            text = f'You’re not allowed in because you might have covid-19 symptoms.'
            self.save_text_to_file(text) 
            time.sleep(2)
            text = f'We recommend you self-isolate. Contact the health department if you have any concerns. Thanks for keeping us safe!'
            self.save_text_to_file(text) 
            self.fail()
        elif answer == 'no':
            self.passed()
        else:
            text = f'Sorry, but we could not understand you. Please clearly say YES or NO'
            self.save_text_to_file(text) 
            time.sleep(2)
            self.question2()

    def passed(self):
        if self.recognized_name != 'Unkown':
            self.save_text_to_file("All clear! Please sanitise your hands before you enter.")
            time.sleep(2)
            self.start()
        else:
            time.sleep(2)
            self.get_phone_number()

    def passed_unkown(self):
        self.save_text_to_file("All clear! Please sanitise your hands before you enter.")
        self.ringbell()
        time.sleep(2)
        self.start()

    def ringbell(self):
        pass

    def get_phone_number(self):
        self.speech_to_text.fine_tune(duration=3)
        self.save_text_to_file("Please say your contact number in Plain English.")
        time.sleep(0.2)
        phone_number = self.speech_to_text.listen_and_predict(online=False)

        self.speech_to_text.fine_tune(duration=2)
        self.save_text_to_file("Answer YES or NO")
        time.sleep(2)
        self.save_text_to_file(f"Is this your contact number? {phone_number}")
        answer = self.speech_to_text.listen_and_predict(online=False)

        if answer == 'yes':
            self.passed_unkown()
        else:
           self.get_phone_number()


    def save_text_to_file(self, text):
        with open('./workplace_screening/state.pkl', 'wb') as file:
            pickle.dump(text, file)

work_place_screening = WorkPlaceScreening()
# try:
work_place_screening.start()
# except:
#     work_place_screening.save_text_to_file("Unforseen error. Starting over")
#     time.sleep(2)
#     work_place_screening.save_text_to_file("STOP! We need to check your mask, temperature and symptoms before you enter.")
