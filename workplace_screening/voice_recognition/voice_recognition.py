import speech_recognition as sr

class SpeechToText(object):
    """
    Used to detect the answer given from a person
    using speech recongtion. 

    Arguments:
    phrase_threshold {float, default=0.2}: 
        minimum seconds of speaking audio before we consider the 
        speaking audio a phrase - values below this are ignored 
        (for filtering out clicks and pops)
    non_speaking_duration {float, default=0.2}:
         seconds of non-speaking audio to keep on both sides of the recording

    Example of usage:
        voice_recognizer = SpeechToText(phrase_threshold=0.2, non_speaking_duration=0.2)
        energy_threshold = voice_recognizer.fine_tune(duration)
        answer = voice_recognizer.predict()
    """

    def __init__(self):
        # obtain audio from the microphone
        self.recorder = sr.Recognizer()
        with sr.Microphone() as source:
            self.recorder.adjust_for_ambient_noise(source)


    def fine_tune(self, duration = 2):
        """
        This function will start the microphone and adjust
        the minimum energy required to detect a voice based
        on the ambiant noise. 

        Arguments:
            duration {int, default=2}:
                Amount of seconds that need to be listened to 
                to determine the optimum energy threshold. The 
                longer the more accurate, but will mean nothing 
                can happen.
        """

        with sr.Microphone() as source:
            self.recorder.adjust_for_ambient_noise(source, duration=duration)
            return self.recorder.energy_threshold   


    def listen_and_predict(self):
        """
        This function will start the microphone and listen for 
        an answer to be provided and extract the answer from it.

         Returns:
            A string value indicating either yes, no or unkown.
        """
        
        with sr.Microphone() as source:
            audio = self.recorder.listen(source) 

        # recognize speech using Sphinx
        try:
           text = self.recorder.recognize_sphinx(audio)
        except sr.UnknownValueError:
            return 'unkown'
        except sr.RequestError as e:
            return 'unkown with error'

        text_split = text.split()
        print(text)

        if 'yes' in text_split and 'no' in text_split:
            return 'unkown'
        elif 'yes' in text_split:
            return 'yes'
        elif 'no' in text_split:
            return 'no'


        return 'unkown'