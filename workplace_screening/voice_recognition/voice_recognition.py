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


    def listen_and_predict(self, online = False, key=None, verbose=False):
        """
        This function will start the microphone and listen for 
        an answer to be provided and extract the answer from it.

        Arguments:
            online {boolean, default=False}:
                If set to true, prediction will be made online
                using the google cloud api, which is the most 
                accurate implementation. If set to false, the 
                pocketsphinx api will be used which works offline,
                but does not perform as good as the online version.

            key {str, default=None}:
                This is not required for the online use as there
                is a generic key that can be used. But this key
                kan be revoked at any time by google so to be safe 
                a personal key should be obtained. To obtain your own
                API key, simply following the steps on the  API Keys 
                <http://www.chromium.org/developers/how-tos/api-keys> page at
                the Chromium Developers site. In the Google Developers 
                Console, Google Speech Recognition is listed as "Speech API".

         Returns:
            A string value indicating either yes, no or unkown.
        """
        
        with sr.Microphone(sample_rate=48000) as source:
            audio = self.recorder.listen(source) 

        # recognize speech using Sphinx
        print('predicting text')
        try:
            if online: 
                text = self.recorder.recognize_google(audio, key=key)
            else:
                text = self.recorder.recognize_sphinx(audio)
        except sr.UnknownValueError:
            return 'Model could not understand the audio'
        except sr.RequestError as e:
            return 'Error when predicting the text'
        
        text_split = text.split()
        
        # acceptable responses for yes and no
        acceptable_yes = set(['yes', 'yet', 'yeah', 'ys', 'yup', 'jip', 'yea', 'yep', 'ja', 'yah'])
        acceptable_no = set(['no', 'know', 'nope', 'not really', 'never'])

        if verbose:
            print(f'Model predicted: {text}')

        if bool(set(text_split).intersection(acceptable_yes)) and bool(set(text_split).intersection(acceptable_no)):
            return 'unkown'
        elif bool(set(text_split).intersection(acceptable_yes)):
            return 'yes'
        elif bool(set(text_split).intersection(acceptable_no)):
            return 'no'

        return 'unkown'
