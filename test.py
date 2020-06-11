from workplace_screening import FaceMaskDetector, FaceIdentifier, FaceIdentifyDataCreation, SpeechToText
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run and test the facial recogntion/mask detection models and functions')

parser.add_argument('-m',
                    '--mask',
                    #required=True,
                    help='location of mask detection model')
parser.add_argument('-f',
                    '--faces',
                    #required=True,
                    help='location of the face encodings pkl file')
parser.add_argument('-e',
                    '--embedding',
                    #required=True,
                    help='location of the face embeddings model')
parser.add_argument('-t',
                    '--tolerance',
                    nargs='?',
                    const = 0.35,
                    default=0.35,
                    help='tolerance to match a face to known encodings')
parser.add_argument('--probability',
                    nargs='?',
                    const = 0.7,
                    default=0.7,
                    help='minimum probability of face to detect a face')
parser.add_argument('-n',
                    '--known',
                    help='location of the known face images folders')
parser.add_argument('-p',
                    '--person',
                    help='name of the new person to add to face encodings')
parser.add_argument('-l',
                    '--livestream',
                    nargs='?',
                    const = False,
                    required=True,
                    help='If true, then a livestream will be started. If stream has started press q to stop.')
parser.add_argument('--voice',
                    nargs='?',
                    const = False,
                    required=True,
                    help='If true, then test voice to text')
parser.add_argument('-o',
                    '--online',
                    nargs='?',
                    const = False,
                    required=True,
                    help='If true, then test voice to text')
parser.add_argument('-v',
                    '--verbose',
                    nargs='?',
                    const = False,
                    help='If true, then print more information')
parser.add_argument('--method',
                    nargs='?',
                    required=False,
                    default='distance',
                    help='If distance use cosine to recognize else if SVM use SVM model')

args = vars(parser.parse_args())
mask_detect_model = args['mask']
encodings_location = args['faces']
embeding_model_location = args['embedding']
faces_folder = args['known']
person_name = args['person']
tolerance = float(args['tolerance'])
face_probability = float(args['probability'])
livestream = args['livestream']
voice = args['voice']
online = args['voice']
verbose = args['verbose']
method = args['method']

if verbose == 'True':
    verbose = True
else:
    verbose = False

if voice=='True':
    voice = True
else:
    voice = False

if livestream=='True':
    livestream=True
else:
    livestream=False

if online == 'True':
    online = True
else:
    online = False

if __name__ == '__main__':
        
    # ----------------- detecting if you are wearing a face mask -----------------
    if mask_detect_model is None:
        pass
    else:
            
        face_mask_detector = FaceMaskDetector(mask_detect_model=mask_detect_model)
        face_mask_detector.start_video_stream()
        if livestream:
            face_mask_detector.capture_frame_and_detect_facemask_live(verbose=verbose)
        else:
            face_mask_detector.capture_frame_and_detect_facemask(verbose=verbose)
            face_mask_detector.display_predictions()

    # # ----------------- Adding a new face to faces dataset -----------------
    if not faces_folder is None:
        face_embedding = FaceIdentifyDataCreation(encodings_location=encodings_location,
                                                embeding_model_location=embeding_model_location)
        if not person_name is None:
            face_embedding.start_video_stream()
            print("Adding new face pictures")
            face_embedding.capture_frame_and_add_face(faces_folder=faces_folder, person_name=person_name, amount_of_examples=6)
        print("Embedding new face features")
        face_embedding.encode_faces(image_path=faces_folder, model=True)


    # ----------------- See if face is recognized -----------------
    if embeding_model_location is None:
        pass
    else:
        print("Loading face embed model")
        face_recognizer = FaceIdentifier(encodings_location=encodings_location,
                                         embeding_model_location=embeding_model_location)
        print("Starting to recognize face")
        face_recognizer.start_video_stream()
        if livestream:
            face_recognizer.capture_frame_and_recognize_faces_live(tolerance=tolerance,verbose=verbose, method = method)
        else:
            face_recognizer.capture_frame_and_recognize_faces(tolerance=tolerance, face_probability=face_probability,
                                                              verbose=verbose, method = method)
            face_recognizer.display_predictions()

    # ----------------- Test a few phrases -----------------
    # just run a few voice tests
    if voice:
        speech_to_text = SpeechToText()
        energy_threshold = speech_to_text.fine_tune(duration=2)
        for i in range(5):
            print('Fine tuning microphone for 1 second')
            speech_to_text.fine_tune(duration=1)
            print("Please speak")
            answer = speech_to_text.listen_and_predict(online=online, verbose=verbose)
            print(f"Your answer: {answer}")