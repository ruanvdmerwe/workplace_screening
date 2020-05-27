from workplace_screening.detect_facemask.detect_facemask import FaceMaskDetector
from workplace_screening.detect_faces.recognize import FaceIdentifier
from workplace_screening.detect_faces.add_faces import FaceIdentifyDataCreation
import argparse

parser = argparse.ArgumentParser(description='Run and test the facial recogntion/mask detection models and functions')

parser.add_argument('-m',
                    '--mask',
                    required=True,
                    help='location of mask detection model')
parser.add_argument('-f',
                    '--faces',
                    required=True,
                    help='location of the face encodings pkl file')
parser.add_argument('-e',
                    '--embedding',
                    required=True,
                    help='location of the face embeddings model')
parser.add_argument('-t',
                    '--tolerance',
                    nargs='?',
                    const = 0.35,
                    help='tolerance to match a face to known encodings')
parser.add_argument('--probability',
                    nargs='?',
                    const = 0.7,
                    help='minimum probability of face to detect a face')
parser.add_argument('-n',
                    '--known',
                    help='location of the known face images folders')
parser.add_argument('-p',
                    '--person',
                    help='name of the new person to add to face encodings')



args = vars(parser.parse_args())
mask_detect_model = args['mask']
encodings_location = args['faces']
embeding_model_location = args['embedding']
faces_folder = args['known']
person_name = args['person']
tolerance = float(args['tolerance'])
face_probability = float(args['probability'])

if __name__ == '__main__':

    # intiliaze all of the models
    face_mask_detector = FaceMaskDetector(mask_detect_model=mask_detect_model)
    face_recognizer = FaceIdentifier(encodings_location=encodings_location,
                                    embeding_model_location=embeding_model_location)
    face_embedding = FaceIdentifyDataCreation(encodings_location=encodings_location,
                                              embeding_model_location=embeding_model_location)

    # ----------------- detecting if you are wearing a face mask -----------------
    face_mask_detector.start_video_stream()
    face_mask_detector.capture_frame_and_detect_facemask()
    face_mask_detector.display_predictions()

    # # ----------------- Adding a new face to faces dataset -----------------
    if hasattr(args, 'known'):
        if hasattr(args, 'person'):
            face_embedding.start_video_stream()
            face_embedding.capture_frame_and_add_face(faces_folder=faces_folder, person_name=person_name, amount_of_examples=6)
        face_embedding.encode_faces(image_path=faces_folder)

    # ----------------- See if face is recognized -----------------
    face_recognizer.start_video_stream()
    face_recognizer.capture_frame_and_recognize_faces(tolerance=tolerance, face_probability=face_probability)
    face_recognizer.display_predictions()
