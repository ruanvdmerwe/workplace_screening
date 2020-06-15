from workplace_screening import FaceMaskDetector, FaceIdentifier, FaceIdentifyDataCreation, SpeechToText
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run and test the facial recogntion/mask detection models and functions')

parser.add_argument('-f',
                    '--faces',
                    #required=True,
                    help='location of the face encodings pkl file')
parser.add_argument('-e',
                    '--embedding',
                    #required=True,
                    help='location of the face embeddings model')
parser.add_argument('-v',
                    '--verbose',
                    nargs='?',
                    const = False,
                    help='If true, then print more information')
parser.add_argument('-n',
                    '--known',
                    nargs='?',
                    required=False,
                    help='Folder where faces are stored')

args = vars(parser.parse_args())
encodings_location = args['faces']
embeding_model_location = args['embedding']
faces_folder = args['known']
verbose = args['verbose']

if verbose == 'True':
    verbose = True
else:
    verbose = False


if __name__ == '__main__':
    # ----------------- Adding a new face to faces dataset -----------------
    face_embedding = FaceIdentifyDataCreation(encodings_location=encodings_location,
                                            embeding_model_location=embeding_model_location)
    print("Embedding new face features")
    face_embedding.encode_faces(image_path=faces_folder, model=True)   