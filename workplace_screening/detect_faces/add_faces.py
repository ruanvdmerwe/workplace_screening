import tensorflow as tf
from ..core.core import ImageAndVideo
from imutils import paths
from imutils import resize
import pickle
import cv2
import os
import numpy as np


class FaceIdentifyDataCreation(ImageAndVideo):

    def __init__(self, encodings_location, embeding_model_location):

        super().__init__()

        self.encodings_location = encodings_location
        # if no encodings exist, add this file
        if os.path.isfile(encodings_location):
            self.encoded_faces = pickle.loads(open(encodings_location, "rb").read()) 
        else:
            data = {"encodings": [], "names": []}
            with open(encodings_location, "wb") as f:
                f.write(pickle.dumps(data))
            self.encoded_faces = pickle.loads(open(encodings_location, "rb").read()) 

        # load emebedding model
        self.embedding_model = tf.lite.Interpreter(model_path=embeding_model_location)
        self.embedding_model.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.embedding_model.get_input_details()
        self.output_details = self.embedding_model.get_output_details()    

    def capture_frame_and_add_face(self, faces_folder, person_name, amount_of_examples=6):

        # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
        print("Begining to take pictures!")

        path = os.path.sep.join([faces_folder, person_name])
        if not os.path.exists(path):
            os.makedirs(path)

        for i in range(amount_of_examples):

            frame = self.vs.read()
            frame = resize(frame, width=400)
            self.load_image_from_frame(frame)
            (self.faces, self.bounding_boxes) = self.detect_faces(probability=0.5)

            path = os.path.sep.join([faces_folder, person_name, f'{str(i).zfill(5)}.png'])
            
            cv2.imwrite(path, frame)
            time.sleep(1.0)

        cv2.destroyAllWindows()
        self.vs.stop()

    def encode_faces(self, image_path, detection_method='hog'):

        imagePaths = list(paths.list_images(image_path))
        imagePaths = [path_ for path_ in imagePaths if path_.split(os.path.sep)[-2] not in self.encoded_faces['names']]

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]

            # load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
            self.load_image_from_file(imagePath)
            (faces, bounding_boxes) = self.detect_faces(probability=0.5, face_size=(160,160))
            bounding_boxes = [(y,w,x,h) for (x, y, w, h) in self.bounding_boxes]
            
            encodings = []
            for face in faces:
                self.embedding_model.set_tensor(self.input_details[0]['index'], face)
                self.embedding_model.invoke()
                predicted_encoding = self.embedding_model.get_tensor(self.output_details[0]['index'])[0]
                encodings.append(predicted_encoding)
            
            for encoding in encodings:
                self.encoded_faces['encodings'].append(encoding)
                self.encoded_faces['names'].append(name)
        
        with open(self.encodings_location, "wb") as f:
            f.write(pickle.dumps(self.encoded_faces))
