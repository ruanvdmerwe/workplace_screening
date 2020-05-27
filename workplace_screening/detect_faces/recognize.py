import tensorflow as tf
from ..core.core import ImageAndVideo
from imutils import resize
import numpy as np
import cv2
import pickle
import os
from scipy.spatial import distance


class FaceIdentifier(ImageAndVideo):

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

        self.embedding_model = tf.lite.Interpreter(model_path=embeding_model_location)
        self.embedding_model.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.embedding_model.get_input_details()
        self.output_details = self.embedding_model.get_output_details()   

    def recognize_faces(self, tolerance):
        
        # get everything in correct format
        boxes = [(y,w,x,h) for (x, y, w, h) in self.bounding_boxes]

        encodings = []
        for i,face in enumerate(self.faces):

            self.embedding_model.set_tensor(self.input_details[0]['index'], face)
            self.embedding_model.invoke()
            predicted_encoding = self.embedding_model.get_tensor(self.output_details[0]['index'])
            #predicted_encoding = self.embedding_model.predict(face)
            encodings.append(predicted_encoding[0])

        self.recognized_faces = []

        for encoding in encodings:
            # check to see if we have found a match
            encoding = encoding.reshape(1,-1)
            similarities = distance.cdist(self.encoded_faces["encodings"], encoding,'cosine')
            similarities = similarities/similarities.max()
            matches = [distance[0] <= tolerance for distance in similarities]

            if True in matches and sum(matches) >= 2:

                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                sims = {}
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.encoded_faces["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                
                name = max(counts, key=counts.get)

                sims = {name:[] for name in np.unique(self.encoded_faces["names"])}

                for i in matchedIdxs:
                    similarity = similarities[i][0]
                    name = self.encoded_faces["names"][i]
                    sims[name].append(similarity)

                average_sim = {name:np.mean(sim)/(counts[name]**2) for name, sim in sims.items() if not np.isnan(np.mean(sim))}
                name = min(average_sim, key=average_sim.get)
            else:
                name = 'Unkown' 

            # update the list of names
            self.recognized_faces.append(name)

        self.colors = [(33, 33, 183) if name == "Unkown" else (0, 102, 0) for name in self.recognized_faces]
        self.labels = ['Unkown Person' if name == "Unkown" else  f'{name} identified' for name in self.recognized_faces]

        return self.labels, self.colors


    def capture_frame_and_recognize_faces(self, tolerance=0.35, face_probability=0.9):

        self.capture_frame_and_load_image()
        self.detect_faces(probability=face_probability, face_size=(160,160))
        self.recognize_faces(tolerance=tolerance)
        self.draw_boxes_around_faces()

        return self.labels, self.colors

    def capture_frame_and_recognize_faces_live(self, tolerance=0.35, face_probability=0.9):

        while True:

            # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
            frame = self.vs.read()
            frame = resize(frame, width=400)

            self.load_image_from_frame(frame)
            self.detect_faces(probability=face_probability, face_size=(160,160))
            self.recognize_faces(tolerance=tolerance)
            self.draw_boxes_around_faces()

            key = cv2.waitKey(1) & 0xFF
            cv2.imshow("Frame", cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
                
        cv2.destroyAllWindows()