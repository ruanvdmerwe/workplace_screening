import tensorflow as tf
from ..core.core import ImageAndVideo
from imutils import paths
from imutils import resize
import pickle
import cv2
import os
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn import preprocessing
from joblib import dump

class FaceIdentifyDataCreation(ImageAndVideo):
    """
    This class can be used to add face examples to both the pickle file
    containing all of the embeddings as well as the folder containing the
    physical images. 

    Arguments:
        encodings_location {str}:      
            Path to the pickle containing encodings. If the path
            specified does not exist the path will be created.
        embeding_model_location {str}:
            Path towards the tf lite model that will be used 
            to create the facial embeddings.


    Example of usage:
        face_embedding = FaceIdentifyDataCreation(encodings_location=encodings_location,
                                                  embeding_model_location=embeding_model_location)
        face_embedding.start_video_stream()
        face_embedding.capture_frame_and_add_face(faces_folder=faces_folder, 
                                                  person_name=person_name,
                                                  amount_of_examples=6)
        face_embedding.encode_faces(image_path=faces_folder)
    """


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

        """
        Capture images using the attached camera and save the images in a folder
        specified.

        Arguments:
            faces_folder {str}:      
                Path to the folder containing the various 
                folders containing the face images.
            person_name {str}:
                Name of person being logged to the system.
            amount_of_examples{int, default = 6}:
                Amount of photos to be taken and saved.
        """

        print(f"capture_frame_and_add_face for '{person_name}'")
        path = os.path.sep.join([faces_folder, person_name])
        if not os.path.exists(path):
            priint(f"creating directory: {path}")
            os.makedirs(path)

        for i in range(amount_of_examples):
            print(f"{i} capturing frame")
            frame = self.vs.read()
            frame = resize(frame, width=400)
            self.load_image_from_frame(frame)
            print("detecting faces")
            self.detect_faces(probability=0.7)

            path = os.path.sep.join([faces_folder, person_name, f'{str(i).zfill(5)}.png'])
            print(f"writing image {path}")
            cv2.imwrite(path, frame)
            time.sleep(1.0)

        cv2.destroyAllWindows()
        self.vs.stop()


    def encode_faces(self, image_path, model=True):

        """
        Read all of the images within a specified image path and create embedding for all 
        of the faces present. The embeddings will be linked to the folder name the images are 
        in.

        Arguments:
            image_path {str}:      
                Path to the folder containing the various 
                folders containing the face images.

        """

        print("encoding faces")
        imagePaths = list(paths.list_images(image_path))
        imagePaths = [path_ for path_ in imagePaths if path_.split(os.path.sep)[-2] not in self.encoded_faces['names']]

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            print(f"{i}: {imagePath}")
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]

            # load the input image and convert it from RGB (OpenCV ordering) to dlib ordering (RGB)
            self.load_image_from_file(imagePath)
            self.detect_faces(probability=0.5, face_size=(160,160))
            bounding_boxes = [(y,w,x,h) for (x, y, w, h) in self.bounding_boxes]
            
            encodings = []
            for face in self.faces:
                self.embedding_model.set_tensor(self.input_details[0]['index'], face)
                self.embedding_model.invoke()
                predicted_encoding = self.embedding_model.get_tensor(self.output_details[0]['index'])[0]
                encodings.append(predicted_encoding)
            
            for encoding in encodings:
                self.encoded_faces['encodings'].append(encoding)
                self.encoded_faces['names'].append(name)
        
        with open(self.encodings_location, "wb") as f:
            f.write(pickle.dumps(self.encoded_faces))
        
        if model == True:

            print("Training model")
            x = np.array(self.encoded_faces['encodings'])
            y = np.array(self.encoded_faces['names'])

            #create integer labels
            le = preprocessing.LabelEncoder()
            le.fit(y)
            y_encoded = le.transform(y)

            # train voting model
            svm = SVC(C= 10, gamma=0.001, kernel='rbf', probability=True)
            knn = KNeighborsClassifier(n_neighbors=4, weights = 'distance')
            rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                            criterion='gini', max_depth=100, max_features='auto',
                            max_leaf_nodes=None, max_samples=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=2,
                            min_weight_fraction_leaf=0.0, n_estimators=800,
                            n_jobs=None, oob_score=False, random_state=42,
                            verbose=0, warm_start=False)
                            
            clf = VotingClassifier(estimators=[('rf', rf), ('knn', knn), ('svm', svm)],
                         voting='soft',
                         weights = (1,1,2))
            clf.fit(x, y)

            # save encoder and model
            dump(le, 'label_encoder.joblib')
            dump(clf, 'face_recognizer.joblib') 
            print("Done training model")

        
