from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from imutils.video import VideoStream
from imutils import resize
from PIL import Image
import numpy as np
import time
import cv2

class ImageAndVideo(object):

    def __init__(self):

       #self.serialize_model = cv2.dnn.readNet(prototxtPath, weightsPath)
       self.serialize_model = MTCNN(min_face_size = 60)

    
    def load_image_from_file(self, image_location):
        """        
        Function that takes in an image and conerts it to the correct
        format to be used by the models. 

        Arguments:
            image {str} -- Image location on system
        """

        # load the input image from disk, clone it, and grab the image spatial dimensions
        self.image = cv2.imread(image_location)
        #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        (self.h, self.w) = self.image.shape[:2]
        
        if self.h >= 1000 or self.w >=1000:
            self.w = int(self.w * 0.3)
            self.h = int(self.h * 0.3)
            self.image = cv2.resize(self.image, (self.w, self.h))

        self.frame_picture = False
    
    def load_image_from_frame(self, frame):
        """        
        Function that takes in an image and conerts it to the correct
        format to be used by the models. 

        Arguments:
            image {str} -- Image location on system
        """

        # load the input image from disk, clone it, and grab the image spatial dimensions
        self.image = frame
        # ensure correct color format as models were trained on
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        (self.h, self.w) = self.image.shape[:2]
        self.frame_picture = True
        
    def detect_faces(self, probability=0.5, face_size = (224,224)):

        """
        This function will scan through the loaded image and return all the faces detected.

        Arguments:
            probability {float} -- Minimum probability of prediction required to return a face

        Returns:
            face {array} -- array of values representing the face
            bounding_boxes {list} -- list of values corrospoding to the box around the face. Format is: [startX, startY, endX, endY]
        """

        if self.frame_picture:
            detections = self.serialize_model.detect_faces(self.image)
        else:
            detections = self.serialize_model.detect_faces(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))

        self.faces = []
        self.bounding_boxes = []
        
        # loop over the detections
        for detection in detections:
            confidence = detection['confidence']
            if confidence >= probability:
                (startX, startY, width, height) = detection['box']
                (endX, endY) = (startX+width, startY+height)

                # ensure the bounding boxes fall within the dimensions of the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(self.w - 1, endX), min(self.h - 1, endY))
                
                self.bounding_boxes.append([startX, startY, endX, endY])

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = self.image[startY:endY, startX:endX]
                try:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, face_size)
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)
                    face = face/255
                    self.faces.append(face)
                except:
                    pass
        
        return self.faces, self.bounding_boxes

    
    def draw_boxes_around_faces(self, labels=[], colors = []):

        """
        Function that alters the original image used in predictions to draw a 
        box around the face and indicate if the person is wearing a face mask or not.

        Returns:
            array -- returns the image as an array.
        """

        if len(labels) < 1 or len(colors) < 1:
            labels = self.labels
            colors = self.colors

        for bounding_box, label, color in zip(self.bounding_boxes, labels, colors):
            cv2.putText(self.image,
                        str(label),
                        (bounding_box[0], bounding_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            cv2.rectangle(self.image,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[2], bounding_box[3]),
                          color, 2)

        return self.image

    def display_predictions(self):

        """
        Function to display the given prediction. Will only work
        if all the previous steps have been done.
        """

        if self.frame_picture:
           img = Image.fromarray(self.image)
        else:
            img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        
        img.show()


    def start_video_stream(self):

        self.vs = VideoStream(src=0).start()
        # allowing camera to warm up
        time.sleep(2.0)


    def capture_frame_and_load_image(self, stop = True):

        # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
        frame = self.vs.read()
        frame = resize(frame, width=400)

        if stop:
            cv2.destroyAllWindows()
            self.vs.stop()

        self.load_image_from_frame(frame)

        return frame





    


    

    
    