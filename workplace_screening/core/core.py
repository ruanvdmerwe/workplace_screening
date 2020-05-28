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
    """
    This class can be used to do all the core functionality required
    in facial computer vision. For instance, some functionality in 
    this class can be used to detect faces in images or to start a video 
    stream.
    """
   
    def __init__(self):
       self.serialize_model = MTCNN(min_face_size = 60)

    
    def load_image_from_file(self, image_location):
        """        
        Function that takes in an image and converts it to the correct
        format to be used by the models. 

        Arguments:
            image {str}:
                Image location on system
        """

        # load the input image from disk, clone it, and grab the image spatial dimensions
        self.image = cv2.imread(image_location)
        (self.h, self.w) = self.image.shape[:2]
        
        if self.h >= 1000 or self.w >=1000:
            self.w = int(self.w * 0.3)
            self.h = int(self.h * 0.3)
            self.image = cv2.resize(self.image, (self.w, self.h))

        self.frame_picture = False

    
    def load_image_from_frame(self, frame):
        """        
        Function that takes in an image from a video stream
        and conerts it to the correct format to be used by the
        models. 

        Arguments:
            frame {opencv2 frame}:
                Frame captured by opencv2
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
            probability {float, default=0.5}:
                Minimum probability of prediction required to return a face
            face_size {tuple, default = (224, 224)}:
                Width and heigth that face image should be converted to.
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

    
    def draw_boxes_around_faces(self, labels=[], colors = []):

        """
        Function that alters the original image used in predictions to draw a 
        box around the face and indicate if the person is wearing a face mask
        or not. If wished, one can provide labels and colors, but the class can 
        create its own.

        Arguments:
        labels {list(str), default=[]}:
            List of labels to put on top of boxes drawn around faces.
        colors {list((r,g,b)), default = []}:
            List of RGB colors that boxes should be.
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
        """
        Start video stream.
        """

        self.vs = VideoStream(src=0).start()
        # allowing camera to warm up
        time.sleep(2.0)


    def capture_frame_and_load_image(self, stop = True):

        """
        Captures the current frame of the video stream and load it
        into the class.

        Arguments:
            stop {boolean, default = True}:
                If True, will stop the video stream otherwise the video stream will
                continue.
        """

        # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
        frame = self.vs.read()
        frame = resize(frame, width=400)

        if stop:
            cv2.destroyAllWindows()
            self.vs.stop()

        self.load_image_from_frame(frame)


    def get_video_steam(self):

        """
        Return the video stream object created.

        Returns:
            video stream class.
        """

        return self.vs


    def get_faces(self):
        """
        Return the current identified faces as 
        well as the bounding boxes.

        Returns:
            faces{list}:
                list of image values for the various faces detected.
            boundix_boxes{list}: 
                list of cooardinates that faces are found in original image.
        """

        return self.faces, self.bounding_boxes


    def get_image(self):
        """
        Return the image as it currently is in the process.

        Returns:
            image{cv2 instance of image}:

        """

        return self.image





    


    

    
    