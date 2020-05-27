import tensorflow as tf
from ..core.core import ImageAndVideo
from imutils import resize
import cv2

class FaceMaskDetector(ImageAndVideo):

    def __init__(self, mask_detect_model):
        """
        Class used to predict the probability of a face
        wearing a face mask. 

        Arguments:
            serialize_face_model {str} -- Path to the folder containing the face serializer model weights and prototxt file
            mask_detect_model {str} -- Path to the model that detects masks file
        """

        super().__init__()
        self.mask_detect_model_location = mask_detect_model

        # load the face mask detector model
        # Load TFLite model and allocate tensors.
        self.mask_model = tf.lite.Interpreter(model_path=mask_detect_model)
        self.mask_model.allocate_tensors()
        # Get input and output tensors.
        self.input_details = self.mask_model.get_input_details()
        self.output_details = self.mask_model.get_output_details()     

    def detect_facemask(self, mask_probability=0.995):
        """
        This function returns the probability of all the faces in an image containing wearing a
        face mask.


        Arguments:
            mask_probability {float} -- Minimum probability required to say mask is identified

        Returns:
            labels {list(str)} -- A string indicating the predicted label as well as the probability. For 
                                  cases where a mask is predicted label will be "Wearing Mask: .2f", where
                                  .2f represents the probality rounded to two decimal points. The other case 
                                  will be "No Mask: .2f".
        """

        self.labels = []
        self.colors = []
        
        for face in self.faces:
            
            self.mask_model.set_tensor(self.input_details[0]['index'], face)
            self.mask_model.invoke()
            mask_prob = self.mask_model.get_tensor(self.output_details[0]['index'])

            label = "Wearing Mask" if (1-mask_prob[0][0]) >= mask_probability else "No Mask"

            color = (0, 102, 0) if "Wearing Mask" in label  else (33, 33, 183)

            self.labels.append(label)
            self.colors.append(color)
            
        return self.labels, self.colors
    
    def capture_frame_and_detect_facemask(self, mask_probability=0.995):

        self.capture_frame_and_load_image()
        self.detect_faces(probability=0.5)
        self.detect_facemask(mask_probability=mask_probability)
        self.draw_boxes_around_faces()
        
        for label in self.labels:
            if "Wearing Mask" in label:
                return True

        return False


    def capture_frame_and_detect_facemask_live(self, mask_probability=0.975, face_probability=0.9):

        while True:

            # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
            frame = self.vs.read()
            frame = resize(frame, width=400)

            self.load_image_from_frame(frame)
            self.detect_faces(probability=face_probability)
            self.detect_facemask(mask_probability=mask_probability)
            self.draw_boxes_around_faces()
            
            # for label, box, color in zip(labels, bounding_boxes, colors):
            #     cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            #     # show the output frame
            #     cv2.imshow("Frame", frame)

            key = cv2.waitKey(1) & 0xFF
            cv2.imshow("Frame", cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
                
        cv2.destroyAllWindows()
            
            
    def clean_up(self):
        self.vs.stop()