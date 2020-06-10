from .detect_facemask.detect_facemask import FaceMaskDetector
from .detect_faces.recognize import FaceIdentifier
from imutils import resize
from imutils.video import VideoStream

FACE_MASK_DETECTOR = FaceMaskDetector(mask_detect_model='./workplace_screening/facemask_detection_model.tflite')
FACE_RECOGNIZER = FaceIdentifier(encodings_location='./workplace_screening/encodings.pkl',
                                 embeding_model_location='./workplace_screening/face_embedding_model.tflite')


def get_predictions(vs):

    frame = resize(vs.read(),400)

    FACE_RECOGNIZER.load_image_from_frame(frame)
    number_of_faces = FACE_RECOGNIZER.detect_faces(probability=0.8, face_size=(160,160))
    recognized_names = FACE_RECOGNIZER.recognize_faces(tolerance=0.35, verbose=False)

    FACE_MASK_DETECTOR.load_image_from_frame(frame)
    FACE_MASK_DETECTOR.detect_faces(probability=0.8, face_size=(224,224))
    mask_detected = FACE_MASK_DETECTOR.detect_facemask()

    return recognized_names, mask_detected, number_of_faces


vs = VideoStream(src=0).start()
while True:
    recognized_names, mask_detected, number_of_faces = get_predictions(vs)
    if number_of_faces >= 1:
        if mask_detected:
            if recognized_names[0] != 'Unkown':
                text = f"Thanks for wearing your mask, {str(recognized_names[0]).capitalize()}. Going to take your temperature now."
            else:
                text = "Thanks for wearing your mask. Going to take your temperature now."
        else:
            if recognized_names[0] != 'Unkown':
                text = f"You are not allowed in without a mask {str(recognized_names[0]).capitalize()}. Please wear your mask"
            else:
                text = "You are not allowed in without a mask. Please wear your mask."
    else:
        text = "STOP! We need to check your mask, temperature and symptoms before you enter."

    file='predictions.txt' 
    with open(file, 'w') as filetowrite:
        filetowrite.write(text)