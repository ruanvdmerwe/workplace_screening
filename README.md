# Setup

1. Clone github reporisoty and change to the project folder
1. Create a new environement:

        virtualenv env -p python3
        source env/bin/activate
        
1. If on Raspberry Pi, install required dependencies for openCV:

        sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-103
        sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
        sudo apt-get install libatlas-base-dev
        sudo apt-get install libjasper-dev

  
1. If on a mac, install the required dependencies for pocketshpinx and pyaudio:

        brew install cmake
        brew install cmu-pocketsphinx
        brew install portaudio
        brew install swig
        brew install openal-soft
        
1. Add a symlink for OpenAl

        cd /usr/local/include
        ln -s /usr/local/Cellar/openal-soft/1.20.1/include/AL/* .
        
1. In terminal, run pip install -r requirements.txt

# Running a few tests

## Creata a livestream

1. Be in project folder
1. Run following command:
python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l True --voice False --online False 
1. In order to stop the stream, just press q
  
## Add a new face to model encodings (start recognizing someone new)

1. Be in project folder
1. Run following command:
   python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l False -n ./workplace_screening/faces/ -p {persons_name} --voice False --online False
   
## Take a snapshot and predict face mask and recognize face

1. Be in project folder
1. Run following command:
   python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l False --voice False --online False
   
## Test voice regocnition 
1. Be in project folder
1. If the online flag is set to true, it will use the google cloud api, otherwhise it will use pocketsphinx.
1. Run following command:
   python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l False --voice True --online {False/True}
