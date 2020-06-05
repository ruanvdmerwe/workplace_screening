# Setup

1. Clone github reporisoty and change to the project folder
        
1. If on Raspberry Pi
        
    1. install required dependencies for openCV, PyAudio and PocktSphinx:
    
            sudo apt-get install cmake
            sudo apt-get install swig
            sudo apt-get install libpulse-dev
            sudo apt-get install libhdf5-dev libhdf5-serial-dev libhdf5-103
            sudo apt-get install libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
            sudo apt-get install libatlas-base-dev
            sudo apt-get install libjasper-dev

    2. install pocketsphinx itself as per [https://howchoo.com/g/ztbhyzfknze/how-to-install-pocketsphinx-on-a-raspberry-pi]()
        
    3. install _virtualenv_ for setting up the python environment, and add it to your _PATH_:

            pip3 install virtualenv
            source ~/.profile

1. If on a Mac

    1. install the required dependencies for pocketshpinx and pyaudio:

            brew install cmake
            brew install cmu-pocketsphinx
            brew install portaudio
            brew install swig
            brew install openal-soft
        
    1. Add a symlink for OpenAl

            cd /usr/local/include
            ln -s /usr/local/Cellar/openal-soft/1.20.1/include/AL/* .
        
1. Create a new environement:

        virtualenv env -p python3
        source env/bin/activate

Now, you have a clean Python 3 environment, with no packages installed. To install the required packages:

        pip install -r requirements.txt

# Running a few tests

## Create a livestream

1. From the project folder:
    
       python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l True --voice False --online False 

1. To stop the stream, press `q`
  
## Add a new face to model encodings (start recognizing someone new)

    python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l False -n ./workplace_screening/faces/ -p {persons_name} --voice False --online False
   
## Take a snapshot and predict face mask and recognize face

    python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l False --voice False --online False
   
## Test voice regocnition 
        
    python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l False --voice True --online {False/True}

If the `online` flag is set to `true`, it will use the _Google Cloud API_, otherwhise it will use _PocketSphinx_.
