# Using this code

To run the GUI that streams video and text to the screen for the user:

    cd workplace_screening_dev
    source env/bin/activate
    python workplace_screening/gui.py

Then, in a separate terminal:

    source .env
    source env/bin/activate
    python workplace_screening/controller.py

But make sure that you've got a trained model locally. If there are images in the `workplace_screning/faces/<name>` directories, then you can [train the model](#training-the-model). 

# Setup

1. Clone github reporisoty and change to the project folder
        
1. If on Raspberry Pi

    1. update & upgrade installed packages

            sudo apt-get update
            sudo apt-get upgrade
        
    1. install required dependencies for openCV, PyAudio and PocktSphinx etc.:
    
            sudo apt-get install cmake swig libpulse-dev libhdf5-dev libhdf5-serial-dev libhdf5-103 libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 libatlas-base-dev libjasper-dev flac -y

    2. install PocketSphinx itself as per [this blog post](https://howchoo.com/g/ztbhyzfknze/how-to-install-pocketsphinx-on-a-raspberry-pi).
        
    3. install _virtualenv_ for setting up the python environment, and add it to your _PATH_:

            pip3 install virtualenv
            source ~/.profile

    1. In the project folder, create a new Python environement:

            cd workplace_screening_dev
            virtualenv env -p python3
            source env/bin/activate

    4. install RPi.GPIO package (only works when running on a raspberry pi)
            
            pip install RPi.GPIO

    5. Now, install the other required packages:

            pip install -r requirements.txt


1. If on a Mac

    1. install the required dependencies for PocketShpinx and pyaudio:

            brew install cmake
            brew install cmu-pocketsphinx
            brew install portaudio
            brew install swig
            brew install openal-soft
        
    1. Add a symlink for OpenAl

            cd /usr/local/include
            ln -s /usr/local/Cellar/openal-soft/1.20.1/include/AL/* .

    1. In the project folder, create a new Python environement:

            cd workplace_screening_dev
            virtualenv env -p python3
            source env/bin/activate

    4. install a mock RPi.GPIO package, so the code doesn't crash when running on a mac
            
            pip install fake_rpi

    5. Now, install the other required packages:

            pip install -r requirements.txt


Now, you have a clean Python 3 environment, with no packages installed. To install the required packages:

    pip install -r requirements.txt

# Running a few tests

## Create a livestream for facemask detection

1. From the project folder:
    
       python test.py -m ./workplace_screening/facemask_detection_model.tflite --l True --voice False --online False 

1. To stop the stream, press `q`

## Training the model

1. Add a new folder containing six photos of the new person to the folder containing all of the previous photo folders. This new folder should be the name of the new person.

1. In this example, the faces are in the path workplace_screening/faces, so add the new face their. Ensure that the encodings pickle file is also correct (the f flag).

1. From the project folder:

        python test.py -n workplace_screening/faces/ -f workplace_screening/encodings.pkl -e workplace_screening/face_embedding_model.tflite --voice False --online False -l False
    
## Create a livestream for face recognition

1. From the project folder:
    
       python test.py -f workplace_screening/encodings.pkl -e workplace_screening/face_embedding_model.tflite --voice False --online False -l True

1. To stop the stream, press `q`
    
## Take a snapshot and predict face mask and recognize face

1. From the project folder:

        python test.py -m ./workplace_screening/facemask_detection_model.tflite -f ./workplace_screening/encodings.pkl -e ./workplace_screening/face_embedding_model.tflite -t 0.35 --probability 0.7 --l False --voice False --online False
   
## Test voice regocnition 

1. This will run five tests to see if the microphone picks up yes/no.

1. From the project folder:
        
        python test.py  --l False --voice True --online {False/True}

If the `online` flag is set to `true`, it will use the _Google Cloud API_, otherwhise it will use _PocketSphinx_.
