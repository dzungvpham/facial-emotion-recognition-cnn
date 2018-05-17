----------------- Set up virtual environment ---------------------
1. Make sure you have python 3.x. This readme assumes that python 3 can be invoked with 'python'.

2. Open bash/terminal/command prompt

3. Clone github repo to your machine:
git clone https://github.com/vietdzung/facial-emotion-recognition-cnn

4. Turn the github repo a virtual environment:
python -m venv --system-site-packages <dir_name>

* NOTE: You can use virtualenv as well. Simply replace 'python -m venv' with 'virtualenv'

5. Cd into the virtual environment and activate it:
- Windows:
 + cd Scripts
 + Type 'activate'
- Unix/Mac:
 + Type 'source bin/activate'

----------------- Install requirements -------------------
* NOTE: Either use pip or pip3. The below instruction uses pip.
If for some reasons, pip/pip3 has to be upgraded, then you can do the following:
python -m pip install --upgrade pip

6. Install opencv-python:
python -m pip install --upgrade opencv-python

7. Install tensorflow:
python -m pip install --upgrade tensorflow

8. Install tflearn:
python -m pip install --upgrade tflearn

------------------ Train model (Optional) -----------------
9. First, convert dataset from .csv to .npy. The resulting file must be in a folder named 'data':
python csv2npy.py

* NOTE: You need to create a Kaggle account and request dataset from:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

10. Configure the model network in emotion_model.py as needed.
- To start training a new model:
python emotion_model.py

- To start training a model from a previous one:
python emotion_model.py <model_path>

------------------ Run app ---------------------
11. If you don't currently have my model or don't want to train one yourself, you can download mine.
https://drive.google.com/open?id=1Y_BnRydcYhWiGbEaw74aRXswbn9qAKCP
- Create a folder in the virtual environment and name it 'models'
- Download and unzip the 3 model files into the above directory

12. Run emotion recognition app:
- To predict in real-time (webcam required):
python emotion_app.py
- To predict a single image:
python emotion_app.py <path_to_img>
- To predict all images in a directory
python emotion_app.py <dir_path>
- To predict a video (NOT WORKING AS EXPECTED):
python emotion_app.py <path_to_video>

* NOTE: predicted images will be saved as <original_file_name>-predict.<file_extension>
Accepted image extensions are: png, jpg, jpeg, pbm, pgm, ppm, bmp, tiff, tif
(you can extend this if you want to)