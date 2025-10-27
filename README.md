## **Tweet foul detection**

This is a university project for an assignment which provide a binary classifier for tweets, where 1 is foul/offensive and 0 is proper.
It uses a fastapi and supports 3 models with different thresholds: low, medium and high.





## **Folder structure**


tweet\_foul\_detection\\
app.py				#FastAPI API

 	train.py			#Training script

 	requirements.txt		#Dependencies

 	Dockerfile			#Docker build instructions

 	Readme.md			#Instructions \& info

tweet\_foul\_detection\\tests\\

 	test\_app.py			#Pytest test cases

tweet\_foul\_detection\\data\\

 	labeled\_data.csv			#Dataset

tweet\_foul\_detection\\model\_artifacts\\

 	foul\_detector.joblib

 	foul\_detector\_high.pkl

 	foul\_detector\_medium.pkl

 	foul\_detector\_low.pkl



## Running locally using python:



1-Install Dependencies using the following code in cmd:

pip install -r requirements.txt



2-Start FastAPI server:

uvicorn app:app --host 0.0.0.0 --port 8000



3-Open the FastAPI interface using the following URL:

http://localhost:8000/docs



4-Test the app by:

-Press on POST /predict

-press "try it out" on the top right side

-You should now be able to edit the text and levels.

-input the tweet into the text area

-input the model level: low, medium or high

-you should now be able to see the output on the bottom



example: {

  "label": 0,      #1 for foul/offensive, 0 for proper

  "probability": 0, #Confidance of the model

  "threshold": 0,   #Threshold of the model used

  "level": "string" #Text used

}



## Running on cmd

1-Install Dependencies using the following code in cmd:

pip install -r requirements.txt



2-Start FastAPI server:

uvicorn app:app --host 0.0.0.0 --port 8000



3-use the following code to test



curl -X POST "http://localhost:8000/predict" \\

-H "Content-Type: application/json" \\

-d "{\\"level\\":\\"(insert level here)\\", \\"text\\":\\"(insert text here)\\"}" #You can replace level and text example

 

## Testing locally



To verify setup and insure all is working correctly run the following code:



pytest tests/test\_app.py -v



You should see: 3 passed in X.XXs



## Running with docker on your own machine

1-Build the Docker image using:

docker build -t foul-detector .



2-Run the container

docker run -d -p 8000:8000 --name foul-detector-app foul-detector



3-open the url

http://localhost:8000/docs



4-4-Test the app by:

-Press on POST /predict

-press "try it out" on the top right side

-You should now be able to edit the text and levels.

-input the tweet into the text area

-input the model level: low, medium or high

-you should now be able to see the output on the bottom



\#example: {

  "label": 0,      #1 for foul/offensive, 0 for proper

  "probability": 0, #Confidance of the model

  "threshold": 0,   #Threshold of the model used

  "level": "string" #Text used

}



## Testing on docker



Just run the following code:

docker run --rm tweet-foul-detection pytest





## Running the app on the cloud:



The app is also available for use using the following URL:

https://tweet-foul-detection.onrender.com/docs



-Test the app simply by opening the url of any browser

-Press on POST /predict

-press "try it out" on the top right side

-You should now be able to edit the text and levels.

-input the tweet into the text area

-input the model level: low, medium or high

-you should now be able to see the output on the bottom



\#example: {

  "label": 0,      #1 for foul/offensive, 0 for proper

  "probability": 0, #Confidance of the model

  "threshold": 0,   #Threshold of the model used

  "level": "string" #Text used

}

## 

