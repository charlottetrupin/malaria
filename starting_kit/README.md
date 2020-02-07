The problem we try to solve here is the diagnosis of malaria using images of cells that are whether infected or not. It is a binary classification problem from the data set found here at <a href=https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria>"Cell images for detecting malaria"</a>. The data set contains around 27 thousand images, that are balanced between the 2 classes. The way we proceed here is to extract features from the images instead of using images directly as inputs to the model we use for predictions.


Prerequisites:
Install Anaconda Python 3.6.6 

Usage:

(1) If you are a challenge participant:

- The three files sample_*_submission.zip are sample submissions ready to go!

- The file README.ipynb contains step-by-step instructions on how to create a sample submission for the Iris challenge. 
At the prompt type:
jupyter-notebook README.ipynb

- modify sample_code_submission to provide a better model

- zip the contents of sample_code_submission (without the directory, but with metadata), or

- download the public_data and run (double check you are running the correct version of python):

  `python ingestion_program/ingestion.py public_data sample_result_submission ingestion_program sample_code_submission`

then zip the contents of sample_result_submission (without the directory).

(2) If you are a challenge organizer and use this starting kit as a template, ensure that:

- you modify README.ipynb to provide a good introduction to the problem and good data visualization

- sample_data is a small data subset carved out the challenge TRAINING data, for practice purposes only (do not compromise real validation or test data)

- the following programs run properly:

    `python ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`

    `python scoring_program/score.py sample_data sample_result_submission scoring_output`

- the metric identified by metric.txt in the utilities directory is the metric used both to compute performances in README.ipynb and for the challenge.

- your code also runs within the Codalab docker (inside the docker, python 3.6 is called python3):

	`docker run -it -v `pwd`:/home/aux codalab/codalab-legacy:py3`
	
	`DockerPrompt# cd /home/aux`
	`DockerPrompt# python3 ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`
	`DockerPrompt# python3 scoring_program/score.py sample_data sample_result_submission scoring_output`
	`DockerPrompt# exit`
