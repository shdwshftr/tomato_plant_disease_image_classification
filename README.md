# Tomato Plant Disease Image Classification
This project aims to predict the type of tomato plant disease based on an image. The goal is to build a model to identifies which type of disease is the tomato plants having, 
to provide information for further actions that will be taken to produce a better quality of the tomatoes.

## Project Structure
```bash
├── codes
│   ├── Tomato_Disease_Classification.ipynb  # ipynb notebook (model building)
│   ├── tomato_deployment.py                 # python code file for model deployment
│   ├── index.html                           # html file for website page
│   ├── style.css                            # css file for website design
├── model
│   ├── tomato.h5                            # Trained CNN model
├── README.md                                # Project documentation
└── requirements.txt                         # Required Python packages
```

## Dataset
The dataset used in this project is the "Plant Village" taken from [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village). 
The dataset contains images of 3 different plants which are pepper bell, potato and tomato. The image data used in this project is only the tomato plant images (10 different classes).

## Project Workflow
1. Data Collection
2. Data Preprocessing
3. Model Building
4. Model Evaluation
5. Model Saving
6. Model Deployment

## Usage
* The process of data collection until model saving can be seen inside the jupyter notebook
* To run the deployed model in a web app, run the tomato_deployment.py file in the terminal with the requirements.txt installed than click on the ip address given.
```bash
Tomato_Disease_Classification.ipynb
tomato_deployment.py
```

## Results
* The CNN model achieved an accuracy of 86.38% in predicting the tomato plant disease correctly.
* The web app can run smoothly and give the predicted outcome displayed along with the image.

## Conclusion
Conclusion are written inside the jupyter notebook (last part).
