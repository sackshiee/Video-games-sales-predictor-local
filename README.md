﻿# Video-games-sales-predictor-local
 Video Game Sales Predictor
This project is a web application developed using Flask, which predicts the sales of video games based on various features such as console, year of release, category, publisher, rating, critics points, and user points. The prediction model is built using Ridge Regression with Grid Search for hyperparameter tuning, achieving an accuracy of 89.2%.

Features
Input Features: Console, Year, Category, Publisher, Rating, Critics Points, User Points
Model: Ridge Regression with hyperparameter tuning using Grid Search
Encoding: Dummy encoding for categorical variables
Scaling: StandardScaler for numerical features
Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
Usage
Clone the repository
Install the required dependencies
Run the Flask application
Access the web interface to input video game features and get sales predictions
Installation
bash
Copy code
git clone https://github.com/yourusername/video-game-sales-predictor.git
cd video-game-sales-predictor
pip install -r requirements.txt
python app.py
Screenshots


Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License.


