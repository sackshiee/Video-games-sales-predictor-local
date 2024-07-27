# Video Game Sales Predictor

This project is a web application developed using Flask, which predicts the sales of video games based on various features such as console, year of release, category, publisher, rating, critics points, and user points. The prediction model is built using Ridge Regression with Grid Search for hyperparameter tuning, achieving an accuracy of 89.2%.

## Features
- **Input Features:** Console, Year, Category, Publisher, Rating, Critics Points, User Points
- **Model:** Ridge Regression with hyperparameter tuning using Grid Search
- **Encoding:** Dummy encoding for categorical variables
- **Scaling:** StandardScaler for numerical features
- **Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)

## Model Performance
- **Mean Absolute Error (MAE):** 1.1818
- **Root Mean Squared Error (RMSE):** 2.0339
- **R² Score:** 0.0888

## Usage
1. Clone the repository
2. Install the required dependencies
3. Run the Flask application
4. Access the web interface to input video game features and get sales predictions

## Contributing
Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.


## Installation
```bash
git clone https://github.com/yourusername/video-game-sales-predictor.git
cd video-game-sales-predictor
pip install -r requirements.txt
python app.py
```


!(![Screenshot (10)](https://github.com/user-attachments/assets/ed816ee6-ec45-4d93-bc97-ef5189a8588e)

