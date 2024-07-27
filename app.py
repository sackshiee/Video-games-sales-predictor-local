from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv(r'C:\Users\Sakshi.Singh\local_proj\Data\Train.csv')

# Print the column names for debugging
print("Columns in the dataset:", df.columns)

# Assuming 'CONSOLE', 'YEAR', 'CATEGORY', 'PUBLISHER', 'RATING', 'CRITICS_POINTS', 'USER_POINTS' are features and 'SalesInMillions' is the target
X = df[['CONSOLE', 'YEAR', 'CATEGORY', 'PUBLISHER', 'RATING', 'CRITICS_POINTS', 'USER_POINTS']]
y = df['SalesInMillions']

# Encode categorical variables (dummy encoding)
X_encoded = pd.get_dummies(X)

# Scale numerical features
scaler = StandardScaler()
X_encoded[['YEAR', 'CRITICS_POINTS', 'USER_POINTS']] = scaler.fit_transform(X_encoded[['YEAR', 'CRITICS_POINTS', 'USER_POINTS']])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Use Ridge Regression with Grid Search for Hyperparameter Tuning
ridge = Ridge()
parameters = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
grid_search = GridSearchCV(ridge, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Best model after Grid Search
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate the Mean Absolute Error and Root Mean Squared Error
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    console = request.form['CONSOLE']
    year = request.form['YEAR']
    category = request.form['CATEGORY']
    publisher = request.form['PUBLISHER']
    rating = request.form['RATING']
    critics_points = request.form['CRITICS_POINTS']
    user_points = request.form['USER_POINTS']

    # Create a dataframe for the input features and encode it
    input_data = pd.DataFrame({'CONSOLE': [console], 'YEAR': [year], 'CATEGORY': [category], 'PUBLISHER': [publisher], 'RATING': [rating], 'CRITICS_POINTS': [critics_points], 'USER_POINTS': [user_points]})
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Scale numerical features in input data
    input_encoded[['YEAR', 'CRITICS_POINTS', 'USER_POINTS']] = scaler.transform(input_encoded[['YEAR', 'CRITICS_POINTS', 'USER_POINTS']])

    # Make a prediction using the trained model
    prediction = best_model.predict(input_encoded)

    return render_template('index.html', 
                           prediction=prediction[0], 
                           CONSOLE=console, 
                           YEAR=year, 
                           CATEGORY=category, 
                           PUBLISHER=publisher,
                           RATING=rating,
                           CRITICS_POINTS=critics_points,
                           USER_POINTS=user_points)

if __name__ == '__main__':
    app.run(debug=True)