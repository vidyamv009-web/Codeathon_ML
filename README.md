# Car Price Prediction

Mini project to predict car prices in the US market using machine learning.

## What this project does
- Loads car dataset
- Cleans and prepares the data
- Trains 5 regression models
- Finds the best model
- Checks feature importance
- Does hyperparameter tuning
- Saves the model using joblib
- Simple Flask UI to predict price

## Models used
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regressor

Best model: Random Forest (after tuning)

## Files
- `CarPricePrediction.ipynb` - main notebook
- `CarPrice_Assignment.csv` - dataset
- `app.py` - flask app
- `templates/index.html` - simple UI page
- `model/car_price_artifacts.joblib` - saved model
- `requirements.txt` - required packages

## How to run notebook
1. Open `CarPricePrediction.ipynb`
2. Run all cells
3. Last cells will save the joblib model

## How to run Flask app
```
pip install -r requirements.txt
python app.py
```
Then open: http://127.0.0.1:5000

## Note
Make sure you run the notebook joblib cell before starting the Flask app.
