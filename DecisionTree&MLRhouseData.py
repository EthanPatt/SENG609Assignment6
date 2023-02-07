# Import functions and Packages

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import array
import joblib
import pandas as pd
from sklearn import tree

# Load the data

df = pd.read_csv("House_data.csv")

# Create X and Y arrays
x = df[["sq_feet","num_bedrooms","num_bathrooms"]]
y = df[["sale_price"]]

# Split the data into training and testing for the modeling (25% - Testing)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)

joblib.dump(clf, 'HousePredictor1.pkl')

# Report how well the model is performing
print("Classifier results: ")

mse_train = mean_absolute_error(y_train, clf.predict(x_train))
print(f" - Training Set Error: {mse_train}")

mse_test = mean_absolute_error(y_test, clf.predict(x_test))
print(f" - Testing Set Error: {mse_test}")

# Load our trained model
model = joblib.load('HousePredictor1.pkl')

# Define the house we want to value
house_1 = [2000,
           3,
           2,]

# Save the value as array
homes = [
    house_1]

# Make the prediction
home_values = model.predict(homes)

# Predict the value for the first input of the array
predicted_value = float(home_values[0])

# Print the results
print("House details:")
print(f"- {house_1[0]} sq feet")
print(f"- {house_1[1]} bedrooms")
print(f"- {house_1[2]} bathrooms")
print(f"Estimated value: ${predicted_value:,.2f}")