# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset from the CSV file provided by the user
@st.cache_data
def load_data():
    """
    This function loads the car dataset from a CSV file into a pandas DataFrame.
    It uses Streamlit's caching to avoid reloading the data on every interaction.
    """
    try:
        car_dataset = pd.read_csv('Cars_Dataset.csv')
        return car_dataset
    except FileNotFoundError:
        st.error("The 'Cars_Dataset.csv' file was not found. Please make sure it's in the same directory as the app.")
        return None

# Preprocess the data and train the model
def train_model(car_dataset):
    """
    This function preprocesses the car dataset and trains a Linear Regression model.
    It handles categorical feature encoding and splits the data into training and testing sets.
    """
    # Encoding categorical columns to numerical values
    car_dataset.replace({'fuel': {'CNG': 0, 'Diesel': 1, 'Electric': 2, 'LPG': 3, 'Petrol': 4}}, inplace=True)
    car_dataset.replace({'seller_type': {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}}, inplace=True)
    car_dataset.replace({'transmission': {'Automatic': 0, 'Manual': 1}}, inplace=True)
    car_dataset.replace({'owner': {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}}, inplace=True)

    # Splitting the data into features (X) and target (Y)
    X = car_dataset.drop(columns=['selling_price', 'name'], axis=1)
    Y = car_dataset['selling_price']

    # Splitting the data for training and testing
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Initializing and training the Linear Regression model
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, Y_train)
    
    return lin_reg_model

# Main app layout and functionality
def main():
    """
    This is the main function that runs the Streamlit application.
    It sets up the user interface for car price prediction.
    """
    st.title("🚗 Car Price Prediction")
    st.markdown("Enter the details of the car to get a price prediction.")

    # Load the data
    car_data = load_data()

    if car_data is not None:
        # Train the model
        model = train_model(car_data.copy())

        # Create input fields for user to enter car details
        st.sidebar.header("Car Features")
        year = st.sidebar.slider("Year", 1990, 2024, 2015)
        km_driven = st.sidebar.slider("Kilometers Driven", 1, 500000, 50000)
        
        fuel_options = {'CNG': 0, 'Diesel': 1, 'Electric': 2, 'LPG': 3, 'Petrol': 4}
        fuel = st.sidebar.selectbox("Fuel Type", list(fuel_options.keys()))
        
        seller_type_options = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}
        seller_type = st.sidebar.selectbox("Seller Type", list(seller_type_options.keys()))

        transmission_options = {'Automatic': 0, 'Manual': 1}
        transmission = st.sidebar.selectbox("Transmission", list(transmission_options.keys()))
        
        owner_options = {'First Owner': 0, 'Second Owner': 1, 'Third Owner': 2, 'Fourth & Above Owner': 3, 'Test Drive Car': 4}
        owner = st.sidebar.selectbox("Owner", list(owner_options.keys()))

        # When the user clicks the predict button
        if st.sidebar.button("Predict Price"):
            # Create a numpy array from the user inputs
            input_data = np.array([[
                year, 
                km_driven, 
                fuel_options[fuel], 
                seller_type_options[seller_type], 
                transmission_options[transmission], 
                owner_options[owner]
            ]])
            
            # Make a prediction
            prediction = model.predict(input_data)
            
            # Display the prediction
            st.success(f"The predicted selling price of the car is: ₹{prediction[0]:,.2f}")

if __name__ == '__main__':
    main()
