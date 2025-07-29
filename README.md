# Car Price Prediction

A web application built with Streamlit to predict the selling price of used cars using a machine learning model.

![Car Price Prediction App Demo](screenshot/1)

---

## 📋 Table of Contents
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Model Details](#-model-details)
- [How to Run Locally](#-how-to-run-locally)
- [About Me](#-about-me)
- [Project Structure](#-project-structure)

---

## ✨ Features
- **Interactive User Interface:** Allows users to input car details like manufacturing year, kilometers driven, fuel type, seller type, transmission, and owner history.
- **Real-time Price Prediction:** Instantly calculates and displays the estimated selling price based on the provided features.
- **Data-Driven Insights:** The model is trained on a comprehensive dataset of over 4,000 used car listings.

---

## 🛠️ Technology Stack
- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-learn
- **Data Manipulation:** Pandas, NumPy

---

## 📊 Dataset
The model is trained on the `Cars_Dataset.csv` file, which contains data on 4,340 used car sales from India. The dataset includes the following columns:
- `name`
- `year`
- `selling_price` (Target Variable)
- `km_driven`
- `fuel`
- `seller_type`
- `transmission`
- `owner`

---

## 🤖 Model Details
The prediction is powered by a **Linear Regression** model. The complete data preprocessing, model training, and evaluation process is documented in the Jupyter Notebook: `Car_Price_Prediction.ipynb`.

The model uses the following features for prediction:
- `year`
- `km_driven`
- `fuel`
- `seller_type`
- `transmission`
- `owner`

---

## 🚀 How to Run Locally

To get this application running on your local machine, please follow these steps.

### Prerequisites
- Python 3.7+
- Pip package manager

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/car-price-prediction-streamlit.git](https://github.com/YOUR_USERNAME/car-price-prediction-streamlit.git)
cd car-price-prediction-streamlit

2. Install Dependencies
Install the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

3. Run the Streamlit App
Start the Streamlit server.

streamlit run app.py

The application should now be open and running in your web browser!

👨‍💻 About Me
Hello! I'm Harsh Bandal, a passionate developer and data enthusiast. I enjoy building projects that solve real-world problems.

GitHub: https://github.com/harry16102003

LinkedIn: https://www.linkedin.com/in/harsh-bandal-3240912b7

Portfolio: https://hbandal-portfolio.netlify.app/

📁 Project Structure
.
├── 📄 app.py                  # Main Streamlit application script
├── 📄 Cars_Dataset.csv        # The dataset used for training
├── 📄 Car_Price_Prediction.ipynb # Jupyter Notebook with model development
├── 📄 requirements.txt        # Python dependencies
└── 📄 README.md               # This file

