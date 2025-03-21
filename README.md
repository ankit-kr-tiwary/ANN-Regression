
# Customer Salary Prediction Models

This repository contains machine learning models for predicting customer churn and employee salaries, deployed as interactive Streamlit applications.

## Project Overview

The project consists of main components:
 Salary Prediction (Regression)

The model is deployed using Streamlit for easy interaction and visualization.

Access Webapp : [https://ann-regression-de37sfyqxgza6zgt3itnfw.streamlit.app/](https://ann-regression-de37sfyqxgza6zgt3itnfw.streamlit.app/)

## Customer Churn Prediction

The customer churn prediction model uses classification techniques to determine whether a customer is likely to churn or not.

### Features

- Utilizes an Artificial Neural Network (ANN) for classification
- Predicts the likelihood of customer churn based on various input features
- Achieves an accuracy of 89.47% on the test set


### Usage

Input customer information such as tenure, contract type, and monthly charges to receive a churn prediction.

## Salary Prediction

The salary prediction model uses regression techniques to estimate an employee's salary based on years of experience.

### Features

- Implements Linear Regression for salary prediction
- Visualizes the relationship between years of experience and salary
- Provides R² scores and Mean Squared Error (MSE) for model evaluation


### Usage

Enter the years of experience to get a predicted salary estimate.

## Deployment

Both models are deployed using Streamlit, allowing for easy interaction through a web interface.

### Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Project Structure

```
├── data/
│   ├── churn_data.csv
│   └── salary_data.csv
├── models/
│   ├── churn_model.pkl
│   └── salary_model.pkl
├── app.py
├── churn_prediction.py
├── salary_prediction.py
├── requirements.txt
└── README.md
```


## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn


## Future Improvements

- Implement more advanced models for regression tasks
- Add feature importance analysis for the churn prediction model
- Incorporate more visualizations and explanatory elements in the Streamlit app

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
