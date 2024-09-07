import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data  # Features
y = diabetes.target  # Target variable (progression of diabetes)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Initialize session state for adjusted coefficients if not already done
if 'adjusted_coefs' not in st.session_state:
    st.session_state.adjusted_coefs = list(model.coef_)

st.image('images/nartec.png', width=200)
st.write("# What if scenarios SUPERAPP")
st.write('Experiment with various "what-if" scenarios to see how different factors influence the outcomes and gain valuable understanding of the underlying patterns in diabetes management')

# Tabs
option = st.radio("Select a tab:", ["Model Visualization", "Model Parameters", "Feature Importance"])

if option == "Model Visualization":
    st.sidebar.header("Adjust Model Parameters")

    # Display sliders to adjust coefficients
    feature_names = diabetes.feature_names
    adjusted_coefs = []

    for i, feature in enumerate(feature_names):
        # Slider for each coefficient
        coef_value = st.sidebar.slider(f"Coefficient for {feature}:", -10.0, 10.0, float(model.coef_[i]))
        adjusted_coefs.append(coef_value)
    
    # Update session state with adjusted coefficients
    st.session_state.adjusted_coefs = adjusted_coefs

    # Create and update the adjusted model using session state
    adjusted_model = LinearRegression()
    adjusted_model.coef_ = np.array(st.session_state.adjusted_coefs)
    adjusted_model.intercept_ = model.intercept_

    # Predict with the adjusted coefficients
    y_pred_adjusted = adjusted_model.predict(X_test)

    # Get the predictions from the original model
    y_pred = model.predict(X_test)

    # Display the formula of the original and adjusted models
    def formula(model, feature_names):
        formula_str = f"y = {model.intercept_:.2f} "
        for coef, feature in zip(model.coef_, feature_names):
            formula_str += f"+ ({coef:.2f} * {feature}) "
        return formula_str

    st.write("### Original Model Formula:")
    st.write(formula(model, feature_names))

    st.write("### Adjusted Model Formula:")
    st.write(formula(adjusted_model, feature_names))

    # Create DataFrame for predictions
    df = pd.DataFrame({
        'Index': np.arange(len(y_test)),
        'Actual Values': y_test,
        'Model Predictions': y_pred,
        'Adjusted Predictions': y_pred_adjusted
    })

    # Create DataFrame for model parameters
    params_df = pd.DataFrame({
        "Parameter": ["Intercept"] + [f"Coefficient {name}" for name in feature_names],
        "Original Value": [model.intercept_] + list(model.coef_),
        "Adjusted Value": [adjusted_model.intercept_] + list(adjusted_model.coef_)
    })

    # Visualization with Plotly
    fig = go.Figure()

    # Add traces for each line
    fig.add_trace(go.Scatter(x=df['Index'], y=df['Actual Values'], mode='lines', name='Actual Values', line=dict(color='cyan', width=2.5)))
    fig.add_trace(go.Scatter(x=df['Index'], y=df['Model Predictions'], mode='lines', name='Model Predictions', line=dict(color='yellow', width=2.5)))
    fig.add_trace(go.Scatter(x=df['Index'], y=df['Adjusted Predictions'], mode='lines', name='Adjusted Predictions', line=dict(color='red', width=2.5)))

    # Update layout
    fig.update_layout(
        title='Comparison between Actual Values and Predictions',
        xaxis_title='Sample Index',
        yaxis_title='Y Value',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white',
        legend=dict(x=0, y=1, traceorder='normal')
    )

    st.plotly_chart(fig)

    # Display model parameters in a table
    st.write("### Model Parameters:")
    st.dataframe(params_df)

elif option == "Model Parameters":
    st.sidebar.header("Adjust Model Parameters")

    # Display sliders to adjust coefficients
    feature_names = diabetes.feature_names
    adjusted_coefs = []

    for i, feature in enumerate(feature_names):
        # Slider for each coefficient
        coef_value = st.sidebar.slider(f"Coefficient for {feature}:", -10.0, 10.0, float(model.coef_[i]))
        adjusted_coefs.append(coef_value)
    
    # Update session state with adjusted coefficients
    st.session_state.adjusted_coefs = adjusted_coefs

    # Adjust the model with the selected coefficients
    adjusted_model = LinearRegression()
    adjusted_model.coef_ = np.array(adjusted_coefs)
    adjusted_model.intercept_ = model.intercept_

    # Predict with the adjusted coefficients
    y_pred = model.predict(X_test)
    y_pred_adjusted = adjusted_model.predict(X_test)

    # Create a DataFrame to show model parameters
    params_df = pd.DataFrame({
        "Parameter": ["Intercept"] + [f"Coefficient {name}" for name in feature_names],
        "Original Value": [model.intercept_] + list(model.coef_),
        "Adjusted Value": [adjusted_model.intercept_] + list(adjusted_model.coef_)
    })

    st.write("### Adjusted Linear Regression Model Parameters:")
    st.dataframe(params_df)

    # Calculate evaluation metrics for both models
    def calculate_metrics(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }

    metrics_model = calculate_metrics(y_test, y_pred)
    metrics_adjusted = calculate_metrics(y_test, y_pred_adjusted)

    # Create DataFrame for metrics
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "MSE", "R2"],
        "Original Model": [metrics_model["MAE"], metrics_model["MSE"], metrics_model["R2"]],
        "Adjusted Model": [metrics_adjusted["MAE"], metrics_adjusted["MSE"], metrics_adjusted["R2"]]
    })

    st.write("### Model Evaluation Metrics:")
    st.dataframe(metrics_df)

elif option == "Feature Importance":
    # In linear regression, feature importance is represented by the coefficients
    importances = model.coef_
    feature_names = diabetes.feature_names

    # Create DataFrame to show feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Display the feature importance table
    st.write("### Feature Importance in the Linear Regression Model:")
    st.dataframe(importance_df)

    # Visualization of feature importance with Plotly
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=importance_df["Feature"],
        y=importance_df["Importance"],
        marker_color='teal'
    ))

    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Feature',
        yaxis_title='Importance',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white'
    )

    st.plotly_chart(fig)
