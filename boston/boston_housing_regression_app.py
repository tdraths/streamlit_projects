import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App

This app makes predictions using the **Boston House Price** dataset.
""")
st.write('---')

boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=['MEDV'])

st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider(
        'CRIM', X.CRIM.min(), X.CRIM.max(), float(X.CRIM.mean()))
    ZN = st.sidebar.slider(
        'ZN', X.ZN.min(), X.ZN.max(), float(X.ZN.mean()))
    INDUS = st.sidebar.slider(
        'INDUS', X.INDUS.min(), X.INDUS.max(), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider(
        'CHAS', X.CHAS.min(), X.CHAS.max(), float(X.CHAS.mean()))
    NOX = st.sidebar.slider(
        'NOX', X.NOX.min(), X.NOX.max(), float(X.NOX.mean()))
    RM = st.sidebar.slider(
        'RM', X.RM.min(), X.RM.max(), float(X.RM.mean()))
    AGE = st.sidebar.slider(
        'AGE', X.AGE.min(), X.AGE.max(), float(X.AGE.mean()))
    DIS  = st.sidebar.slider(
        'DIS', X.DIS.min(), X.DIS.max(), float(X.DIS.mean()))
    RAD = st.sidebar.slider(
        'RAD', X.RAD.min(), X.RAD.max(), float(X.RAD.mean()))
    TAX = st.sidebar.slider(
        'TAX', X.TAX.min(), X.TAX.max(), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider(
        'PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), float(X.PTRATIO.mean()))
    B = st.sidebar.slider(
        'B', X.B.min(), X.B.max(), float(X.B.mean()))
    LSTAT = st.sidebar.slider(
        'LSTAT', X.LSTAT.min(), X.LSTAT.max(), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM, 'ZN': ZN, 'INDUS': INDUS, 'CHAS': CHAS, 'NOX': NOX,
            'RM': RM, 'AGE': AGE, 'DIS': DIS, 'RAD': RAD, 'TAX': TAX,
            'PTRATIO': PTRATIO, 'B': B, 'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.header('Specified Input Parameters')
st.write(df)
st.write('---')

load_reg = pickle.load(open('boston_housing_reg.pkl', 'rb'))

prediction = load_reg.predict(df)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')

explainer = shap.TreeExplainer(load_reg)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature Importance based on SHAP Values')
shap.summary_plot(shap_values, X)
st.pyplot(plt, bbox_inches='tight')
st.write('---')

fig1 = plt.figure()
plt.title('Feature Importance based on SHAP Values (Bar)')
shap.summary_plot(shap_values, X, plot_type='bar')
st.pyplot(fig1, bbox_inches='tight')