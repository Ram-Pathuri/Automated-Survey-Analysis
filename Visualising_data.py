import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
def load_data():
    data = pd.read_csv("data_sample/sample.csv")
    data1 = pd.read_csv('data_sample/sample1.csv')
    return data, data1

# Load the data
df, df1 = load_data()

# Select the data frame to analyze
data_choice = st.selectbox('Select the data frame to analyze', ['Original Data', 'Transformed Data'])

# Set the selected data frame
if data_choice == 'Original Data':
    df = df
elif data_choice == 'Transformed Data':
    df = df1

st.title('Visualize the Data')

# Select columns
columns = df.columns.tolist()
selected_columns = st.multiselect('Select columns to visualize', columns)

# Select graph type
graph_type = st.selectbox('Select the type of graph',
                          ['Line Plot', 'Bar Plot', 'Scatter Plot', 'Histogram', 'Heatmap', 'Distribution',
                           'Factor_analysis', 'Multiple Regression'])

# Generate graph based on user inputs
if len(selected_columns) == 2 and graph_type == 'Scatter Plot':
    st.write(f'Scatter Plot of {selected_columns[0]} vs {selected_columns[1]}')
    fig, ax = plt.subplots()
    ax.scatter(df[selected_columns[0]], df[selected_columns[1]])
    ax.set_xlabel(selected_columns[0])
    ax.set_ylabel(selected_columns[1])
    st.pyplot(fig)

elif len(selected_columns) == 1 and graph_type == 'Histogram':
    st.write(f'Histogram of {selected_columns[0]}')
    fig, ax = plt.subplots()
    ax.hist(df[selected_columns[0]], bins=20)
    ax.set_xlabel(selected_columns[0])
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

elif len(selected_columns) == 1 and graph_type == 'Line Plot':
    st.write(f'Line Plot of {selected_columns[0]}')
    fig, ax = plt.subplots()
    ax.plot(df[selected_columns[0]])
    ax.set_xlabel('Index')
    ax.set_ylabel(selected_columns[0])
    st.pyplot(fig)

    st.write('Factor Loadings:')
    st.write(loadings)

    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Factor Loadings Heatmap')
    st.pyplot()

    plt.figure(figsize=(10, 6))
    for i in range(n_factors):
        plt.hist(factor_df[f'Factor_{i + 1}'], bins=30, alpha=0.5, label=f'Factor_{i + 1}')
    plt.title('Distribution of Factor Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    st.pyplot()

