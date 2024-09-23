import streamlit as st
import pandas as pd
import os
from Exploaratory_data_analysis import ESD  # Ensure this is the correct path to your ESD class

# Define the path to save the file
save_path = os.path.join("data_sample", "sample.csv")

# Function to save uploaded files
def save_uploaded_file(uploaded_file):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

# Streamlit UI
st.title("Upload and Process Data")

# Dropdown for file types
file_type = st.selectbox("Select file type", ["CSV", "JSON", "SQL", "Text"])

# File uploader
uploaded_file = st.file_uploader("Upload your file", type=["csv", "json", "sql", "txt"])

if uploaded_file is not None:
    # Save the file
    save_path = save_uploaded_file(uploaded_file)
    st.success(f"File saved as: {save_path}")

    # Process the uploaded file
    if file_type == "CSV":
        df = pd.read_csv(save_path)
    elif file_type == "JSON":
        df = pd.read_json(save_path)
    elif file_type == "SQL":
        import sqlite3
        conn = sqlite3.connect(":memory:")
        df = pd.read_sql(conn.cursor().execute("SELECT * FROM table").fetchall(), conn)
    elif file_type == "Text":
        df = pd.read_csv(save_path, delimiter='\t')
    else:
        st.error("Unsupported file type")

    st.write("Data preview:")
    st.write(df.head())

    # Create an instance of ESD
    esd = ESD()
    esd.data = df  # Set the data for the ESD instance

    # Apply data transformations
    esd.data_Preprocessing()
    st.write("Data after preprocessing:")
    st.write(esd.data)

    # Display additional data analysis results
    st.subheader("Data Description")
    st.write(esd.data_describe())

