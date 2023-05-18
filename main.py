import os
import uuid
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the directory where uploaded files will be stored
UPLOAD_DIRECTORY = "uploaded_files"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Main function
def main():
    st.title('Streamlit CSV Plotter')

    # Create tabs
    tab = st.sidebar.selectbox("Choose a Tab", ["Introduction", "Plotting"])

    if tab == "Introduction":
        st.header("Introduction")
        st.write("This is a Streamlit app that allows you to upload CSV files and select columns to plot. \
            The uploaded files are stored historically, enabling you to select which files you want to plot.")

    elif tab == "Plotting":
        st.header("CSV Plotter")

        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

        # If a file has been uploaded
        if uploaded_file is not None:

            # If a new file has been uploaded
            if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
                st.session_state.uploaded_file_name = uploaded_file.name

                file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
                st.write(file_details)

                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(df)

                    # Generate a unique id for the file and save it to the directory
                    file_id = str(uuid.uuid4())
                    file_path = os.path.join(UPLOAD_DIRECTORY, f"{file_id}.csv")
                    df.to_csv(file_path)

                except Exception as e:
                    st.write("There was an error processing this file.")
                    return

        # List the files in the upload directory
        files = [f for f in os.listdir(UPLOAD_DIRECTORY) if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, f))]

        if files:
            selected_file = st.sidebar.selectbox("Select a previously uploaded file", files)
            selected_file_path = os.path.join(UPLOAD_DIRECTORY, selected_file)

            if selected_file_path:
                selected_df = pd.read_csv(selected_file_path)

                # Select only numeric columns for x and y axis
                numeric_columns = selected_df.select_dtypes(include=['float64', 'int64']).columns
                x_column = st.sidebar.selectbox("Select column for x-axis", numeric_columns)
                y_column = st.sidebar.selectbox("Select column for y-axis", numeric_columns)

                # Select plot type
                plot_type = st.sidebar.selectbox("Select plot type", ["line", "histogram", "scatter"])

                # Plot selected columns
                if plot_type == "line":
                    st.line_chart(selected_df[[x_column, y_column]])
                elif plot_type == "histogram":
                    fig, ax = plt.subplots()
                    ax.hist(selected_df[y_column].dropna(), bins=30)
                    st.pyplot(fig)
                elif plot_type == "scatter":
                    fig, ax = plt.subplots()
                    ax.scatter(selected_df[x_column], selected_df[y_column])
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
