import os
import uuid
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import hashlib

# Define the directory where uploaded files will be stored
UPLOAD_DIRECTORY = "uploaded_files"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

def main() -> None:
    """
    Main function to run the streamlit application.
    """
    st.title('Streamlit CSV Plotter')

    # Create tabs
    tab = st.sidebar.radio("Choose a Tab", ["Introduction", "Plotting", "Inference"])

    if tab == "Introduction":
        display_introduction()

    elif tab == "Plotting":
        plot_uploaded_file()

    elif tab == "Inference":
        make_predictions()

def display_introduction() -> None:
    """
    Display the introduction tab.
    """
    st.header("Introduction")
    st.write("This is a Streamlit app that allows you to upload CSV files and select columns to plot. \
        The uploaded files are stored historically, enabling you to select which files you want to plot.")

def plot_uploaded_file() -> None:
    """
    Handles the file upload and plotting process.
    """
    st.header("CSV Plotter")

    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

    # If a file has been uploaded
    if uploaded_file is not None:
        # If a new file has been uploaded
        if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
            handle_file_upload(uploaded_file)

    # List the files in the upload directory
    files = [f for f in os.listdir(UPLOAD_DIRECTORY) if os.path.isfile(os.path.join(UPLOAD_DIRECTORY, f))]

    if files:
        # Save the selected file in the session state
        if 'selected_file' not in st.session_state:
            st.session_state.selected_file = files[0]

        st.session_state.selected_file = st.sidebar.selectbox("Select a previously uploaded file", files, index=files.index(st.session_state.selected_file), key='select_file')

        handle_file_selection(files)

def handle_file_upload(uploaded_file) -> None:
    """
    Handles the upload and storage of the file.

    Args:
        uploaded_file: The uploaded file.
    """
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

def handle_file_selection(files: list) -> None:
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = files[0]

    st.session_state.selected_file = st.sidebar.selectbox("Select a previously uploaded file", files, index=files.index(st.session_state.selected_file), key='prev_file')
    selected_file_path = os.path.join(UPLOAD_DIRECTORY, st.session_state.selected_file)

    if selected_file_path:
        selected_df = pd.read_csv(selected_file_path)

        numeric_columns = selected_df.select_dtypes(include=['float64', 'int64']).columns

        if 'x_column' not in st.session_state:
            st.session_state.x_column = numeric_columns[0]
        if 'y_column' not in st.session_state:
            st.session_state.y_column = numeric_columns[0]

        st.session_state.x_column = st.sidebar.selectbox("Select column for x-axis", numeric_columns, index=list(numeric_columns).index(st.session_state.x_column))
        st.session_state.y_column = st.sidebar.selectbox("Select column for y-axis", numeric_columns, index=list(numeric_columns).index(st.session_state.y_column))

        n_clusters = st.sidebar.slider("Number of clusters for KMeans", 2, 10, 2)

        plot_type = st.sidebar.selectbox("Select plot type", ["line", "histogram", "scatter", "k-means scatter"], key='plots')

        apply_button = st.sidebar.button('Apply')
        delete_button = st.sidebar.button('Delete Selected File')

        if delete_button:
            os.remove(selected_file_path)
            st.sidebar.success(f'{st.session_state.selected_file} has been deleted.')
            del st.session_state.selected_file
            return

        if apply_button:
            handle_plotting(selected_df, plot_type, st.session_state.x_column, st.session_state.y_column, n_clusters)

def handle_plotting(selected_df: pd.DataFrame, plot_type: str, x_column: str, y_column: str, n_clusters: int) -> None:
    """
    Handles the plotting of selected columns.

    Args:
        selected_df: The selected DataFrame.
        plot_type: The type of plot to generate.
        x_column: The column for the x-axis.
        y_column: The column for the y-axis.
        n_clusters: The number of clusters for KMeans.
    """
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
    if plot_type == "k-means scatter":
        handle_kmeans_scatter(selected_df, x_column, y_column, n_clusters)

def handle_kmeans_scatter(selected_df: pd.DataFrame, x_column: str, y_column: str, n_clusters: int) -> None:
    """
    Handles the creation and plotting of a KMeans scatter plot.

    Args:
        selected_df: The selected DataFrame.
        x_column: The column for the x-axis.
        y_column: The column for the y-axis.
        n_clusters: The number of clusters for KMeans.
    """
    # Prepare data for KMeans
    clean_df = selected_df[[x_column, y_column]].dropna()

    if clean_df.empty:
        st.error('The selected columns contain only NaN values. Please select other columns.')
        return

    X = clean_df.values

    # Create and fit KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)

    # Train a classifier on the clustering result
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X, kmeans.labels_)

    # Save the classifier and the data to files
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    np.savetxt('data.csv', X, delimiter=',')
    np.savetxt('labels.csv', kmeans.labels_, delimiter=',')

    st.success('Classifier model saved successfully!')

    # Create scatter plot with color-coding for each cluster
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    # Add a legend for the clusters
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper right", title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)


def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def generate_plot_data(clf, X, labels):
    unique_labels = np.unique(labels)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return unique_labels, Z, xx, yy

def plot_scatter(data, labels, clf, xx, yy, input_array=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the original data points with their clusters
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(data[mask, 0], data[mask, 1], label=f'Cluster {int(label)}')

    # Plot the contourf overlay
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)

    if input_array is not None:
        ax.scatter(input_array[:, 0], input_array[:, 1], c='red', marker='*', s=150, label='Prediction')

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    st.pyplot(fig)



def make_predictions() -> None:
    st.header("Make Predictions")
    if "classifier.pkl" in os.listdir():
        with open('classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        X = np.loadtxt('data.csv', delimiter=',')
        labels = np.loadtxt('labels.csv', delimiter=',')

        if 'input_values' not in st.session_state:
            st.session_state.input_values = ''

        st.session_state.input_values = st.text_input('Enter your values (comma-separated):', value=st.session_state.input_values)

        if st.button('Predict'):
            input_array = np.array(st.session_state.input_values.split(','), dtype=float).reshape(1, -1)
            prediction = clf.predict(input_array)
            st.write(f'The predicted cluster for your values is: {prediction[0]}')

            # Generate the plot data if it doesn't exist in the session state
            if 'plot_data' not in st.session_state:
                # Generate the plot data
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                     np.arange(y_min, y_max, 0.1))
                st.session_state.plot_data = (X, labels, clf, xx, yy)

            # Plot the data
            plot_scatter(*st.session_state.plot_data, input_array)
    else:
        st.write("No classifier has been trained yet. Please create a classifier at the Plotting page.")

if __name__ == "__main__":
    main()
