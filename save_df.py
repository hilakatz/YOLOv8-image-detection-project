import os
import pandas as pd

def save_dataframe_as_csv(df, folder_name, file_name):
    # Check if the folder exists
    if not os.path.exists(folder_name):
        # Create a new folder
        os.makedirs(folder_name)

    # Define the file path
    file_path = os.path.join(folder_name, file_name+'.csv')

    # Save the DataFrame as a CSV file
    df.to_csv(file_path, index=False)

    print(f"DataFrame saved as CSV file in folder '{folder_name}' with filename '{file_name}'.")
