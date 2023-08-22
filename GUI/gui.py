import PySimpleGUI as sg
import os.path
import subprocess  # To run image processing function
import pandas as pd
IMAGE_FORMAT = None
FILE_FORMAT = None


def process_images(image_folder, annotations_folder, database_name, image_format, file_format):
    # Run your image processing function using the provided folder path
    # Replace this with your actual image processing logic
    subprocess.run(['python', 'image_processing.py', image_folder,annotations_folder, database_name, image_format, file_format])


def run_dashboard(selected_database):
    # Run your dashboard code here
    # Replace this with your actual dashboard code
    subprocess.run(['python', 'dashboard.py', selected_database])


def delete_selected_files_in_data_folder(selected_files):
    folder_path = "data"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) and filename in selected_files:
                os.remove(file_path)
                iou_to_delete = os.path.splitext(filename)[0]
                iou_file_path = os.path.join(folder_path, f"{iou_to_delete}.txt")
                if os.path.isfile(iou_file_path):
                    os.remove(iou_file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


def get_processed_databases():
    # check if the data folder exists, if not create it
    if not os.path.exists("data"):
        os.makedirs("data")
    data_folder = "data"
    processed_databases = [
        f for f in os.listdir(data_folder) if f.endswith(".csv") and f != "coco128.csv"
    ]
    return processed_databases


# First the window layout in 2 columns
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("Annotations Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-ANNO FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]


csv_viewer_column = [
    [sg.Text("Choose a processed database from the list:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    # Display the list of processed images in the right size
    [sg.Listbox(values=get_processed_databases(), enable_events=True, size=(40, 20), key="-CSV LIST-")],
    [sg.Button("Delete Database", key="-DELETE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeparator(),
        sg.Column(csv_viewer_column),
    ],
    [
        sg.Button("Process", key="-PROCESS-"),
        sg.Button("Show Dashboard", key="-DASHBOARD-"),
        sg.Button("Exit"),
    ],
]

window = sg.Window("Image Processor and Dashboard", layout)

# Run the Event Loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Exit":
        # delete_files_in_data_folder()
        break

    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".gif", ".jpg"))
        ]
        IMAGE_FORMAT = fnames[0].split(".")[-1]
        window["-FILE LIST-"].update(fnames)

    elif event == "-ANNO FOLDER-":
        folder = values["-ANNO FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".txt",".xml"))
        ]
        FILE_FORMAT = fnames[0].split(".")[-1]
        window["-FILE LIST-"].update(fnames)

    elif event == "-PROCESS-":
        images_folder = values["-FOLDER-"]
        annotations_folder = values["-ANNO FOLDER-"]
        if images_folder:
            # Ask for a database name using an input popup
            database_name = sg.popup_get_text("Enter a name for the database:")
            if database_name:
                # Call your image processing function
                process_images(images_folder, annotations_folder, database_name, IMAGE_FORMAT, FILE_FORMAT)
                sg.popup("Image processing completed!", title="Process Images")
        # Update the list of processed images
        window["-CSV LIST-"].update(values=get_processed_databases())

    elif event == "-DASHBOARD-":
        # Open another window for the dashboard (replace with your actual dashboard code)
        selected_processed_file = values["-CSV LIST-"]
        if selected_processed_file:
            selected_processed_file = selected_processed_file[0]
            run_dashboard(selected_processed_file)

    elif event == "-DELETE-":
        selected_files = values["-CSV LIST-"]
        if selected_files:
            selected_files = selected_files[0]
            delete_selected_files_in_data_folder(selected_files)
            window["-CSV LIST-"].update(values=get_processed_databases())


window.close()
