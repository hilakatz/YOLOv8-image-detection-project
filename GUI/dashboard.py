import PySimpleGUI as sg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

IMAGE_PROPERTIES = ['iou', 'aspect_ratio', 'brightness', 'contrast', 'sharpness', 'noise', 'saturation', 'entropy', 'edges', 'estimate_noise', 'red_channel', 'blue_channel', 'green_channel','salt_pepper_noise','blurriness']

def plot_histogram(data_series,selected_column):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(data_series, kde=True)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(f"{selected_column} Histogram")
    plt.tight_layout()
    temp_histogram_plot = "temp_histogram_plot.png"
    plt.savefig(temp_histogram_plot)
    plt.clf()  # Clear the figure to release memory
    return temp_histogram_plot


# Function to run the dashboard
def run_dashboard(data_path, iou_path, baseline_data_path):
    # Load the processed data from the CSV file
    processed_data = pd.read_csv(data_path)
    with open(f'data/{iou_path}.txt','r') as f:
        iou = f.read()

    baseline_data = pd.read_csv(baseline_data_path)
    # Change avg_score name to iou_score for readability
    processed_data.rename(columns={"avg_score": "iou"}, inplace=True)
    columns_to_present = [column for column in processed_data.columns if column in IMAGE_PROPERTIES]
    stat = ""
    # Update the layout to include the histogram section
    layout = [
        [sg.Frame("IOU Score", [[sg.Text(f"{iou}", key="-IOU-SCORE-")]])],
        [sg.Text("Select two columns for correlation plot, \n or just one from the left for histogram plot:     "), sg.Text(f"     {stat} Statistics Table:", key="-STATISTICS-TABLE-NAME-")],
        [sg.Listbox(columns_to_present, size=(15, 6), key="-COLUMN1-", enable_events=True), sg.Listbox(columns_to_present, size=(15, 6), key="-COLUMN2-", enable_events=True), sg.VSeparator(), sg.Table(values=[],size=(100, 6), headings=["Statistic", "Processed Value", "Baseline Value"], key="-STATISTICS-TABLE-")],
        [sg.Button("Generate Correlation Plot"), sg.Button("Generate Histogram"), sg.Button("Exit"), sg.Checkbox("Show Baseline", key="-SHOW-HISTOGRAM-", enable_events=True)],
        [sg.Image(key="-PLOT-", size=(800, 600))],
    ]

    window = sg.Window("Dashboard", layout)

    show_histogram = False  # Initialize the flag for histogram view

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "Generate Correlation Plot":
            selected_column1 = values["-COLUMN1-"][0]
            selected_column2 = values["-COLUMN2-"][0]

            # Generate correlation plot
            if selected_column1 != selected_column2:
                # Clear previous plot
                temp_plot_file = "temp_plot.png"
                if os.path.exists(temp_plot_file):
                    os.remove(temp_plot_file)

                sns.set(style="white")
                plot = sns.scatterplot(data=processed_data, x=selected_column1, y=selected_column2)
                plot.set_xlabel(selected_column1)
                plot.set_ylabel(selected_column2)
                plot.set_title(f"Correlation Plot between {selected_column1} and {selected_column2}")
                plt.tight_layout()

                # Save the plot to a temporary file and update the Image element
                plot.get_figure().savefig(temp_plot_file)
                window["-PLOT-"].update(filename=temp_plot_file)
                plot.get_figure().clf()
            else:
                sg.popup("Please select two different columns for the correlation plot.")

        elif event == "-SHOW-HISTOGRAM-":
            show_histogram = values["-SHOW-HISTOGRAM-"]

        elif event == "Generate Histogram":
            selected_column = values["-COLUMN1-"][0]  # Choose the column for the histogram
            window["-STATISTICS-TABLE-NAME-"].update(value=f"     {selected_column} Statistics Table:")
            selected_column_data = processed_data[selected_column]
            # Update the values for the corresponding statistics in the table
            window["-STATISTICS-TABLE-"].update(values=[
                ["Count", selected_column_data.count(), ""],
                ["Mean", selected_column_data.mean(), ""],
                ["Median", selected_column_data.median(), ""],
                ["Minimum", selected_column_data.min(), ""],
                ["Maximum", selected_column_data.max(), ""],
                ["Standard Deviation", selected_column_data.std(), ""]
            ])
            if show_histogram:
                baseline_column_data = baseline_data[selected_column]
                window["-STATISTICS-TABLE-"].update(values=[
                    ["Count", selected_column_data.count(), baseline_column_data.count()],
                    ["Mean", selected_column_data.mean(), baseline_column_data.mean()],
                    ["Median", selected_column_data.median(), baseline_column_data.median()],
                    ["Minimum", selected_column_data.min(), baseline_column_data.min()],
                    ["Maximum", selected_column_data.max(), baseline_column_data.max()],
                    ["Standard Deviation", selected_column_data.std(), baseline_column_data.std()]
                ])

                # Plot histograms
                sns.set(style="whitegrid")
                plt.figure(figsize=(8, 5))
                sns.histplot(selected_column_data, kde=True,  label="Processed Data")
                sns.histplot(baseline_column_data, kde=True, label="Baseline Data")
                plt.xlabel("Value")
                plt.ylabel("Frequency")
                plt.title(f"{selected_column} Histogram")
                plt.legend()
                plt.tight_layout()
                temp_histogram_plot = "temp_histogram_plot.png"
                plt.savefig(temp_histogram_plot)

                # histogram_plot_path = plot_histogram(selected_column_data,selected_column)
                # # Update the Image element with the new histogram plot
                window["-PLOT-"].update(filename=temp_histogram_plot)
                plt.clf()
            else:
                histogram_plot_path = plot_histogram(selected_column_data, selected_column)
                # Update the Image element with the new histogram plot
                window["-PLOT-"].update(filename=histogram_plot_path)
        temp_plot_file = "temp_plot.png"
        if os.path.exists(temp_plot_file):
            os.remove(temp_plot_file)
        temp_histogram_plot = "temp_histogram_plot.png"
        if os.path.exists(temp_histogram_plot):
            os.remove(temp_histogram_plot)

    window.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dashboard.py <folder_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    iou = os.path.splitext(data_path)[0]
    # Path to the processed data CSV file
    data_path = f"data/{data_path}"

    # Run the dashboard
    run_dashboard(data_path, iou, baseline_data_path="data/coco128.csv")
