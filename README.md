# Population-Centric Optimization of Air Quality Monitoring Networks in Data-Sparse Urban Regions

[!(doi.org/10.5194/egusphere-egu25-4723)](doi.org/10.5194/egusphere-egu25-4723)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Description

This project provides a framework for determining the optimal placement of monitors (e.g., air quality sensors, surveillance cameras, public service kiosks) based on population density and other demographic data. The goal is to maximize coverage and effectiveness by placing resources where they are most needed. The core of this project is an algorithm that ingests population data and outputs a set of optimal coordinates for monitor placement.

## Key Features

* **Data-Driven Optimization:** Uses population data to find the most effective locations for monitors.
* **Algorithm:** Implements a weighted k-means clustering algorithm to calculate optimal placements based on population density.
* **Interactive Web Application:** A user-friendly Streamlit application (`app.py`) for visual analysis and optimization.
* **Extensible:** The project is designed to be adaptable for different types of monitors and geographical areas.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have Python 3.8 or later installed. You will also need `pip` to install the required packages.

* [Python 3.8+](https://www.python.org/downloads/)
* [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/mahadnav/population-optimized-monitors.git](https://github.com/mahadnav/population-optimized-monitors.git)
    cd population-optimized-monitors
    ```

2.  **Create and activate a virtual environment (recommended):**
    * **On macOS and Linux:**
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```
    * **On Windows:**
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

The main script for running the optimization is `main.py`, which launches an interactive web application built with Streamlit.

```sh
streamlit run app.py
```

## Interactive Web Application (`main.py`)

The Streamlit application provides a guided, visual workflow for optimizing monitor placement. It leverages libraries like GeoPandas for spatial analysis, Folium for interactive maps, and Plotly for data visualization.

### Purpose

The application is designed to allow users, including those without a technical background, to perform a sophisticated geospatial analysis. It abstracts the complex backend processing into a series of simple, interactive steps to determine the most effective locations for air quality monitors based on where people live.

### Application Workflow

The application guides the user through the following sequential steps:

1.  **Define Your Airshed:** The user begins by drawing a rectangle on an interactive map to define the geographical area of interest (the "airshed"). This boundary is used for all subsequent analyses.

2.  **Generate Grid & Upload Population Data:**
    * Once the airshed is confirmed, the application generates a high-resolution grid (0.01° x 0.01°) over the entire area.
    * The user is then prompted to upload a GeoTIFF file containing population data for that region (e.g., from [WorldPop](https://www.worldpop.org/)).

3.  **Run Population Analysis:**
    * With a single click, the user initiates the core analysis. The application uses `rasterstats` to calculate the total population within each cell of the grid.
    * This process, known as zonal statistics, is computationally intensive and is handled with a progress bar for user feedback.

4.  **Review Population Data:** The results are presented in three tabs for review:
    * **Population Map:** A heatmap visualization of population density across the airshed.
    * **Population Distribution:** A histogram showing the frequency of population counts in grid cells, categorized into "Low" and "High" density.
    * **Download Grid Data:** An option to download the processed population-per-cell data as a CSV file.

5.  **Configure & Run Optimization:**
    * The user specifies the number of monitors to be placed in both high-density and low-density areas, as well as the minimum allowable distance between any two monitors.
    * The application then runs a **weighted k-means clustering algorithm** on the low and high-density grid cells separately. The population of each cell serves as the weight, ensuring that the resulting cluster centers (the optimal monitor locations) are pulled toward areas with more people.
    * Finally, it merges any resulting monitor locations that are closer than the specified minimum distance.

6.  **Review Final Results:** The final optimized monitor locations are displayed in three tabs:
    * **Optimized Monitors Map:** An interactive map showing the final proposed monitor locations as markers.
    * **Visualize Clusters:** Scatter plots that visualize how the grid cells were grouped into clusters during the optimization.
    * **Download Data:** A table and a download button for the final monitor coordinates in CSV format.

## File Structure

Here is an overview of the key files in this repository:

```
.
├── .gitignore       # Specifies intentionally untracked files to ignore
├── LICENSE          # Contains the project's license (MIT License)
├── README.md        # This file
├── main.py          # The main entry point for the Streamlit application
├── requirements.txt # A list of the Python packages required to run the project
└── helpers/
    └── utils.py     # Helper functions for the core analysis (e.g., k-means)
```

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  **Fork the Project**
2.  **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3.  **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4.  **Push to the Branch** (`git push origin feature/AmazingFeature`)
5.  **Open a Pull Request**

## License

This project is distributed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

* Population raster data from WorldPop
* A project associated to Pakistan Air Quality Initiative (PAQI).
