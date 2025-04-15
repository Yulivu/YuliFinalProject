#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(directory):
    """Ensure that a directory exists. Create if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data():
    """
    Load CSV dataProcess files from the 'dataProcess/processing' directory.
    Files:
      - earthquakes.csv: Earthquake event information.
      - plate_boundaries.csv: Plate boundary points.
      - plates.csv: Tectonic plate outlines.
      - plates_info.csv: Plate general information.
      - plate_poles.csv: Euler poles for plates.
    """
    # Get the project's root directory based on the script location.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    eq_path = os.path.join(project_root, 'dataProcess', 'processing', 'earthquakes.csv')
    eq_df = pd.read_csv(eq_path)

    plate_boundaries_path = os.path.join(project_root, 'dataProcess', 'processing', 'plate_boundaries.csv')
    plate_boundaries_df = pd.read_csv(plate_boundaries_path)

    plates_path = os.path.join(project_root, 'dataProcess', 'processing', 'plates.csv')
    plates_df = pd.read_csv(plates_path)

    plates_info_path = os.path.join(project_root, 'dataProcess', 'processing', 'plates_info.csv')
    plates_info_df = pd.read_csv(plates_info_path)

    plate_poles_path = os.path.join(project_root, 'dataProcess', 'processing', 'plate_poles.csv')
    plate_poles_df = pd.read_csv(plate_poles_path)

    return eq_df, plate_boundaries_df, plates_df, plates_info_df, plate_poles_df

def classify_fault_type(rake):
    """
    Classify fault type based on the rake angle (in degrees) using a simple scheme:
      - Strike-slip: -45° <= rake <= 45° or (rake <= -135° or rake >= 135°)
      - Reverse: 45° < rake < 135°
      - Normal: -135° < rake < -45°
    Parameters:
      rake (float): Rake angle in degrees.
    Returns:
      str: The fault type classification.
    """
    # Normalize rake to the range [-180, 180]
    if rake > 180:
        rake = rake - 360
    elif rake < -180:
        rake = rake + 360

    if (-45 <= rake <= 45) or (rake <= -135 or rake >= 135):
        return "Strike-slip"
    elif 45 < rake < 135:
        return "Reverse"
    elif -135 < rake < -45:
        return "Normal"
    else:
        return "Oblique"

def add_fault_classification(eq_df):
    """
    Add a new column 'fault_type' to the earthquakes dataframe based on the 'rake1' value.
    """
    eq_df['rake1'] = pd.to_numeric(eq_df['rake1'], errors='coerce')
    eq_df['fault_type'] = eq_df['rake1'].apply(classify_fault_type)
    return eq_df

def analyze_earthquakes(eq_df, output_dir):
    """
    Perform exploratory dataProcess analysis on the earthquakes dataset.
    Analyses include:
      - Yearly earthquake count time series.
      - Histogram of mb values.
      - Fault type classification based on rake1.
      - Bar chart of earthquake counts by fault type.
      - Scatter plot of earthquake locations colored by fault type.
      - Histogram of rake1 distribution.
    """
    # Add fault type classification using rake1
    eq_df = add_fault_classification(eq_df)

    # Time series: count of earthquakes per year
    year_counts = eq_df['year'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.plot(year_counts.index, year_counts.values, marker='o')
    plt.title('Earthquake Counts by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'earthquake_counts_by_year.png'))
    plt.close()

    # Histogram of mb values (body wave magnitude)
    plt.figure(figsize=(10, 6))
    plt.hist(eq_df['mb'].dropna(), bins=30)
    plt.title('Distribution of mb')
    plt.xlabel('mb')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'mb_distribution.png'))
    plt.close()

    # Bar chart of fault type counts
    fault_counts = eq_df['fault_type'].value_counts()
    plt.figure(figsize=(10, 6))
    fault_counts.plot(kind='bar')
    plt.title('Earthquake Counts by Fault Type')
    plt.xlabel('Fault Type')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'fault_type_counts.png'))
    plt.close()

    # Scatter plot of earthquake locations colored by fault type
    plt.figure(figsize=(10, 8))
    fault_types = eq_df['fault_type'].unique()
    for ftype in fault_types:
        subset = eq_df[eq_df['fault_type'] == ftype]
        plt.scatter(subset['longitude'], subset['latitude'], s=5, label=ftype, alpha=0.6)
    plt.title('Earthquake Locations by Fault Type')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'earthquake_locations_fault_type.png'))
    plt.close()

    # Histogram of rake1 distribution
    plt.figure(figsize=(10, 6))
    plt.hist(eq_df['rake1'].dropna(), bins=30)
    plt.title('Distribution of Rake1')
    plt.xlabel('Rake1 (degrees)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'rake1_distribution.png'))
    plt.close()

def plot_plate_boundaries(plate_boundaries_df, output_dir):
    """
    Plot plate boundary points.
    Each point represents a location along a boundary between two plates.
    """
    plt.figure(figsize=(10, 8))
    boundary_types = plate_boundaries_df['boundary_type'].unique()
    for btype in boundary_types:
        subset = plate_boundaries_df[plate_boundaries_df['boundary_type'] == btype]
        plt.scatter(subset['longitude'], subset['latitude'], s=5, label=btype, alpha=0.7)
    plt.title('Plate Boundaries')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'plate_boundaries.png'))
    plt.close()

def plot_plates(plates_df, output_dir):
    """
    Plot tectonic plate outlines.
    Each plate's outline is drawn as a connected line based on the given points.
    """
    plt.figure(figsize=(10, 8))
    for plate_id, group in plates_df.groupby('plate_id'):
        plt.plot(group['longitude'], group['latitude'], marker='o', linestyle='-', markersize=2, label=plate_id)
    plt.title('Plate Outlines')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(output_dir, 'plate_outlines.png'), bbox_inches='tight')
    plt.close()

def main():
    # Determine output directory for exploration results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'dataProcess', 'explore')
    ensure_dir(output_dir)

    # Load dataProcess files
    eq_df, plate_boundaries_df, plates_df, plates_info_df, plate_poles_df = load_data()

    # Conduct earthquake exploratory analysis (including fault motion classification)
    analyze_earthquakes(eq_df, output_dir)

    # Plot plate boundaries and plate outlines (using reliable 2002PB dataProcess)
    plot_plate_boundaries(plate_boundaries_df, output_dir)
    plot_plates(plates_df, output_dir)

    # Additional suggestions for feature engineering:
    #   1. Compute the distance from earthquake epicenters to the nearest plate boundary.
    #   2. Analyze the relationships between magnitude, depth, and fault type.
    #   3. Incorporate Euler pole dataProcess (from plate_poles) to derive relative plate motion features.
    #   4. Explore temporal patterns in seismicity to capture dynamic changes.
    #
    # These features will be valuable inputs for machine learning models aiming at
    # earthquake fault motion pattern recognition and plate boundary identification.
    print("Exploratory dataProcess analysis completed. Results saved in:", output_dir)

if __name__ == '__main__':
    main()
