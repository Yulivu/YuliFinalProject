#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Processing PB2002 plate boundary dataProcess
"""

import os
import pandas as pd
import re

# File path settings
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataProcess', 'raw')
PROCESSING_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataProcess', 'processing')

# Ensure output directory exists
os.makedirs(PROCESSING_DATA_DIR, exist_ok=True)


def process_plate_boundaries():
    """Process plate boundary dataProcess into CSV format"""
    print("Processing plate boundaries...")

    boundaries_file = os.path.join(RAW_DATA_DIR, 'plate_boundaries', 'PB2002_boundaries.dig.txt')

    # Check if file exists
    if not os.path.exists(boundaries_file):
        print(f"Error: File {boundaries_file} does not exist!")
        return None

    # For storing boundary dataProcess
    boundaries = []

    with open(boundaries_file, 'r') as f:
        current_segment = None
        segment_points = []
        reference = None
        segment_count = 0

        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check if it's an end marker
            if "end of line segment" in line:
                if current_segment is not None and segment_points:
                    # Save current segment
                    boundaries.append({
                        'segment_name': current_segment,
                        'reference': reference,
                        'segment_id': segment_count,
                        'points': segment_points.copy()
                    })
                    # Prepare for next segment
                    segment_points = []
                    segment_count += 1
                continue

            # Check if it's a segment name line
            if '-' in line and not line.startswith('+') and not line.startswith('-') and not line.startswith(' '):
                # Split segment name and reference
                parts = line.split(None, 1)  # Split at most once
                current_segment = parts[0]
                reference = parts[1] if len(parts) > 1 else None

            # Parse coordinate lines
            elif line.startswith('+') or line.startswith('-') or line.startswith(' '):
                try:
                    # Clean spaces and split
                    coords = line.strip().replace(' ', '')
                    coords = coords.split(',')

                    if len(coords) >= 2:
                        lon = float(coords[0])
                        lat = float(coords[1])
                        segment_points.append((lon, lat))
                except Exception as e:
                    print(f"Coordinate parsing error: {e}, line: {line}")

    # Save in a processable format
    output_file = os.path.join(PROCESSING_DATA_DIR, 'plate_boundaries.csv')

    # Convert boundary dataProcess to DataFrame format
    rows = []
    for boundary in boundaries:
        for i, (lon, lat) in enumerate(boundary['points']):
            rows.append({
                'segment_name': boundary['segment_name'],
                'reference': boundary['reference'],
                'segment_id': boundary['segment_id'],
                'point_index': i,
                'longitude': lon,
                'latitude': lat
            })

    df = pd.DataFrame(rows)

    # Extract plate info from segment name
    df['plate1'] = df['segment_name'].str.split('-').str[0]
    df['plate2'] = df['segment_name'].str.split('-').str[1]

    # Get boundary type information from steps.dat file
    steps_file = os.path.join(RAW_DATA_DIR, 'plate_boundaries', 'PB2002_steps.dat.txt')
    if os.path.exists(steps_file):
        try:
            # Read steps.dat file to get boundary types
            steps_types = {}
            with open(steps_file, 'r') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split()
                        if len(parts) >= 15:  # Ensure line has enough fields
                            segment = parts[1].replace(':', '')
                            boundary_type = parts[14]
                            if segment not in steps_types:
                                steps_types[segment] = []
                            steps_types[segment].append(boundary_type)

            # Assign most common type for each segment
            for segment, types in steps_types.items():
                if types:
                    # Find most common type
                    from collections import Counter
                    most_common_type = Counter(types).most_common(1)[0][0]
                    steps_types[segment] = most_common_type

            # Add boundary type to main DataFrame
            df['boundary_type'] = df['segment_name'].map(steps_types)
        except Exception as e:
            print(f"Error processing steps.dat file: {e}")

    df.to_csv(output_file, index=False)
    print(f"Saved {len(boundaries)} plate boundary segments to {output_file}")

    return boundaries


def process_plates():
    """Process plate polygon dataProcess into CSV format"""
    print("Processing plate polygons...")

    plates_file = os.path.join(RAW_DATA_DIR, 'plate_boundaries', 'PB2002_plates.dig.txt')

    # Check if file exists
    if not os.path.exists(plates_file):
        print(f"Error: File {plates_file} does not exist!")
        return None

    # For storing plate dataProcess
    plates = []
    plate_segments = []

    with open(plates_file, 'r') as f:
        current_plate = None
        segment_points = []
        segment_count = 0

        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check if it's an end marker
            if "end of line segment" in line:
                # Save current line segment
                if current_plate is not None and segment_points:
                    plate_segments.append({
                        'plate_name': current_plate,
                        'segment_id': segment_count,
                        'points': segment_points.copy()
                    })
                    segment_points = []
                    segment_count += 1
                continue

            # Check if it's a plate name line (usually two capital letters)
            if re.match(r'^[A-Z]{2}$', line):
                # If there's already a plate and unsaved points, save them
                if current_plate is not None and segment_points:
                    plate_segments.append({
                        'plate_name': current_plate,
                        'segment_id': segment_count,
                        'points': segment_points.copy()
                    })

                # Start new plate
                current_plate = line
                segment_points = []
                segment_count = 0

                # Add plate info
                plates.append({
                    'plate_id': current_plate
                })

            # Parse coordinate lines
            elif (line.startswith('+') or line.startswith('-') or line.startswith(' ')) and ',' in line:
                try:
                    # Clean spaces and split
                    coords = line.strip().replace(' ', '')
                    coords = coords.split(',')

                    if len(coords) >= 2:
                        lon = float(coords[0])
                        lat = float(coords[1])
                        segment_points.append((lon, lat))
                except Exception as e:
                    print(f"Coordinate parsing error: {e}, line: {line}")

        # Add the last segment
        if current_plate is not None and segment_points:
            plate_segments.append({
                'plate_name': current_plate,
                'segment_id': segment_count,
                'points': segment_points
            })

    # Get plate details directly from PB2002_poles.dat.txt
    # This is more reliable than trying to parse the XLS file
    poles_dat = os.path.join(RAW_DATA_DIR, 'plate_boundaries', 'PB2002_poles.dat.txt')
    plate_details = {}
    plate_names = {
        'AF': 'Africa',
        'AM': 'Amur',
        'AN': 'Antarctica',
        'AP': 'Altiplano',
        'AR': 'Arabia',
        'AS': 'Aegean Sea',
        'AT': 'Anatolia',
        'AU': 'Australia',
        'BH': 'Balmoral Reef',
        'BR': 'Bering',
        'BS': 'Banda Sea',
        'BU': 'Burma',
        'CA': 'Caribbean',
        'CL': 'Caroline',
        'CO': 'Cocos',
        'CR': 'Conway Reef',
        'EA': 'Easter',
        'EU': 'Eurasia',
        'FT': 'Futuna',
        'GP': 'Galapagos',
        'IN': 'India',
        'JF': 'Juan Fernandez',
        'JZ': 'Juan de Fuca',
        'KE': 'Kermadec',
        'MA': 'Mariana',
        'MN': 'Manus',
        'MO': 'Maoke',
        'MS': 'Molucca Sea',
        'NA': 'North America',
        'NB': 'North Bismarck',
        'ND': 'North Andes',
        'NH': 'New Hebrides',
        'NI': 'Niuafo\'ou',
        'NZ': 'Nazca',
        'OK': 'Okhotsk',
        'ON': 'Okinawa',
        'PA': 'Pacific',
        'PM': 'Panama',
        'PS': 'Philippine Sea',
        'RI': 'Rivera',
        'SA': 'South America',
        'SB': 'South Bismarck',
        'SC': 'Scotia',
        'SL': 'Shetland',
        'SO': 'Somalia',
        'SS': 'Solomon Sea',
        'SU': 'Sunda',
        'SW': 'Sandwich',
        'TI': 'Timor',
        'TO': 'Tonga',
        'WL': 'Woodlark',
        'YA': 'Yangtze'
    }

    if os.path.exists(poles_dat):
        with open(poles_dat, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        plate_id = parts[0]
                        try:
                            plate_details[plate_id] = {
                                'plate_name': plate_names.get(plate_id, plate_id),
                                'pole_latitude': float(parts[1]),
                                'pole_longitude': float(parts[2]),
                                'rotation_rate': float(parts[3])
                            }
                        except Exception as e:
                            print(f"Error parsing pole dataProcess for {plate_id}: {e}")

    # Save plate info
    plate_info_file = os.path.join(PROCESSING_DATA_DIR, 'plates_info.csv')

    # Fill in plate details
    for plate in plates:
        plate_id = plate['plate_id']
        if plate_id in plate_details:
            plate.update(plate_details[plate_id])
        else:
            # Use the predefined names even if not in poles.dat
            plate['plate_name'] = plate_names.get(plate_id, plate_id)

    # Save plate info
    plates_df = pd.DataFrame(plates)
    plates_df.to_csv(plate_info_file, index=False)
    print(f"Saved {len(plates)} plate details to {plate_info_file}")

    # Save plate boundary points
    plates_file = os.path.join(PROCESSING_DATA_DIR, 'plates.csv')

    # Convert plate dataProcess to DataFrame format
    rows = []
    for segment in plate_segments:
        for i, (lon, lat) in enumerate(segment['points']):
            rows.append({
                'plate_id': segment['plate_name'],
                'segment_id': segment['segment_id'],
                'point_index': i,
                'longitude': lon,
                'latitude': lat
            })

    df = pd.DataFrame(rows)
    df.to_csv(plates_file, index=False)
    print(f"Saved {len(plate_segments)} plate boundary segments to {plates_file}")

    return plates


def process_poles():
    """Process plate pole dataProcess into CSV format"""
    print("Processing plate poles...")

    poles_file = os.path.join(RAW_DATA_DIR, 'plate_boundaries', 'PB2002_poles.dat.txt')

    # Check if file exists
    if not os.path.exists(poles_file):
        print(f"Error: File {poles_file} does not exist!")
        return None

    # Manually parse file
    poles = []

    with open(poles_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comment lines and empty lines
            if line.startswith('#') or not line:
                continue

            # Find where reference citation begins
            ref_start = None
            for i, char in enumerate(line):
                if char == '[':
                    ref_start = i
                    break

            if ref_start is not None:
                # Extract plate code and numerical dataProcess only
                data_part = line[:ref_start].strip()

                # Split dataProcess part
                parts = data_part.split()

                if len(parts) >= 4:
                    try:
                        pole = {
                            'plate_id': parts[0],
                            'pole_latitude': float(parts[1]),
                            'pole_longitude': float(parts[2]),
                            'rotation_rate': float(parts[3])
                        }
                        poles.append(pole)
                    except Exception as e:
                        print(f"Pole dataProcess parsing error: {e}, line: {line}")

    # Save as CSV
    if poles:
        poles_df = pd.DataFrame(poles)
        output_file = os.path.join(PROCESSING_DATA_DIR, 'plate_poles.csv')
        poles_df.to_csv(output_file, index=False)
        print(f"Saved {len(poles)} plate pole records to {output_file}")
        return poles_df
    else:
        print("No pole dataProcess found")
        return None


def main():
    """Main function"""
    print("Starting plate dataProcess processing...")

    # Process plate boundary dataProcess
    process_plate_boundaries()

    # Process plate polygon dataProcess
    process_plates()

    # Process plate pole dataProcess
    process_poles()

    print("Plate dataProcess processing complete!")


if __name__ == "__main__":
    main()