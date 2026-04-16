import cdsapi
import os # <- Import this to handle folder paths

client = cdsapi.Client()
dataset = "reanalysis-era5-pressure-levels"

months = ["01", "02", "03", "04", "05", "06", 
          "07", "08", "09", "10", "11", "12"]

# DEFINE YOUR EXACT DESTINATION FOLDER HERE
# Change this to your actual Snellius scratch or project path
output_dir = "/scratch-shared/ekasteleyn/era5_data/" 

# Create the folder if it doesn't exist yet
os.makedirs(output_dir, exist_ok=True)

# Base parameters shared between both requests
base_request = {
    "variable": ["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"],
    "year": ["2020"],
    "day": [str(i).zfill(2) for i in range(1, 32)], # Auto-generates 01 to 31
    "time": ["00:00"],
    "pressure_level": ["1", "2", "3", "5", "7", "10", "20", "30", "50", "70", "100", "125", "150", "175", "200", "225", "250", "300", "350", "400", "450", "500", "550", "600", "650", "700", "750", "775", "800", "825", "850", "875", "900", "925", "950", "975", "1000"],
    "data_format": "grib",
    "download_format": "unarchived",
    "grid": ["0.5", "0.5"]  
}

for month in months:
    print(f"Downloading data for Month: {month}")
    
    # 1. Download the 10 Ensemble Members
    req_members = base_request.copy()
    req_members["month"] = [month]
    req_members["product_type"] = ["ensemble_members"]
    
    # Safely combine the folder path with the filename
    file_members = os.path.join(output_dir, f"era5_members_2020_{month}.grib")
    client.retrieve(dataset, req_members, file_members)
    
    # 2. Download the 1 Ensemble Mean
    req_mean = base_request.copy()
    req_mean["month"] = [month]
    req_mean["product_type"] = ["ensemble_mean"]
    
    # Safely combine the folder path with the filename
    file_mean = os.path.join(output_dir, f"era5_mean_2020_{month}.grib")
    client.retrieve(dataset, req_mean, file_mean)