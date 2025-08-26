import os
import pandas as pd
import pyreadstat

# Set your folder path here
input_folder = f"Data/need_tranform"
output_folder = f"Data/need_tranform"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop over all .sav files in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".sav"):
        sav_path = os.path.join(input_folder, filename)
        csv_name = os.path.splitext(filename)[0] + ".csv"
        csv_path = os.path.join(output_folder, csv_name)
        
        try:
            # Load the .sav file
            df, meta = pyreadstat.read_sav(sav_path)
            
            # Save to CSV (utf-8 encoding)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Converted: {filename} → {csv_name}")
        except Exception as e:
            print(f"❌ Error with {filename}: {e}")

print("Done converting all .sav files.")
