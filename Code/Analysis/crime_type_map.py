# Author : Ravisutha Sakrepatna
# Project: Chicago Crime Predictions
# Purpose of this segment: Map IUCR codes to crime types

import pandas as pd

def map_codes (path, offset=10000):
    """
    Maps IUCR codes to crime types.
    -----
    Input: Path to IUCR data
    Offset: Offset used earlier
    Output: Dictionary to convert crime code to primary crime type
    """
    # Variable declaration
    map_code_des = {}
    
    df = pd.read_csv (path)
    
    for i, row in df.iterrows():
      index = i + offset
      
      if row['PRIMARY DESCRIPTION'] not in map_code_des:
        map_code_des[row['PRIMARY DESCRIPTION']] = []
        
      map_code_des[row['PRIMARY DESCRIPTION']].append(index)
    
    return (map_code_des)

    

if __name__ == "__main__":
    map_codes ("../../Data/Static/IUCR.csv")
    
