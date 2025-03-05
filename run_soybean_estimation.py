#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Soybean Meal Futures SDE Parameter Estimation Program
"""

from csv_data_estimation import run_parameter_estimation
import os
from datetime import datetime

if __name__ == "__main__":
    print("=== Soybean Meal Futures SDE Parameter Estimation ===")
    print("CSV file: soybean_meal_futures.csv")
    print("Using closing price data for parameter estimation\n")
    
    # Create output directory if it doesn't exist
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"soybean_meal_futures_results_{timestamp}.pdf")
    
    # Run parameter estimation
    estimated_params, L_estimated, param_stats = run_parameter_estimation(
        csv_file_path="soybean_meal_futures.csv",
        price_column="收盘价",
        date_column="交易时间",
        save_results=True,
        output_file=output_file
    )
    
    print("\n=== Parameter Estimation Complete ===")
    print(f"Results saved to: {output_file}")
    print("Please check the PDF file for detailed results") 