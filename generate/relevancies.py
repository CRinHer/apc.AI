import numpy as np
import csv

# Generate 60,000 random numbers between 0 and 1
random_numbers = np.random.rand(60000)

# Specify the CSV file name
csv_file = './data/relevancies.csv'

# Write the numbers to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write each number as a separate row
    for number in random_numbers:
        writer.writerow([number])
