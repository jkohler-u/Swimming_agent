import csv

# You can save your data to a file named 'data.csv' or change the filename here
filename = 'results/human_swimmer_no_foreward/test_results.csv'
# Average Forward Velocity: -0.0358
# Percentage Head Above Water: 3.83%
# Longest streak of head below water: 201 steps
    

def analyze_csv(file_path):
    velocities = []
    head_above_count = 0
    total_rows = 0
    
    max_underwater_streak = 0
    current_underwater_streak = 0

    with open(file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            total_rows += 1
            
            # 1. Collect velocity for average
            velocities.append(float(row['forward_velocity']))
            
            # 2. Track Head Above Water status
            # Note: CSV reads everything as strings, so we compare to 'True'
            is_above = row['head_above_water'] == 'True'
            
            if is_above:
                head_above_count += 1
                # Reset the underwater streak because the head is now above water
                current_underwater_streak = 0
            else:
                # Increment the current streak of being underwater (False)
                current_underwater_streak += 1
                if current_underwater_streak > max_underwater_streak:
                    max_underwater_streak = current_underwater_streak

    # Calculations
    avg_velocity = sum(velocities) / len(velocities) if velocities else 0
    percentage_above = (head_above_count / total_rows * 100) if total_rows > 0 else 0

    print(f"Analysis Results:")
    print(f"-----------------")
    print(f"Average Forward Velocity: {avg_velocity:.4f}")
    print(f"Percentage Head Above Water: {percentage_above:.2f}%")
    print(f"Longest streak of head below water: {max_underwater_streak} steps")

if __name__ == "__main__":
    analyze_csv(filename)