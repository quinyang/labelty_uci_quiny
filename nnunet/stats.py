import json
import os

# The exact path where nnU-Net saves the final validation scorecard
stats_path = "/home/jiakuny1/Projects/nnUNet_data/nnUNet_results/Dataset101_Dental/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation/summary.json"

if not os.path.exists(stats_path):
    print("Could not find the summary.json file!")
else:
    with open(stats_path, 'r') as file:
        data = json.load(file)
    
    print("\n" + "="*40)
    print(" FINAL MODEL STATISTICS (DICE SCORES)")
    print("="*40)
    
    # Extract the mean scores for each class
    class_metrics = data.get('mean', {})
    
    for class_id, metrics in class_metrics.items():
        # A Dice score of 1.0 is perfect, 0.0 is complete failure
        dice = metrics.get('Dice', 'NaN')
        
        # Format the output to look like a clean table
        if isinstance(dice, float):
            print(f" Class {class_id:2} | Score: {dice:.4f} ({dice*100:.1f}%)")
        else:
            print(f" Class {class_id:2} | Score: {dice} (Not present in validation)")
            
    print("="*40 + "\n")