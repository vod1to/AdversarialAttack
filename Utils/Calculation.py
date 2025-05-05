import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def analyze_model_l2_degradation(file_path, model_name):
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Convert to float values
    values = [float(line.strip()) for line in lines]
    
    # Define attack types
    attacks = ["FGSM", "PGD", "BIM", "MIFGSM", "CW", "SPSA", "Square", "Baseline"]
    
    # Assuming each attack has 100 values: 50 genuine pairs followed by 50 impostor pairs
    genuine_data = {}
    impostor_data = {}
    
    # Split data into genuine and impostor pairs
    for i, attack in enumerate(attacks):
        start_idx = i * 100
        mid_idx = start_idx + 50
        end_idx = start_idx + 100
        
        if i < 7:  # For the attack types
            genuine_values = values[start_idx:mid_idx]
            impostor_values = values[mid_idx:end_idx]
        else:  # For the baseline
            genuine_values = values[700:750]
            impostor_values = values[750:800]
            
        genuine_data[attack] = genuine_values
        impostor_data[attack] = impostor_values
    
    # Calculate averages
    genuine_avg = {attack: np.mean(values) for attack, values in genuine_data.items()}
    impostor_avg = {attack: np.mean(values) for attack, values in impostor_data.items()}
    
    # Calculate degradation - for genuine pairs, smaller L2 is better
    # For impostor pairs, larger L2 is better
    genuine_baseline = genuine_avg["Baseline"]
    impostor_baseline = impostor_avg["Baseline"]
    
    genuine_degradation = {}
    impostor_degradation = {}
    
    for attack in attacks[:-1]:  # Exclude baseline
        # For genuine pairs: negative change is good, positive change is bad
        genuine_current = genuine_avg[attack]
        genuine_deg = ((genuine_current - genuine_baseline) / genuine_baseline) * 100
        genuine_degradation[attack] = genuine_deg
        
        # For impostor pairs: positive change is good, negative change is bad
        impostor_current = impostor_avg[attack]
        impostor_deg = ((impostor_current - impostor_baseline) / impostor_baseline) * 100
        impostor_degradation[attack] = impostor_deg
    
    # Create DataFrames
    df_genuine_avg = pd.DataFrame([genuine_avg], index=[f"{model_name}_Genuine"])
    df_impostor_avg = pd.DataFrame([impostor_avg], index=[f"{model_name}_Impostor"])
    
    df_genuine_deg = pd.DataFrame([genuine_degradation], index=[f"{model_name}_Genuine"])
    df_impostor_deg = pd.DataFrame([impostor_degradation], index=[f"{model_name}_Impostor"])
    
    # Create visualizations with appropriate interpretation
    
    # Genuine pairs visualization (negative change is good - inverting the color scheme)
    plt.figure(figsize=(10, 6))
    attacks_without_baseline = attacks[:-1]
    genuine_deg_values = [genuine_degradation[a] for a in attacks_without_baseline]
    bars = plt.bar(attacks_without_baseline, genuine_deg_values)
    
    # Color bars based on interpretation (red for bad, green for good)
    for i, bar in enumerate(bars):
        if genuine_deg_values[i] > 0:  # Increase in L2 for genuine pairs is bad
            bar.set_color('red')
        else:  # Decrease in L2 for genuine pairs is good
            bar.set_color('green')
    
    plt.title(f"L2 Distance Change for {model_name} - Genuine Pairs\n(negative % is better robustness)")
    plt.xlabel("Attack Type")
    plt.ylabel("% Change from Baseline")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_name}_genuine_l2_change.png", dpi=300)
    
    # Impostor pairs visualization (positive change is good)
    plt.figure(figsize=(10, 6))
    impostor_deg_values = [impostor_degradation[a] for a in attacks_without_baseline]
    bars = plt.bar(attacks_without_baseline, impostor_deg_values)
    
    # Color bars based on interpretation (red for bad, green for good)
    for i, bar in enumerate(bars):
        if impostor_deg_values[i] < 0:  # Decrease in L2 for impostor pairs is bad
            bar.set_color('red')
        else:  # Increase in L2 for impostor pairs is good
            bar.set_color('green')
    
    plt.title(f"L2 Distance Change for {model_name} - Impostor Pairs\n(positive % is better robustness)")
    plt.xlabel("Attack Type")
    plt.ylabel("% Change from Baseline")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{model_name}_impostor_l2_change.png", dpi=300)
    
    # Print results
    print(f"\nResults for {model_name}:")
    
    print("\nGenuine Pairs - L2 Distance Change (negative % is better):")
    for attack, value in genuine_degradation.items():
        status = "BETTER" if value < 0 else "WORSE"
        print(f"{attack}: {value:.2f}% ({status})")
    
    print("\nImpostor Pairs - L2 Distance Change (positive % is better):")
    for attack, value in impostor_degradation.items():
        status = "BETTER" if value > 0 else "WORSE"
        print(f"{attack}: {value:.2f}% ({status})")
    
    # Save results to CSV
    df_genuine_avg.to_csv(f"{model_name}_genuine_l2_averages.csv")
    df_impostor_avg.to_csv(f"{model_name}_impostor_l2_averages.csv")
    df_genuine_deg.to_csv(f"{model_name}_genuine_l2_change.csv")
    df_impostor_deg.to_csv(f"{model_name}_impostor_l2_change.csv")
    
    return df_genuine_deg, df_impostor_deg

# Run the analysis
if __name__ == "__main__":
    model_name = "Facenet"  # Change this for each model
    file_path = "E:\AdversarialAttack-2\L2_Values\l2_values_Facenet.txt"  # Change to your file naming convention
    
    genuine_deg, impostor_deg = analyze_model_l2_degradation(file_path, model_name)
    
    print(f"Analysis complete for {model_name}. Check the generated CSV files and images.")