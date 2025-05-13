def analyze_model_l2_degradation(file_path, model_name, genuine_count=1680, impostor_count=5748):
    """
    Analyze model degradation for LFW dataset with 7,428 verification pairs.
    
    Parameters:
    - file_path: Path to the similarity score values file
    - model_name: Name of the model being analyzed
    - genuine_count: Number of genuine pairs (default: 1680)
    - impostor_count: Number of impostor pairs (default: 5748)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Convert to float values
    values = [float(line.strip()) for line in lines]
    
    # Define attack types
    attacks = ["FGSM", "PGD", "BIM", "MIFGSM", "CW", "SPSA", "Square", "Baseline"]
    
    # Calculate total pairs per attack type
    pairs_per_attack = genuine_count + impostor_count
    
    genuine_data = {}
    impostor_data = {}
    
    # Split data into genuine and impostor pairs for each attack
    for i, attack in enumerate(attacks):
        start_idx = i * pairs_per_attack
        mid_idx = start_idx + genuine_count
        end_idx = start_idx + pairs_per_attack
        
        if i < 7:  # For the attack types
            genuine_values = values[start_idx:mid_idx]
            impostor_values = values[mid_idx:end_idx]
        else:  # For the baseline
            genuine_values = values[7 * pairs_per_attack : 7 * pairs_per_attack + genuine_count]
            impostor_values = values[7 * pairs_per_attack + genuine_count : 8 * pairs_per_attack]
            
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
        genuine_current = genuine_avg[attack]
        genuine_deg = ((genuine_current - genuine_baseline) / genuine_baseline) * 100
        genuine_degradation[attack] = genuine_deg
        
        impostor_current = impostor_avg[attack]
        impostor_deg = ((impostor_current - impostor_baseline) / impostor_baseline) * 100
        impostor_degradation[attack] = impostor_deg
    
    # Create directory for graphs if it doesn't exist
    os.makedirs("Graphs", exist_ok=True)
    
    # Create visualizations with appropriate interpretation
    plt.figure(figsize=(10, 6))
    attacks_without_baseline = attacks[:-1]
    genuine_deg_values = [genuine_degradation[a] for a in attacks_without_baseline]
    bars = plt.bar(attacks_without_baseline, genuine_deg_values)
    
    for i, bar in enumerate(bars):
        if genuine_deg_values[i] < 0:  
            bar.set_color('green')
        else:  
            bar.set_color('red')
    
    plt.title(f"Similarity Score Change for {model_name} - Genuine Pairs\n(Negative % is better robustness)")
    plt.xlabel("Attack Type")
    plt.ylabel("% Change from Baseline")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Graphs/{model_name}_genuine_l2_change.png", dpi=300)
    
    plt.figure(figsize=(10, 6))
    impostor_deg_values = [impostor_degradation[a] for a in attacks_without_baseline]
    bars = plt.bar(attacks_without_baseline, impostor_deg_values)
    
    for i, bar in enumerate(bars):
        if impostor_deg_values[i] > 0:  
            bar.set_color('green')
        else:  
            bar.set_color('red')
    
    plt.title(f"Similarity Score Change for {model_name} - Impostor Pairs\n(Positive % is better robustness)")
    plt.xlabel("Attack Type")
    plt.ylabel("% Change from Baseline")
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Graphs/{model_name}_impostor_l2_change.png", dpi=300)
    
    print(f"\nResults for {model_name}:")
    
    #INVERT THIS IF SIMILARITY SCORE
    print("\nGenuine Pairs - L2 Distance Change (negative % is better):")
    for attack, value in genuine_degradation.items():
        status = "BETTER" if value < 0 else "WORSE"
        print(f"{attack}: {value:.2f}% ({status})")
    
    print("\nImpostor Pairs - L2 Distance Change (positive % is better):")
    for attack, value in impostor_degradation.items():
        status = "BETTER" if value > 0 else "WORSE"
        print(f"{attack}: {value:.2f}% ({status})")
    
    return genuine_degradation, impostor_degradation


# Run the analysis
if __name__ == "__main__":
    model_name = "AdaFace"  # Change this for each model
    file_path = "./Verification_metric/SimScore_values_Ada.txt" 
    
    analyze_model_l2_degradation(file_path, model_name, genuine_count=1680, impostor_count=5748)
    
    print(f"Analysis complete for {model_name}. Check the generated images.")