Install Facenet using pip: !pip install facenet-pytorch
Run files in testing for experimental results. Clean up can be commented out under the "evaluate_attack" function
Adjust amount of pairs prepared in "prepair_pairs" functions, set to 0 for full dataset evaluation. 
Files under the utils were used to generate the graphs and examples in the dissertaion. 
- cleanUp.py to clean every adversarial example in dataset. MAKE SURE TO DO THIS BEFORE RUNNING IF YOU HAVE CLEANING DISABLED. 
- Calculation.py to display L2/Sim score degradation.(Already done under graphs folder)
- generateExample.py to visualize the perturbation(one example provided called heatmap_result.png)
Verification_metric folder is used to store the L2/Sim score value after runs
