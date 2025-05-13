INSTALL FACENET WITH PIP: !pip install facenet-pytorch
DOWNLOAD THE LFW DATASET @ KAGGLE https://www.kaggle.com/datasets/jessicali9530/lfw-dataset. 
Right now the code in the test files is looking for a directory called lfw_funneled that contains directories to images. Adjust if needed. 


Run files in testing for experimental results. Clean up can be commented out under the "evaluate_attack" function
Adjust amount of pairs prepared in "prepair_pairs" functions, set to 0 for full dataset evaluation. 
Files under the utils were used to generate the graphs and examples in the dissertaion. 
- cleanUp.py to clean every adversarial example in dataset. MAKE SURE TO DO THIS BEFORE RUNNING IF YOU HAVE CLEANING DISABLED. 
- Calculation.py to display L2/Sim score degradation.(Already done under graphs folder)
- generateExample.py to visualize the perturbation(one example provided called heatmap_result.png)
Verification_metric folder is used to store the L2/Sim score value after runs
All experiments were conducted using PyTorch 2.6.0+cu118 on a single NVIDIA RTX 3060 Laptop GPU with 6GB VRAM. The implementation was done in Python 3.12.0. All done on a Windows 11 OS. 
Weights are provided under Model/Weights and sources are cited in the disertation. 
