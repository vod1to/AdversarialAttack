import os

def delete_adversarial_files(base_dir):
    # Walk through all directories and files
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if the file ends with adv.jpg
            if file.endswith("adv.jpg"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    count += 1
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    
    print(f"Total files deleted: {count}")

base_dir = './lfw_funneled'  

delete_adversarial_files(base_dir)