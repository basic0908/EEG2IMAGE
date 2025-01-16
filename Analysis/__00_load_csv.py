import os
import pandas as pd
import numpy as np



def data_organizer():
    cwd = os.getcwd()
    print(f'cwd : {cwd}')
    folder_path = os.path.join(cwd, 'data\\viewRec')
    output_folder = os.path.join(cwd, 'data\\Vie2Image')

    print(f'folder_path : {os.path.exists(folder_path)}')
    print(f'output_folder exists : {os.path.exists(output_folder)}')
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            
            df = pd.read_csv(file_path, header=0)
            
            subject_name = file_name.split('-')[0]
            
            for i in range(0, len(df), 6000):
                chunk = df.iloc[i:i+6000]

                img_filename = chunk.iloc[0,3]

                img_dataset = img_filename.split('\\')[0]
                img_category = img_filename.split('\\')[1]

                if img_dataset == 'Colors':
                    img_dataset = 'Color'
                elif img_dataset == 'MNIST':
                    img_dataset = 'Digit'
                elif img_dataset == 'Chars74K':
                    img_dataset = 'Char'
                else:
                    img_dataset = 'Image'
                
                output_file = os.path.join(output_folder, img_dataset, f"{subject_name}_{img_category}.csv")
                chunk.iloc[:, :3].to_csv(output_file, index=False)
                print(f"Successfully saved at {output_file}")

def load_alphabet(folder_path):
    # Initialize X and Y
    X = np.zeros((220, 2, 6000))  # Shape (220, 2, 6000)
    Y = np.zeros(220, dtype=int)  # Shape (220,)
    
    # Loop through all CSV files in the folder
    for ctr, file in enumerate(sorted(os.listdir(folder_path))):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=0)
            
            # Extract first two columns and reshape for X
            X[ctr] = df.iloc[:, :2].values.T[:, :6000]
            
            # Determine the class for Y based on the file name
            cls = file.split('_')[1]  # Assuming file names follow a specific format
            if cls[0] == 'A':
                Y[ctr] = 0
            elif cls[0] == 'C':
                Y[ctr] = 1
            elif cls[0] == 'F':
                Y[ctr] = 2
            elif cls[0] == 'H':
                Y[ctr] = 3
            elif cls[0] == 'J':
                Y[ctr] = 4
            elif cls[0] == 'M':
                Y[ctr] = 5
            elif cls[0] == 'P':
                Y[ctr] = 6
            elif cls[0] == 'S':
                Y[ctr] = 7
            elif cls[0] == 'T':
                Y[ctr] = 8
            elif cls[0] == 'Y':
                Y[ctr] = 9

    return X, Y

def load_object(folder_path):
    """
    Load object data from CSV files in a folder.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        tuple: X (numpy.ndarray), Y (numpy.ndarray)
    """
    # Initialize X and Y
    X = np.zeros((220, 2, 6000))  # Shape (220, 2, 6000)
    Y = np.zeros(220, dtype=int)  # Shape (220,)
    
    # Object class mapping
    object_classes = {
        "Apple": 0, "Car": 1, "Dog": 2, "Gold": 3, "Mobile": 4,
        "Rose": 5, "Scooter": 6, "Tiger": 7, "Wallet": 8, "Watch": 9
    }
    
    # Loop through all CSV files in the folder
    for ctr, file in enumerate(sorted(os.listdir(folder_path))):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=0)  # Skip the header
            
            # Extract first two columns and reshape for X
            X[ctr] = df.iloc[:, :2].values.T[:, :6000]
            
            # Determine the class for Y based on the file name
            try:
                cls = file.split('_')[-1].split('.')[0]  # Extract class and remove extension
                Y[ctr] = object_classes[cls.strip()]  # Map object name to label
            except KeyError:
                raise ValueError(f"Invalid object class in file name: {file}")
    
    return X, Y


def load_digit(folder_path):
    """
    Load digit data from CSV files in a folder.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        tuple: X (numpy.ndarray), Y (numpy.ndarray)
    """
    # Initialize X and Y
    X = np.zeros((220, 2, 6000))  # Shape (220, 2, 6000)
    Y = np.zeros(220, dtype=int)  # Shape (220,)
    
    # Loop through all CSV files in the folder
    for ctr, file in enumerate(sorted(os.listdir(folder_path))):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, header=0)  # Skip the header
            
            # Extract first two columns and reshape for X
            X[ctr] = df.iloc[:, :2].values.T[:, :6000]
            
            # Determine the class for Y based on the file name
            try:
                cls = file.split('_')[1].split('.')[0]  # Extract class and remove extension
                Y[ctr] = int(cls.strip())  # Parse numeric class directly
            except (IndexError, ValueError):
                print(f"Skipping invalid file: {file}")
                continue
    
    return X, Y
