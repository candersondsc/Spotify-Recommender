import os
import subprocess
import sys

#Create directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    print("----- Spotify Music Recommender System Setup -----")

    #Create project structure / directories
    print("\nCreating additional project directories...")
    create_directory("data/raw")
    create_directory("data/processed")
    create_directory("models")
    create_directory("streamlit/static")
    print("Directories created successfully.")

    #Create virtual environment if it does not already exist
    print("\nCreating virtual environment...")
    venv_name = "venv"
    
    if not os.path.exists(venv_name):
        try:
            subprocess.check_call([sys.executable, "-m", "venv", venv_name])
            print(f"Virtual environment {venv_name} created successfully.")
        except subprocess.CalledProcessError:
            print(f"Error creating virtual environment. Please try manually with 'python3 -m venv {venv_name}'")
            return
    else:
        print(f"Virtual environment '{venv_name}' already exists.")    
    
    #Prompt for dataset files
    print("\nYou need to place your Spotify datasets in the 'data/raw' folder.")
    print("Rename them to 'spotify_dataset1.csv' and 'spotify_dataset2.csv'")
    input("\nPress Enter once you've added the dataset files...")
    
    #Check if files exist
    if not (os.path.exists("data/raw/spotify_dataset1.csv") and os.path.exists("data/raw/spotify_dataset2.csv")):
        print("\nWarning: Dataset files not found.")
        print("Add them before running the pipeline.")
    else:
        print("\nDataset files found.")
    
    print("\n----- Setup Complete -----")
    print("Next steps:")
    print("1. Select the virtual environment as your Python interpreter in VSCode:")
    print("   - Select from the pop-up in the bottom right or")
    print("   - Select the interpreter with 'venv' in its path manually in the status bar")
    print("2. Install requirements in the terminal with:")
    print("   pip install -r requirements.txt")
    print("3. Run the pipeline with: python pipeline.py")

if __name__ == "__main__":
    main()