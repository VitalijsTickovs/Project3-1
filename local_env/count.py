import os

def process_txt_file(file_path, counts):
    # Read lines from the input file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Count occurrences of '0', '1', and '2' as the first character of each line
    for line in lines:
        first_char = line.strip()[0] if line.strip() else None
        if first_char in counts:
            counts[first_char] += 1

def process_folder(folder_path):
    # Get a list of all files in the specified folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Initialize counts
    counts = {'0': 0, '1': 0, '2': 0}

    # Process each .txt file in the folder
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        process_txt_file(file_path, counts)

    # Print the final counts
    print("Final Counts:")
    for char, count in counts.items():
        print(f"{char}: {count}")

if __name__ == "__main__":
    # Specify the path to the folder containing the .txt files
    folder_path = 'C:\School\dataset\FinalData\\annotation - Copy'.replace('\\', '/')

    # Process the folder
    process_folder(folder_path)
