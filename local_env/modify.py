import os

def process_txt_file(file_path):
    # Read lines from the input file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process each line and modify the first instance of the character
    modified_lines = []
    for line in lines:
        if line.startswith('1 '):
            modified_lines.append('2 ' + line[2:])
        elif line.startswith('2 '):
            modified_lines.append('1 ' + line[2:])
        else:
            modified_lines.append(line)

    # Write the modified lines back to the original file
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

def process_folder(folder_path):
    # Get a list of all files in the specified folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Process each .txt file in the folder
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        process_txt_file(file_path)

if __name__ == "__main__":
    # Specify the path to the folder containing the .txt files
    folder_path = 'C:\School\dataset\\test'.replace('\\', '/')

    # Process the folder
    process_folder(folder_path)
