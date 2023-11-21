import os

folder_path = 'C:\School\dataset\Angle_close_mediumHighLight_annotate\obj_train_data'  # Replace with the path to your folder
folder_path = folder_path.replace('\\','/')
# Ensure the specified path exists
if os.path.exists(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Construct the new filename with the added prefix
        new_filename = 'cmhl_' + filename

        # Create the full paths for the old and new filenames
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_filepath, new_filepath)

    print("Files renamed successfully.")
else:
    print("The specified folder path does not exist.")
