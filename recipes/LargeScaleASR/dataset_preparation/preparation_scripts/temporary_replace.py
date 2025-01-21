import sys
import shutil
import os

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def get_all_files(directory):

    file_paths = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths[file] = file_path
    
    return file_paths 

def find_and_replace_path(source_file, new_file, extracted_folder, output_folder):
    lines = open(
            source_file, "r", encoding="utf-8"
        ).readlines()
    header = lines[0]
    valid_corpus_lines = lines[1:]
    
    all_files = get_all_files(extracted_folder)

    cpt = 0
    for file, path in all_files.items():
        if ".wav" in file:
            cpt += 1
        if cpt == 10:
            break

    new_corpus = []
    cpt = 0
    new_file = open(new_file, 'w', encoding='utf-8')
    new_file.write(header)
    for line in valid_corpus_lines:
        ID, duration, start, wav, spk_id, sex, text = line.split('\n')[0].split(',')
        real_path = all_files[ID]
        if real_path is None:
            print(real_path)
            print("STOP")
            return

        new_path = os.path.join(output_folder, ID)
        if not os.path.isfile(new_path):
            shutil.copy(real_path, new_path)

        new_file.write(ID+","+duration+","+new_path+","+spk_id+","+sex+","+text+"\n")

    new_file.close()


find_and_replace_path(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])