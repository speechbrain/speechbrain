import sys
import json

    
def write_json(json_file_name, data):
    with open(json_file_name, mode='w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2, separators=(',', ': '))

def generate_json(folder_path, split):
    yaml_file = read_file(split_folder + "/" + split + ".yaml")
    translations_file = read_file(split_folder + "/" + split + ".fra")

    assert len(yaml_file) == len(translations_file)

    output_json = dict()
    for i in range(len(yaml_file)):
        content = yaml_file[i]
        utt_id = content.split(", wav: ")[1].split("}")[0]
        output_json[utt_id] = dict()
        output_json[utt_id]["path"] = "data_proc/" + folder_path.replace("/txt","/wav") + "/" + utt_id + ".wav"
        output_json[utt_id]["trans"] = translations_file[i]
        output_json[utt_id]["duration"] = content.split("{duration: ")[1].split(",")[0]
    
    return output_json
    
def read_file(f_path):
    return [line for line in open(f_path)]

if __name__ == '__main__':
    root_input_folder = sys.argv[1]
    for split in ["train", "valid", "test"]:
        split_folder = "/".join([root_input_folder, split, "txt"]) 
        
        output_json = generate_json(split_folder, split)

        write_json(sys.argv[2] + "/" + split + ".json", output_json)
