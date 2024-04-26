
import json
import os
from pathlib import Path
import copy

file_path_layer_times = "./layer_times.txt"

json_list_all_devices = {}
json_list_per_device = []
temp_dict = {}
with open(file_path_layer_times, "r") as file:
    # Loop through each line in the file
    
    for line in file:
        if line[:-1] == "zkn":
            # print("Here2")
            # print(json_list_per_device)
            json_list_all_devices[device_name] = copy.copy(json_list_per_device)
            # print(json_list_all_devices)
            json_list_per_device.clear()
            
        elif line[-2] == ":":
            # print("here")
            device_name = line[:-2]
            # print(device_name)
            json_list_all_devices[device_name] = []
        else:
            # print("Here1")
            temp_dict["forward"] = float(line[:-1])
            temp_dict["backward"] = 3*float(line[:-1])
            temp_dict["mem_required"] = [0.0, 0.0]
            json_list_per_device.append(copy.copy(temp_dict))
            # print(json_list_per_device)
    
    
# print(json_list_all_devices)
base_path = "./oobleck/"
for key in json_list_all_devices:
    if not os.path.exists(base_path + key):
        os.makedirs(base_path + key)
    path =  Path(base_path + key + "/mb1.json")
    # print(path)
    with path.open(mode="w") as f:
            json.dump(json_list_all_devices[key], f)
            f.flush()
 
    

