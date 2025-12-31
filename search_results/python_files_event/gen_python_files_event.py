import numpy as np
from glob import glob
import os

# code rebuild and add comments by Gemini 2.5 pro
# --- Configuration Parameters ---
likelihoodlist = ['oldlikelihood', 'newlikelihood']
notchlist = ['nonotch', 'notch']

# --- Event Configurations ---
# Package parameters for each event into a dictionary for easy iteration
events_config = [
    {
        'name': 'GW150914',
        'durations': np.array([114.4, 57.2]),
        'noise_realizations': np.array([1, 1]),
        'python_files': np.array([1, 1])
    },
    {
        'name': 'GW231226',
        'durations': np.array([144.9, 72.4]),
        'noise_realizations': np.array([1, 1]), # Assuming the same configuration as GW150914
        'python_files': np.array([1, 1])       # Assuming the same configuration as GW150914
    },
    {
        'name': 'GW250114',
        'durations': np.array([58.0]),
        'noise_realizations': np.array([1]), # Assuming the same configuration as GW150914
        'python_files': np.array([1])       # Assuming the same configuration as GW150914
    }
]

# Template filenames
file = 'python_file_tem.py'
# --- Main loop to iterate through all events ---
for event in events_config:
    evenetname = event['name']
    durationlist = event['durations']
    number_of_noise_realization_list = event['noise_realizations']
    python_file_number_list = event['python_files']

    print(f"--- Processing Event: {evenetname} ---")

    # Iterate through different duration settings
    for duration_index in range(len(durationlist)):  # Example indices for durations
        duration = durationlist[duration_index]
        number_of_noise_realization = number_of_noise_realization_list[duration_index]
        # python_file_number = python_file_number_list[duration_index]

        # Iterate through different likelihood settings
        for likelihood_index in [0,1]:
            likelihood_string = likelihoodlist[likelihood_index]

            # In this example, notch_index is fixed to 1
            notch_index = 1
            notch_string = notchlist[notch_index]

            # --- 1. Create directory and generate clean script command ---
            folder_name = f'python_files_event_all'
            if not os.path.exists("../" + folder_name):
                os.mkdir("../" + folder_name)

            # str3 = 'whether_print_header = 0' if duration_index == 0 else 'whether_print_header = 1'
            str3 = 'whether_print_header = 0'
            str4 = f'nduration = {duration_index}'
            str5 = f'likelihood_index = {likelihood_index}'
            str6 = f'notch_index = {notch_index}'
            str7 = f"eventname = '{evenetname}'"

            file_data = ""
            # Read the template and replace placeholders
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    # line = line.replace('lhi = tem', str1)
                    # line = line.replace('lhf = tem', str2)
                    line = line.replace('whether_print_header = tem', str3)
                    line = line.replace('nduration = tem', str4)
                    line = line.replace('likelihood_index = tem', str5)
                    line = line.replace('notch_index = tem', str6)
                    line = line.replace('eventname = tem', str7)
                    file_data += line

            # Write to the new Python file
            output_py_filename = f'event_{evenetname}_{likelihood_string}_{duration_index}.py'
            # print(f"Generating: {folder_name}/{output_py_filename} ({str1}, {str2})") # Can be uncommented for more detailed output
            with open(f"../{folder_name}/{output_py_filename}", "w", encoding="utf-8") as f:
                f.write(file_data)


file = 'python_file_tem_earlyer.py'
# --- Main loop to iterate through all events ---
for event in events_config:
    evenetname = event['name']
    durationlist = event['durations']
    number_of_noise_realization_list = event['noise_realizations']
    python_file_number_list = event['python_files']

    print(f"--- Processing Event: {evenetname} ---")

    # Iterate through different duration settings
    for duration_index in range(len(durationlist)):  # Example indices for durations
        duration = durationlist[duration_index]
        number_of_noise_realization = number_of_noise_realization_list[duration_index]
        # python_file_number = python_file_number_list[duration_index]

        # Iterate through different likelihood settings
        for likelihood_index in [0,1]:
            likelihood_string = likelihoodlist[likelihood_index]

            # In this example, notch_index is fixed to 1
            notch_index = 1
            notch_string = notchlist[notch_index]

            # --- 1. Create directory and generate clean script command ---
            folder_name = f'python_files_event_all'
            if not os.path.exists("../" + folder_name):
                os.mkdir("../" + folder_name)

            # str3 = 'whether_print_header = 0' if duration_index == 0 else 'whether_print_header = 1'
            str3 = 'whether_print_header = 0'
            str4 = f'nduration = {duration_index}'
            str5 = f'likelihood_index = {likelihood_index}'
            str6 = f'notch_index = {notch_index}'
            str7 = f"eventname = '{evenetname}'"

            file_data = ""
            # Read the template and replace placeholders
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    # line = line.replace('lhi = tem', str1)
                    # line = line.replace('lhf = tem', str2)
                    line = line.replace('whether_print_header = tem', str3)
                    line = line.replace('nduration = tem', str4)
                    line = line.replace('likelihood_index = tem', str5)
                    line = line.replace('notch_index = tem', str6)
                    line = line.replace('eventname = tem', str7)
                    file_data += line

            # Write to the new Python file
            output_py_filename = f'event_{evenetname}_{likelihood_string}_{duration_index}_earlyer.py'
            # print(f"Generating: {folder_name}/{output_py_filename} ({str1}, {str2})") # Can be uncommented for more detailed output
            with open(f"../{folder_name}/{output_py_filename}", "w", encoding="utf-8") as f:
                f.write(file_data)
