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
        'durations': np.array([57.2]),
        'noise_realizations': np.array([150]),
        'python_files': np.array([15])
    },
    {
        'name': 'GW231226',
        'durations': np.array([72.4]),
        'noise_realizations': np.array([150]), # Assuming the same configuration as GW150914
        'python_files': np.array([15])       # Assuming the same configuration as GW150914
    }
]

# Template filenames
file = 'python_file_tem.py' 
slurm_file = 'run_tem.slurm'

# --- Initialize strings and lists for script generation ---
sbatch_file_data = ""
clear_commands = [] # Use a list to collect clean commands to avoid duplicates

# --- Main loop to iterate through all events ---
for event in events_config:
    evenetname = event['name']
    durationlist = event['durations']
    number_of_noise_realization_list = event['noise_realizations']
    python_file_number_list = event['python_files']
    
    print(f"--- Processing Event: {evenetname} ---")

    # Iterate through different duration settings
    for duration_index in range(len(durationlist)):
        duration = durationlist[duration_index]
        number_of_noise_realization = number_of_noise_realization_list[duration_index]
        python_file_number = python_file_number_list[duration_index]
        
        # Iterate through different likelihood settings
        for likelihood_index in [0, 1]:
            likelihood_string = likelihoodlist[likelihood_index]
            
            # In this example, notch_index is fixed to 1
            notch_index = 0
            notch_string = notchlist[notch_index]
            
            # --- 1. Create directory and generate clean script command ---
            folder_name = f'python_files_{notch_string}_duration{duration:.0f}_{evenetname}'
            if not os.path.exists("../" + folder_name):
                os.mkdir("../" + folder_name)

            clear_line = f"rm -rf {folder_name}\n"
            if clear_line not in clear_commands:
                clear_commands.append(clear_line)
                
            # --- 2. Loop to generate Python files ---
            for n in np.arange(0, python_file_number, 1):
                # Calculate the range of noise realizations for each Python file to process
                lhi = n * (number_of_noise_realization // python_file_number)
                lhf = (n + 1) * (number_of_noise_realization // python_file_number)
                
                str1 = f'lhi = {lhi}'
                str2 = f'lhf = {lhf}'
                
                # Only the first file (n=0) prints the header
                str3 = 'whether_print_header = 0' if n == 0 else 'whether_print_header = 1'
                    
                str4 = f'nduration = {duration_index+1}'
                str5 = f'likelihood_index = {likelihood_index}'
                str6 = f'notch_index = {notch_index}'
                str7 = f"eventname = '{evenetname}'"
                
                file_data = ""
                # Read the template and replace placeholders
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.replace('lhi = tem', str1)
                        line = line.replace('lhf = tem', str2)
                        line = line.replace('whether_print_header = tem', str3)
                        line = line.replace('nduration = tem', str4)
                        line = line.replace('likelihood_index = tem', str5)
                        line = line.replace('notch_index = tem', str6)
                        line = line.replace('eventname = tem', str7)
                        file_data += line
                
                # Write to the new Python file
                output_py_filename = f'{notch_string}_{likelihood_string}_{n}.py'
                # print(f"Generating: {folder_name}/{output_py_filename} ({str1}, {str2})") # Can be uncommented for more detailed output
                with open(f"../{folder_name}/{output_py_filename}", "w", encoding="utf-8") as f:
                    f.write(file_data)
#
#             # --- 3. Generate SLURM script for the current set of Python files ---
#             slurm_file_data = ""
#             with open(slurm_file, "r", encoding="utf-8") as f:
#                 for line in f:
#                     # Replace the SLURM job array range
#                     if '#SBATCH --array=tem' in line:
#                         # The array range is from 0 to python_file_number - 1
#                         line = line.replace('#SBATCH --array=tem', f'#SBATCH --array=0-{python_file_number - 1}')
#                     # Replace the name of the Python script to be executed
#                     if 'srun python tem${SLURM_ARRAY_TASK_ID}.py' in line:
#                         # SLURM will replace ${SLURM_ARRAY_TASK_ID} with the task ID
#                         python_command = f'srun python {notch_string}_{likelihood_string}_${{SLURM_ARRAY_TASK_ID}}.py'
#                         line = line.replace('srun python tem${SLURM_ARRAY_TASK_ID}.py', python_command)
#                     slurm_file_data += line
#
#             # Write to the new SLURM file
#             output_slurm_filename = f'{notch_string}_{likelihood_string}.slurm'
#             print(f"Generated SLURM script: {folder_name}/{output_slurm_filename}")
#             with open(f"../{folder_name}/{output_slurm_filename}", "w", encoding="utf-8") as f:
#                 f.write(slurm_file_data)
#
#             # --- 4. Add the sbatch command to the main run script ---
#             linebefore = f'cd {folder_name}\n'
#             lineafter = 'cd ..\n'
#             sbatch_command = f'sbatch {output_slurm_filename}\n'
#             sbatch_file_data += linebefore + sbatch_command + lineafter
#             print("-" * 30)
#
# # --- 5. Finally, write the main run script and the clean script ---
# print("\nGenerating final control scripts: run.sh and clear_python_files.sh")
# with open('../run.sh', "w", encoding="utf-8") as f:
#     f.write(sbatch_file_data)
#
# with open('../clear_python_files.sh', "w", encoding="utf-8") as f:
#     f.write("".join(clear_commands)) # Join the commands from the list and write to the file
#
# print("\nAll files generated successfully for all events!")
