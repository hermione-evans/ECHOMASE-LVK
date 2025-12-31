import numpy as np
from glob import glob
import os
## this file is our template

import numpy as np

injectlist = ['notch', 'inject']
likelihoodlist = ['oldlikelihood','newlikelihood']
durationlist = np.array([48.6, 36.5, 24.3, 12.1, 4.9])

file = 'python_file_tem.py' 
pbs_file = 'run_tem.pbs'


test = 0
index_file_data = ""
benchmarkindex = 0
run_file_data = ""
python_file_number_list = np.arange(0,50,1)
number_of_noise_realization = 150
sbatch_file_data = ""
clear_file_data =""

# '_duration={:.0f}'.format(duration)+
for duration_index in np.arange(0,5,1):
    duration = durationlist[duration_index]
    for likelihood_index in np.arange(0,2,1):
        likelihood_string = likelihoodlist[likelihood_index]
        for inject_index in np.arange(0,2,1):
            inject_string = injectlist[inject_index]
            folder_name = 'python_files_'+inject_string+'_duration_{:.0f}'.format(duration)
            if not os.path.exists("../"+folder_name):
                os.mkdir("../"+folder_name)
            line = "rm -rf "+folder_name+'\n'
            clear_file_data = clear_file_data + line
            for n in python_file_number_list:
                # Here the n is used to label the number of the python file
                str1 = 'lhi = %d'%(n*(number_of_noise_realization//len(python_file_number_list)))
                str2 = 'lhf = %d'%((n+1)*(number_of_noise_realization//len(python_file_number_list)))
                # str3 = 'whether_print_header = 1'
                if n == 0:
                    str3 = 'whether_print_header = 0'
                else:
                    str3 = 'whether_print_header = 1'
                str4 = 'nduration = %d'%duration_index
                str5 = 'likelihood_index = %d'%likelihood_index
                str6 = 'inject_index = %d'%inject_index
                file_data = ""
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        if 'lhi = tem' in line:
                            line = line.replace('lhi = tem',str1)
                        if 'lhf = tem' in line:
                            line = line.replace('lhf = tem',str2)
                        if 'whether_print_header = tem' in line:
                            line = line.replace('whether_print_header = tem',str3)
                        if 'nduration = tem' in line:
                            line = line.replace('nduration = tem',str4)
                        if 'likelihood_index = tem' in line:
                            line = line.replace('likelihood_index = tem',str5)
                        if 'inject_index = tem' in line:
                            line = line.replace('inject_index = tem',str6)
                        
                        file_data += line
                test = 1
                print(n,str1,str2,str3,str4,str5,str6,inject_string+'_'+likelihood_string+'_%d.py'%n)
                with open("../"+folder_name+'/'+inject_string+'_'+likelihood_string+'_%d.py'%n,"w",encoding="utf-8") as f:
                    f.write(file_data)
                    
            pbs_file_data = ""
            with open(pbs_file, "r", encoding="utf-8") as f:
                for line in f:
                    if '#PBS -t tem' in line:
                        line = line.replace('tem', f'{python_file_number_list[0]}-{python_file_number_list[-1]}')
                    if '#PBS -N tem' in line:
                        line = line.replace('tem', f'run_{inject_string}_{likelihood_string}_{int(duration)}')
                    # if 'srun python tem${SLURM_ARRAY_TASK_ID}.py' in line:
                    #     line = line.replace('srun python tem${SLURM_ARRAY_TASK_ID}.py','srun python '+inject_string+'_'+likelihood_string+'_${SLURM_ARRAY_TASK_ID}.py')
                    if 'python tem${IDX}.py' in line:
                        line = line.replace('tem', f'{inject_string}_{likelihood_string}_')
                    pbs_file_data = pbs_file_data + line
            with open("../"+folder_name+'/'+inject_string+'_'+likelihood_string+'.pbs',"w",encoding="utf-8") as f:
                f.write(pbs_file_data)
            
            linebefore = 'cd '+folder_name+'\n'
            lineafter = 'cd .. \n' + 'sleep 180 \n'
            sbatch_file_data = sbatch_file_data + linebefore + 'qsub '+inject_string+'_'+likelihood_string+'.pbs\n' + lineafter
with open('../run.sh',"w",encoding="utf-8") as f:
    f.write(sbatch_file_data)
with open('../clear_python_files.sh',"w",encoding="utf-8") as f:
    f.write(clear_file_data)
