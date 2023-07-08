import multiprocessing
import numpy as np
import nn

def run_main(arg):
    nn.main(arg)

if __name__ == '__main__':

    arguments = np.array([
    'validation_exp(\'sigma_y\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 1, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 2, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 3, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 4, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 5, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 6, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 8, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 10, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp_cross2(\'sigma_y\', 20, \'Ti33_25a\', \'Ti33_25a\', fac=2.0)',
    'validation_exp(\'sigma_y\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 1, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 2, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 3, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 4, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 5, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 6, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 8, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 10, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp_cross2(\'sigma_y\', 20, \'Ti33_25a\', \'Ti33_25a\', fac=1)',
    'validation_exp(\'sigma_y\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 1, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 2, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 3, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 4, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 5, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 6, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 8, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 10, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)',
    'validation_exp_cross2(\'sigma_y\', 20, \'Ti33_25a\', \'Ti33_25a\', fac=0.9)'
    ])

    processes = []
    num_processes = len(arguments)
    for i in range(num_processes):        
        process = multiprocessing.Process(target=run_main, args=(arguments[i],))
        processes.append(process)

    for process in processes:
        process.start()
    for process in processes:
        process.join()

        