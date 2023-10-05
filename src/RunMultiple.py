import multiprocessing
import numpy as np
import nn

def run_main(arg):
    nn.main(arg)

if __name__ == '__main__':

    arguments = np.array([
        "validation_exp_cross2('Estar', 1, 'B3090', 'B3090')",
        "validation_exp_cross2('sigma_y', 1, 'B3090', 'B3090')",
        "validation_exp_cross2('Estar', 20, 'B3090', 'B3090')",
        "validation_exp_cross2('sigma_y', 20, 'B3090', 'B3090')"
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