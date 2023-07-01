import multiprocessing
import numpy as np
import nn

def run_main(arg):
    nn.main(arg)

if __name__ == '__main__':

    arguments = np.array([
    'validation_exp(\'Estar\', \'Ti33_500a\')',
    'validation_exp_cross2(\'Estar\', 1, \'Ti33_500a\', \'Ti33_500a\')'
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