import multiprocessing
import numpy as np
import nn

def run_main(arg):
    nn.main(arg)

if __name__ == '__main__':

    arguments = np.array([
        "validation_exp('Estar', 'Ti33_250a', typ='n')"
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

'''
        "validation_temperature('Estar', 0, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 1, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 2, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 3, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 4, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 5, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 6, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 8, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 10, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 20, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=25)",
        "validation_temperature('Estar', 0, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 1, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 2, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 3, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 4, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 5, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 6, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 8, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 10, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)",
        "validation_temperature('Estar', 20, ['Ti33_25a', 'Ti33_250a'], ['Ti33_25a', 'Ti33_250a'], typ='n', temp=250)"
'''