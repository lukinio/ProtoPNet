#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
import re


def plot_param(log_file_path: str, param_name: str):
    train_values = []
    test_values = []
    with open(log_file_path) as file:
        parse_section = False
        epoch = -1
        is_test = False
        is_push = False
        for line in file:
            mt = re.match(r'^(\w+):\s+(\d+)', line)
            if mt:
                parse_section = mt.group(1) == 'epoch'
                if parse_section:
                    epoch = int(mt.group(2))
                    is_push = False
                continue
            if parse_section and line.startswith('\t'):
                mt = re.match(r'^\s+(\w+)\s*$', line)
                if mt:
                    is_test = mt.group(1) == 'test'
                    is_push |= mt.group(1) == 'push'
                    continue
                mt = re.match(r'^\s+([^:]+):\s+([\d.]+)', line)
                if mt:
                    if mt.group(1) == param_name and not is_push:
                        if is_test:
                            test_values.append(float(mt.group(2)))
                        else:
                            train_values.append(float(mt.group(2)))
    plt.plot(train_values, label='Train')
    plt.plot(test_values, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel(param_name.capitalize())
    plt.legend()
    plt.show()


log_file = sys.argv[1]
param = sys.argv[2]
plot_param(log_file, param)
