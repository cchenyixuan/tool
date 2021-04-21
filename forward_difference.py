import numpy as np
import pandas as pd
import csv
from sys import argv
import time
import os

try:
    print(argv)
    home, file_path, file_number, content, cardiac_cycle, output_dir, name_rule = argv
    file_number = int(file_number)
    cardiac_cycle = float(cardiac_cycle)
    print(file_path)
    print(file_number)
    print(content)
    print(cardiac_cycle)
    time.sleep(5)
except:
    print('not enough args')
    file_path = input('file name:')
    file_number = input('file number:')
    file_number = int(file_number)
    content = input('function:')
    cardiac_cycle = input('cardiac cycle:')
    cardiac_cycle = float(cardiac_cycle)
    output_dir = "./"
    print('start in 5 seconds...')
    time.sleep(5)


def preprocess(dataframes):
    data = []
    for dataframe in dataframes:
        points = []
        cord = list(dataframe)
        for j in range(len(cord) // 3):  # dataframe_column
            points.append((cord[j * 3:j * 3 + 3]))

        data.append(points)

    return data


def update_movement(data):
    array = np.array
    norm = np.linalg.norm

    updated_data = []
    for i in range(len(data[0])):
        checkingpoint = []

        max_movement = 0
        for j in range(len(data)):
            checkingpoint.append(data[j][i])
        for m in range(len(checkingpoint) - 1):
            for n in range(m + 1, len(checkingpoint)):
                distance = norm(array(checkingpoint[m]) - array(checkingpoint[n]))
                if distance > max_movement:
                    max_movement = distance
        xyzm = data[0][i] + [max_movement]
        updated_data.append(xyzm)

    return updated_data


def update_velocity_accelerate(data):
    array = np.array
    norm = np.linalg.norm

    velocity_data = []
    accelerate_data = []
    for i in range(len(data[0])):
        checkingpoint = []
        max_velocity = 0
        max_accelerate = 0
        for j in range(len(data)):
            checkingpoint.append(data[j][i])
        for m in range(len(checkingpoint) - 2):
            point1 = array(checkingpoint[m])
            point2 = array(checkingpoint[m + 1])
            point3 = array(checkingpoint[m + 2])
            velocity1 = (point2 - point1) / cardiac_cycle
            velocity2 = (point3 - point2) / cardiac_cycle
            accelerate = norm((velocity2 - velocity1) / cardiac_cycle)  # 前進差分
            velocity1 = norm(velocity1)
            if velocity1 > max_velocity:
                max_velocity = velocity1
            if accelerate > max_accelerate:
                max_accelerate = accelerate
        xyzv = data[0][i] + [max_velocity]
        xyza = data[0][i] + [max_accelerate]
        velocity_data.append(xyzv)
        accelerate_data.append(xyza)

    return (velocity_data, accelerate_data)


def instant_velocity(data):  # 2timesteps
    array = np.array
    norm = np.linalg.norm

    velocity_data = []
    for i in range(len(data)):
        points_velo = []
        for j in range(len(data[i][0])):
            v1 = array(data[i][0][j])
            v2 = array(data[i][1][j])
            velo = norm(v2 - v1)
            xyzv = list(v1) + [velo]
            points_velo.append(xyzv)
        velocity_data.append(points_velo)

    return velocity_data


def instant_accelerate(data):
    array = np.array
    norm = np.linalg.norm

    accelerate_data = []
    for i in range(len(data)):  # faces
        points_acce = []
        for j in range(len(data[i][0])):  # points
            v1 = array(data[i][0][j])  # [facei][time0][pointj]
            v2 = array(data[i][1][j])
            v3 = array(data[i][2][j])
            acce = norm(v3 - v2 - v2 + v1)
            xyza = list(v1) + [acce]
            points_acce.append(xyza)
        accelerate_data.append(points_acce)

    return accelerate_data


def update_min_movement(data):
    array = np.array
    norm = np.linalg.norm

    updated_data = []
    for i in range(len(data[0])):
        checkingpoint = []

        min_movement = 100000000
        for j in range(len(data)):
            checkingpoint.append(data[j][i])
        for m in range(len(checkingpoint) - 1):

            distance = norm(array(checkingpoint[m]) - array(checkingpoint[m + 1]))
            if distance < min_movement:
                min_movement = distance
        xyzm = data[0][i] + [min_movement]
        updated_data.append(xyzm)

    return updated_data  # in place of minvelo


def update_min_velocity(data):
    array = np.array
    norm = np.linalg.norm

    updated_data = []
    for i in range(len(data[0])):
        checkingpoint = []

        min_velocity = 100000000
        for j in range(len(data)):
            checkingpoint.append(data[j][i])
        for m in range(len(checkingpoint) - 1):

            velocity = norm((array(checkingpoint[m]) - array(checkingpoint[m + 1])) / cardiac_cycle)
            if velocity < min_velocity:
                min_velocity = velocity
        xyzm = data[0][i] + [min_velocity]
        updated_data.append(xyzm)

    return updated_data


def update_min_acceleration(data):
    array = np.array
    norm = np.linalg.norm

    updated_data = []
    for i in range(len(data[0])):
        checkingpoint = []

        min_acceleration = 100000000
        for j in range(len(data)):
            checkingpoint.append(data[j][i])
        for m in range(len(checkingpoint) - 2):

            velocity1 = (array(checkingpoint[m]) - array(checkingpoint[m + 1])) / cardiac_cycle
            velocity2 = (array(checkingpoint[m + 1]) - array(checkingpoint[m + 2])) / cardiac_cycle
            acceleration = norm((velocity2 - velocity1) / cardiac_cycle)
            if acceleration < min_acceleration:
                min_acceleration = acceleration
        xyzm = data[0][i] + [min_acceleration]
        updated_data.append(xyzm)

    return updated_data


def update_median_average_standard_movement(data):
    array = np.array
    norm = np.linalg.norm
    median = np.median
    average = np.average
    std = np.std

    updated_median_data = []
    updated_average_data = []
    updated_standard_data = []

    for i in range(len(data[0])):
        checkpoint_list = []

        movement_list = []
        for j in range(len(data)):
            checkpoint_list.append(data[j][i])
        for m in range(len(checkpoint_list) - 1):
            for n in range(m + 1, len(checkpoint_list)):
                distance = norm(array(checkpoint_list[m]) - array(checkpoint_list[n]))
                movement_list.append(distance)
        median_movement = median(movement_list)
        average_movement = average(movement_list)
        standard_movement = std(movement_list)

        xyzm = data[0][i] + [median_movement]
        xyza = data[0][i] + [average_movement]
        xyzs = data[0][i] + [standard_movement]

        updated_median_data.append(xyzm)
        updated_average_data.append(xyza)
        updated_standard_data.append(xyzs)

    return (updated_median_data, updated_average_data, updated_standard_data)


def update_median_average_standard_velocity_accelerate(data):  # 2021/02/10 works
    array = np.array
    norm = np.linalg.norm
    median = np.median
    average = np.average
    std = np.std

    median_velocity_data = []
    average_velocity_data = []
    standard_velocity_data = []

    median_accelerate_data = []
    average_accelerate_data = []
    standard_accelerate_data = []

    for i in range(len(data[0])):
        checkingpoint = []
        velocity_data = []
        accelerate_data = []

        for j in range(len(data)):
            checkingpoint.append(data[j][i])
        for m in range(len(checkingpoint) - 2):
            point1 = array(checkingpoint[m])
            point2 = array(checkingpoint[m + 1])
            point3 = array(checkingpoint[m + 2])
            velocity1 = (point2 - point1) / cardiac_cycle
            velocity2 = (point3 - point2) / cardiac_cycle
            accelerate = norm((velocity2 - velocity1) / cardiac_cycle)  # 前進差分
            velocity1 = norm(velocity1)
            velocity_data.append(velocity1)
            accelerate_data.append(accelerate)

        median_velocity = median(velocity_data)
        average_velocity = average(velocity_data)
        standard_velocity = std(velocity_data)

        median_accelerate = median(accelerate_data)
        average_accelerate = average(accelerate_data)
        standard_accelerate = std(accelerate_data)

        xyzvm = data[0][i] + [median_velocity]
        xyzva = data[0][i] + [average_velocity]
        xyzvs = data[0][i] + [standard_velocity]

        xyzam = data[0][i] + [median_accelerate]
        xyzaa = data[0][i] + [average_accelerate]
        xyzas = data[0][i] + [standard_accelerate]

        median_velocity_data.append(xyzvm)
        average_velocity_data.append(xyzva)
        standard_velocity_data.append(xyzvs)

        median_accelerate_data.append(xyzam)
        average_accelerate_data.append(xyzaa)
        standard_accelerate_data.append(xyzas)

    return (median_velocity_data, average_velocity_data, standard_velocity_data, median_accelerate_data,
            average_accelerate_data, standard_accelerate_data)


def save_file(file_name, calculated_data):
    # forward_difference.py//{}.csv
    with open(output_dir + '{}.csv'.format(file_name), 'w', newline='') as file:
        file_writer = csv.writer(file)
        file_writer.writerows(calculated_data)
        file.close()


def analyze(content, timepoints=100):
    columns = {1: 'max movement',
               2: 'max velocity and acceleration',
               3: 'min movement',
               4: 'min velocity',
               5: 'min acceleration',
               6: 'median average standard movement',
               7: 'median average standard velocity and acceleration'}
    content = str(columns[int(content)])
    functions = {'max movement': update_movement,
                 'max velocity and acceleration': update_velocity_accelerate,
                 'min movement': update_min_movement,
                 'min velocity': update_min_velocity,
                 'min acceleration': update_min_acceleration,
                 'median average standard movement': update_median_average_standard_movement,
                 'median average standard velocity and acceleration': update_median_average_standard_velocity_accelerate}

    calculated_data0 = []
    calculated_data1 = []
    calculated_data2 = []
    calculated_data3 = []
    calculated_data4 = []
    calculated_data5 = []

    calculated_data = {0: calculated_data0,
                       1: calculated_data1,
                       2: calculated_data2,
                       3: calculated_data3,
                       4: calculated_data4,
                       5: calculated_data5}

    file_name = {'max movement': ('移動最大値',),
                 'max velocity and acceleration': ('速度最大値', '加速度最大値'),
                 'min movement': ('移動最小値',),
                 'min velocity': ('速度最小値',),
                 'min acceleration': ('加速度最小値',),
                 'median average standard movement': ('移動中央値', '移動平均値', '移動標準偏差'),
                 'median average standard velocity and acceleration': (
                     '速度中央値', '速度平均値', '速度標準偏差', '加速度中央値', '加速度平均値', '加速度標準偏差')}
    activated_function = file_name[content]
    read_csv = pd.read_csv
    for i in range(file_number):
        print(i)
        dataframes = []
        for timepoint in range(timepoints):
            data_temp = read_csv(file_path + "/" + csvs[i])
            data_temp = data_temp.iloc[timepoint, 2:]
            dataframes.append(data_temp)
        data = preprocess(dataframes)
        analyzed_data = functions[content](data)

        if len(activated_function) == 1:
            calculated_data0 += analyzed_data
        else:
            for datai in range(len(activated_function)):
                calculated_data[datai] += analyzed_data[datai]

    for i in range(len(activated_function)):
        if len(calculated_data[i]) > 0:
            save_file(activated_function[i], calculated_data[i])


import traceback
import re

try:
    csvs = []
    files = os.listdir(file_path)
    for i in range(1000):
        item = name_rule.format(i)
        if item in files:
            csvs.append(item)
    if len(csvs) != len(files):
        find_name = re.compile(r"([0-9]+).csv", re.S)
        print("File number not matched! Using flexible name-rule. Please check name-rule:'{}'".format(name_rule))
        file_dict = {}
        for item in files:
            index = re.findall(find_name, item)[0]
            file_dict[index] = item
        opt_list = list(file_dict.keys())
        opt_list.sort()
        for item in opt_list:
            csvs.append(file_dict[item])
            
    print('csv file analyzing...')
    for item in csvs:
        print(item)
    data_temp = pd.read_csv(file_path + "/" + csvs[0])
    timepoints = data_temp.shape[0]
    print("{} files found.".format(len(csvs)))
    print('{} time-steps found in given data'.format(timepoints))

    analyze(content, timepoints)
except:
    traceback.print_exc()
    input()
