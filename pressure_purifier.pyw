import os
import numpy as np
import csv
import sys


case_name, point_dir, time_step, pressure_dir = sys.argv[1:]


def create_dir(directory):
    try:
        path = directory
        os.makedirs(path)
    except FileExistsError:
        print("dir '{}' already exists, skip...".format(path))
        pass


point_files = os.listdir(point_dir)
point_cloud = []
for item in point_files:
    with open(point_dir + "/" + item, "r") as f:
        csv_reader = csv.reader(f)
        for step, row in enumerate(csv_reader):
            if step == int(time_step) + 1:
                temp = row[2:]
                for i in range(len(temp) // 3):
                    point_cloud.append(temp[3 * i: 3 * i + 3])
                break
        f.close()
pressure_list = []
with open(pressure_dir, "r") as f:
    csv_reader = csv.reader(f)
    for step, row in enumerate(csv_reader):
        point = row[:3]
        pressure = row[3]
        normal = row[4:]
        pressure_list.append([point, pressure, normal])
    f.close()
data = []
array = np.array
norm = np.linalg.norm
for step, point in enumerate(point_cloud):
    temp = point.copy()
    point = array(point, dtype=np.float16)
    pair_point = None
    pair_pressure = None
    pair_normal = None
    min_distance = 1000000000000
    for target_point, pressure, normal in pressure_list:
        target_point = array(target_point, dtype=np.float16)
        distance = norm(point - target_point)
        if distance < min_distance:
            pair_point = target_point
            pair_pressure = pressure
            pair_normal = normal
            min_distance = distance
        if distance < 2:
            break
    temp.append(pair_pressure)
    temp += pair_normal
    data.append(temp)

    if step % 100 == 0:
        print(step)
    #    break

create_dir("./data/pressure/" + case_name + "_purified/")
with open("./data/pressure/" + case_name + "_purified/" + "{}.csv".format(int(time_step) / 100), "w", newline="") as f:
    csv_writer = csv.writer(f)
    for row in data:
        csv_writer.writerow(row)
    f.close()
