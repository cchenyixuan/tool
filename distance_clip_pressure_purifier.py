import os
import csv
import numpy as np
import sys

# case_name, case_dir, cut_slice, time_step, pressure_dir = sys.argv[1:]
case_name = "Mie02"
case_dir = "./data/zio/Mie_case02_retry_20200516_after_sorting/"
cut_slice = 17
time_step = 33
pressure_dir = './data/pressure/Mie02/0.440257.csv'


def create_dir(directory):
    try:
        path = directory
        os.makedirs(path)
    except FileExistsError:
        print("dir '{}' already exists, skip...".format(path))
        pass


point_files = os.listdir(case_dir)
point_dist = {}

for step, item in enumerate(point_files):
    point_cloud = []
    with open(case_dir + "/" + item, "r") as f:
        csv_reader = csv.reader(f)
        for slices, row in enumerate(csv_reader):
            if slices == int(time_step) + 1:
                temp = row[2:]
                for i in range(len(temp) // 3):
                    point_cloud.append(temp[3 * i: 3 * i + 3])
                break
        f.close()
    point_dist[step] = point_cloud
pressure_list = []  # x y z p nx ny nz
with open(pressure_dir, "r") as f:
    csv_reader = csv.reader(f)
    for step, row in enumerate(csv_reader):
        point = row[:3]
        pressure = row[3]
        normal = row[4:]
        temp = [float(row[i]) for i in range(7)]
        pressure_list.append(temp)
    f.close()

operations = point_dist[cut_slice]
operations = np.array(operations, dtype=np.float32)  # shape=(n, 3)
x_max, x_min = np.max(operations[:, 0]), np.min(operations[:, 0])
y_max, y_min = np.max(operations[:, 1]), np.min(operations[:, 1])
z_max, z_min = np.max(operations[:, 2]), np.min(operations[:, 2])

print("x", x_max, x_min)
print("y", y_max, y_min)
print("z", z_max, z_min)
print(len(operations))
print(len(pressure_list))
target_pressure = []  # 0.725s cost, CPU = i7-10850H
for item in pressure_list:
    if x_min < item[0] < x_max and y_min < item[1] < y_max and z_min < item[2] < z_max:
        target_pressure.append(item)

print(len(target_pressure))
input()
# TODO calculate pairs
# TODO save pairs in cache
# TODO finish todo in starter.58, combine data in cache, generate output-file and clear cache.

