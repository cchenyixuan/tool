import os
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
point_dist = {}

for step, item in enumerate(point_files):
    point_cloud = []
    with open(point_dir + "/" + item, "r") as f:
        csv_reader = csv.reader(f)
        for step, row in enumerate(csv_reader):
            if step == int(time_step) + 1:
                temp = row[2:]
                for i in range(len(temp) // 3):
                    point_cloud.append(temp[3 * i: 3 * i + 3])
                break
        f.close()
    point_dist[step] = point_cloud
pressure_list = []
with open(pressure_dir, "r") as f:
    csv_reader = csv.reader(f)
    for step, row in enumerate(csv_reader):
        point = row[:3]
        pressure = row[3]
        normal = row[4:]
        temp = [float(row[i]) for i in range(7)]
        pressure_list.append(temp)
    f.close()

operations = list(point_dist.keys())




