import csv
import re
import os


def create_dir(directory):
    try:
        path = directory
        os.makedirs(path)
    except FileExistsError:
        print("dir '{}' already exists, skip...".format(path))
        pass


def save_file(data_file, case_name):
    find_number = re.compile(r"T=(.*) Bo")
    time_stamp = str(re.findall(find_number, data_file[0][0])[0])
    if time_stamp == "0":
        time_stamp = "0.0"
    filename = "./data/pressure/" + case_name + "/" + time_stamp + ".csv"
    with open(filename, "w", newline='') as f:
        csv_writer = csv.writer(f)
        # csv_writer.writerow(head)
        for step, item in enumerate(data_file):
            if step == 0 or step == len(data_file) - 1:
                continue
            csv_writer.writerow(item[:7])
        # csv_writer.writerow(["","","",""])
        f.close()
    print("Done with {}".format(filename))


def convert_file(case_name, pressure_file):
    find_wall = re.compile(r"Boundary", re.S)
    data = []
    create_dir("./data/pressure/"+case_name)
    with open("./data/pressure/"+pressure_file, "r", newline='') as file:
        csv_reader = csv.reader(file)
        for step, item in enumerate(csv_reader):
            if step == 0:
                head = item
                continue
            try:
                flag = re.findall(find_wall, str(item))[0]
                if flag == "Boundary":
                    print(item)
                if step <= 5:
                    pass
                else:
                    save_file(data, case_name)
                    data = []
            except:
                pass
            data.append(item)

            # if step>=10:
            #     break
        save_file(data, case_name)
        file.close()
