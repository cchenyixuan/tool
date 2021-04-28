import os
import json
import time
import re
import csv


def solve(**kwargs):
    """
    kwargs={
            solver:{},
            case_name:{},
            case_dir:{},
            kwargs:{pressure:,...}
            }
    """
    solver = kwargs["solver"]
    case_name = kwargs["case_name"]
    time_step = kwargs["time_step"]
    case_dir = kwargs["case_dir"]
    pressure = kwargs["pressure"]  # dict{i:path_i}

    find_pyw = re.compile(r"pyw.exe", re.S)
    find_time_slice = re.compile(r"(.+)-(.+).csv", re.S)
    if solver == "pressure_purifier.pyw":
        for i in time_step:
            while True:
                running_pyw = os.popen('wmic process get description, processid').read()
                running_pyw = len(re.findall(find_pyw, running_pyw))
                if running_pyw >= os.cpu_count() - 1:
                    time.sleep(5)
                else:
                    break
            prof = [case_name, case_dir, i, pressure[i / 100]]  # [case_name, point_dir, time_step, pressure_dir]
            with open('./temp.bat', 'w') as file:
                file.write('set current_path=%~dp0')
                file.write('\n')
                file.write('start  %current_path%\\{} %1 %2 %3 %4'.format(solver))
                file.close()
            with open('./global.json', 'w') as file:
                json.dump(prof, file)
                file.close()
            with open('./global.json', 'r') as file:
                data = json.load(file)
                file.close()

            para = "temp.bat {} {} {} {}".format(data[0], data[1], data[2], data[3])
            os.system(para)
    if solver == "distance_clip_pressure_purifier.pyw":
        slices = kwargs["slices"]
        create_dir("./data/pressure/{}_purified".format(case_name))
        for i in time_step:
            # check i time step done
            input(i)
            for cut_slice in slices:
                prof = [case_name, case_dir, cut_slice, i, pressure[float("{0:.2f}".format(i/100))]]
                # [case_name, point_dir, cut_slice, time_step, pressure_dir]

                while True:
                    running_pyw = os.popen('wmic process get description, processid').read()
                    running_pyw = len(re.findall(find_pyw, running_pyw))
                    if running_pyw >= os.cpu_count() - 1:
                        time.sleep(5)
                    else:
                        break

                with open('./temp.bat', 'w') as file:
                    file.write('set current_path=%~dp0')
                    file.write('\n')
                    file.write('start  %current_path%\\{} %1 %2 %3 %4 %5'.format(solver))
                    file.close()
                with open('./global.json', 'w') as file:
                    json.dump(prof, file)
                    file.close()
                with open('./global.json', 'r') as file:
                    data = json.load(file)
                    file.close()

                para = "temp.bat {} {} {} {} {}".format(data[0], data[1], data[2], data[3], data[4])
                os.system(para)
            while True:
                time.sleep(5)
                running_pyw = os.popen('wmic process get description, processid').read()
                running_pyw = len(re.findall(find_pyw, running_pyw))
                if running_pyw >= 1:
                    pass
                else:
                    break

            # TODO collect output data and clean the caches
            temp_files = os.listdir("./cache")
            file_dict = {}
            for item in temp_files:
                t, s = re.findall(find_time_slice, item)[0]
                file_dict[int(s)] = "./cache/" + item
            dataframe = []
            opt_list = list(file_dict.keys())
            opt_list.sort()
            for index in opt_list:
                with open(file_dict[index], "r") as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        dataframe.append(row)
                    f.close()
            with open("./data/pressure/{}_purified/{}.csv".format(case_name, float("{0:.2f}".format(i/100))), "w",
                      newline="") as f:
                csv_writer = csv.writer(f)
                for row in dataframe:
                    csv_writer.writerow(row)
                f.close()
            print("Done with time-step-{}".format(i))
            print("clear cache...")
            for item in temp_files:
                os.remove("./cache/" + item)
            print("Done!")
            time.sleep(5)

        pass

    # check if run as background
    background = kwargs["background"]
    if background == "bear":
        print("master call pass")

    elif background != 1:
        check_background()
    else:
        import threading
        t1 = threading.Thread(target=check_background)
        t1.start()


def check_background():
    find_pyw = re.compile(r"pyw.exe", re.S)
    while True:
        running_pyw = os.popen('wmic process get description, processid').read()
        running_pyw = len(re.findall(find_pyw, running_pyw))
        if running_pyw == 0:
            print("Task completed!")
            break
        time.sleep(10)


def create_dir(directory):
    try:
        path = directory
        os.makedirs(path)
    except FileExistsError:
        print("dir '{}' already exists, skip...".format(path))
        pass
