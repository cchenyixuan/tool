import os
import json
import time
import re


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
    if solver is "pressure_purifier.pyw":
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
    if solver is "distance_clip_pressure_purifier.pyw":
        slices = kwargs["slices"]
        for i in time_step:
            # check i time step done
            while True:
                running_pyw = os.popen('wmic process get description, processid').read()
                running_pyw = len(re.findall(find_pyw, running_pyw))
                if running_pyw >= 1:
                    time.sleep(5)
                else:
                    print("Done with time-step-{}".format(i))
                    # TODO collect output data and clean the caches
                    break
            for cut_slice in slices:
                prof = [case_name, case_dir, cut_slice, i, pressure[i / 100]]
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
