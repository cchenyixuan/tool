import os
import json


def main(file_path, activated, cardio_circle, output_dir, name_rule):
    file_number = len(os.listdir(file_path))
    with open(file_path+"/"+os.listdir(file_path)[0], "r") as f:
        time_slice = -1
        for step, row in enumerate(f):
            time_slice += 1
            pass
        f.close()
    if time_slice >= 100:
        cardio_circle = cardio_circle / time_slice
    elif time_slice >= 90:
        print("WARNING: Only {} time-slice detected!".format(time_slice))
        cardio_circle = cardio_circle / time_slice
    else:
        print("WARNING: Only {} time-slice detected, please confirm the cardio circle...".format(time_slice))
        cardio_circle = input("Cardio Circle:")
    prof = [file_path, file_number, str(activated), cardio_circle, output_dir, name_rule]

    with open('./forward_difference_starter.bat', 'w') as file:
        file.write('set current_path=%~dp0')
        file.write('\n')
        file.write('start  %current_path%\\forward_difference.py %1 %2 %3 %4 %5 %6')
        file.close()

    with open('./global.json', 'w') as file:
        json.dump(prof, file)

    with open('./global.json', 'r') as file:
        data = json.load(file)

    for item in data[2]:
        para = "forward_difference_starter.bat {} {} {} {} {} {}"\
            .format(data[0], data[1], item, data[3], data[4], data[5])
        os.system(para)
