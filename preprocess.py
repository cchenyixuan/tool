import csv
import re
import os


def preprocess(file_path):
    find_number = re.compile(r'z}x([0-9]*)', re.S)
    csvs = os.listdir(file_path)
    for point_slice in csvs:

        data = []
        with open(file_path+"/"+point_slice, 'r') as file:
            file_reader = csv.reader(file)
            for step, item in enumerate(file_reader):
                if step == 0:
                    # print(item)#re.find total number
                    for head in item:
                        try:
                            count = int(re.findall(find_number, head)[0])
                            point_number = count
                            print(point_number)
                        except:
                            # not found or already converted
                            pass
                    try:
                        headers = ['Length', 'Area']
                        for i in range(point_number):
                            headers.append('x{}'.format(i))
                            headers.append('y{}'.format(i))
                            headers.append('z{}'.format(i))
                    except UnboundLocalError:
                        print("{} already reformed or may contain errors, skip...".format(point_slice))
                        break

                else:
                    data.append(item)
        if not data:
            continue
        with open(file_path+"/"+point_slice, 'w', newline='') as file:
            file_writer = csv.writer(file)
            file_writer.writerow(headers)
            file_writer.writerows(data)
        print('Done with {}, {} points'.format(point_slice, point_number))

    return
