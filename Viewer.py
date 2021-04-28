# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:40:18 2021

@author: 情報基礎数学専攻
"""

import numpy as np
import open3d as o3d
import re
import csv
import traceback


def color_bar(data, maxP=-1000000000, minP=10000000000):
    if maxP == -1000000000 and minP == 10000000000:
        for item in data:
            if item[3] > maxP:
                maxP = item[3]
            if item[3] < minP:
                minP = item[3]
        delta = maxP - minP
    else:
        delta = maxP - minP

    for step, item in enumerate(data):
        color_index = int((data[step][3] - minP) / (delta) * 1152)
        if 128 > color_index >= 0:
            r = 127 - color_index
            g = 0
            b = 255
            color = [r, g, b]
        elif 384 > color_index >= 128:
            r = 0
            g = color_index - 128
            b = 255
            color = [r, g, b]
        elif 640 > color_index >= 384:
            r = 0
            g = 255
            b = 639 - color_index
            color = [r, g, b]
        elif 896 > color_index >= 640:
            r = color_index - 640
            g = 255
            b = 0
            color = [r, g, b]
        elif 1152 > color_index >= 896:
            r = 255
            g = 1151 - color_index
            b = 0
            color = [r, g, b]
        elif color_index < 0:
            color = [150, 0, 200]
        elif color_index >= 1152:
            color = [0, 0, 0]

        data[step] += color
    return data, maxP, minP


def plot(file_name):
    # find_coord=re.compile(r'(-?[0-9]*\.[0-9]*),(-?[0-9]*\.[0-9]*),(-?[0-9]*\.[0-9]*),([0-9]*[\.]?[0-9]*)',re.S)
    find_coord_relaxed = re.compile(
        r'(-?[0-9]+[\.][0-9]+[eE]-?[\+]?[0-9]+|-?[0-9]+\.?[0-9]+),(-?[0-9]+[\.][0-9]+[eE]-?[\+]?[0-9]+|-?[0-9]+\.?[0-9]+),(-?[0-9]+[\.][0-9]+[eE]-?[\+]?[0-9]+|-?[0-9]+\.?[0-9]+),(-?[0-9]+[\.][0-9]+[eE]-?[\+]?[0-9]+|-?[0-9]+\.?[0-9]+)',
        re.S)
    find_coord_relaxed_with_zero = re.compile(
        r'(-?[0-9]+[\.][0-9]+[eE]-?[\+]?[0-9]+|-?[0-9]+\.?[0-9]+),(-?[0-9]+[\.][0-9]+[eE]-?[\+]?[0-9]+|-?[0-9]+\.?[0-9]+),(-?[0-9]+[\.][0-9]+[eE]-?[\+]?[0-9]+|-?[0-9]+\.?[0-9]+),(0)',
        re.S)
    # find_coord=re.compile(r'([\+\-0-9]*\.[0-9]*E\+?\-?[0-9]*)',re.S)
    # find_color=re.compile()
    try:
        while True:
            user_defined_boundary = input('set max and min?[Y/N]')
            if user_defined_boundary == 'N' or user_defined_boundary == '':
                maxP = -1000000000
                minP = 10000000000
                break
            elif user_defined_boundary == 'Y' or user_defined_boundary == "y":
                maxP = float(input('max:'))
                minP = float(input('min:'))
                break
            else:
                print('invalid input, retry...')
                pass

        data = []
        with open(file_name, 'r') as file:
            csv_reader = csv.reader(file)
            for step, row in enumerate(csv_reader):
                try:
                    temp = row
                except IndexError:
                    print('weird number at {}, try to use relaxed re.'.format(step + 1))
                    temp = re.findall(find_coord_relaxed_with_zero, row)
                    temp = list(temp[0])
                for i in range(4):
                    temp[i] = float(temp[i])
                data.append(temp)
            file.close()

        data, maxP, minP = color_bar(data, maxP, minP)

        # data:[[p1],[p2],[p3].....,[pn]]
        print('max is {}, min is {}, delta={}'.format(maxP, minP, maxP - minP))
        input('Any key to preview...')
        point_cloud = np.array(data)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        # pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
        # pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,3:6])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 4:7] / 255)
        o3d.visualization.draw_geometries([pcd])
    except:
        print(traceback.print_exc())
        input()
