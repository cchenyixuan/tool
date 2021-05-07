import os


class Names:
    """File names saved here with paths"""

    def __init__(self):
        self.cases_dir = {"Mie02": "./data/zio/Mie_case02_retry_20200516_after_sorting/",
                          "Mie03": "./data/zio/Mie_case03_WL204_3rd_growing_after_sorting/",
                          "Mie02_10": "./data/zio/M02_10slices_zio/",
                          "Mie03_twin": "./data/zio/M03_2020_twins_zio_AS/",
                          "MI02": "./data/zio/MI02_WL266_ab_AS/"}

        self.pressure_file = {"Mie02": "M02_Pressure_100.csv",
                              "Mie02_10": "M02_Pressure_100.csv",
                              "Mie03": "M02_Pressure_100.csv"}

        self.cases_name_rule = {"Mie02": "Mie_case02_{}.csv",
                                "Mie03": "MIe_case03_a_{}.csv",
                                "Mie02_10": "Mie_case02_{}.csv",
                                "Mie03_twin": "M03_2020_{}.csv",
                                "MI02": "MI02_WL266_ab_{}.csv"}

        self.cases_cardio_circle = {"Mie02": 0.804,
                                    "Mie03": 0.804,
                                    "Mie02_10": 0.804}

        self.cases_pressure_dir = {}

        self.project_dir = "./unnamed_project/"
        try:
            os.makedirs(self.project_dir)
        except FileExistsError:
            pass
        self.data_name = ""
        self.data = {}
        self.movement_data = {}
        self.point_number = 0


class DataManager(Names):
    def __init__(self):
        super().__init__()
        pass

    def load_data(self, data_name):
        import os
        import csv
        import re
        self.data_name = data_name
        # load point_cloud data
        find_index = re.compile(r"([0-9]+).csv", re.S)
        data = os.listdir(self.cases_dir[data_name])
        direction = self.cases_dir[data_name]
        for item in data:
            index = int(re.findall(find_index, item)[0])
            self.data[index] = {}
            with open(direction + item, "r") as file:
                csv_reader = csv.reader(file)
                for step, row in enumerate(csv_reader):
                    if step == 0:
                        continue
                    self.data[index][step - 1] = row[2:]
        # call load_output to refresh present states
        self.load_output()
        self.load_point_number()

    def load_output(self):
        # load output data
        self.project_dir = "./" + self.data_name + "/"
        try:
            os.makedirs(self.project_dir)
        except FileExistsError:
            pass
        files = os.listdir(self.project_dir)
        # no data found
        if files is None:
            pass
        # normal load data
        else:
            for file in files:
                self.movement_data[file[:-4]] = self.project_dir + "/" + file

    def load_point_number(self):
        # load point number
        for i in range(1, len(self.data.keys()) + 1):
            self.point_number += len(self.data[i][0]) / 3
        self.point_number = int(self.point_number)

    def show(self, time_index):
        import numpy as np
        import open3d as o3d
        slices = list(self.data.keys())
        point_data = []
        for i in slices:
            data_slice = self.data[i][time_index]
            point_number = int(len(data_slice) // 3)
            for j in range(point_number):
                point_data.append(data_slice[j * 3:j * 3 + 3])
        point_cloud = np.array(point_data)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        o3d.visualization.draw_geometries([pcd])

    def create_ply(self, time_index, name="p_mesh_c{}.ply", crop=True):
        import numpy as np
        import open3d as o3d
        try:
            os.makedirs("./mesh/" + self.data_name)
        except FileExistsError:
            pass
        slices = list(self.data.keys())
        point_data = []
        for i in slices:
            data_slice = self.data[i][time_index]
            point_number = int(len(data_slice) // 3)
            for j in range(point_number):
                point_data.append(data_slice[j * 3:j * 3 + 3])
        point_cloud = np.array(point_data, dtype=np.float32)
        point_position = []
        for item in point_cloud:
            point_position.append(item[:3])

        x_, y_, z_ = 0, 0, 0
        for item in point_position:
            x_ += item[0]
            y_ += item[1]
            z_ += item[2]
        x_ /= point_cloud.shape[0]
        y_ /= point_cloud.shape[0]
        z_ /= point_cloud.shape[0]
        middle = np.array([x_, y_, z_])
        normal_vectors = []
        for item in point_position:
            normal = item - middle
            normal_vectors.append(normal)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.normals = o3d.utility.Vector3dVector(normal_vectors[:])  # fake normals with low accu
        poisson_mesh = \
            o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1,
                                                                      linear_fit=False)[0]
        bbox = pcd.get_axis_aligned_bounding_box()
        p_mesh_crop = poisson_mesh.crop(bbox)
        if crop is True:
            o3d.io.write_triangle_mesh("./mesh/" + self.data_name + "/" + name.format(time_index), p_mesh_crop)
        else:
            o3d.io.write_triangle_mesh("./mesh/" + self.data_name + "/" + name.format(time_index),
                                       poisson_mesh)  # a method not cropping the mesh


# Done
class Analysis(DataManager):
    def __init__(self, data_name=None):
        super().__init__()
        if data_name is None:
            print('Try "Analysis(data name)"')
            print("Or try '.load_data(data name)'")
        else:
            self.load_data(data_name)

    def show_data(self, datatype):
        print("Here is DataManager.Analysis.show_data!")
        import Viewer
        self.load_output()
        try:
            data = self.movement_data[datatype]
            print(data)
            Viewer.plot(data)
        except KeyError:
            print("File {}.csv not found, unable to process!".format(datatype))

        pass

    def calculate_data(self, activated=1234567):
        print("Here is DataManager.Analysis.calculate_data!")
        import preprocess
        import starter_forward_difference
        file_path = self.cases_dir[self.data_name]
        cardio_circle = self.cases_cardio_circle[self.data_name]
        output_dir = self.project_dir
        name_rule = self.cases_name_rule[self.data_name]
        preprocess.preprocess(file_path)
        starter_forward_difference.main(file_path, activated, cardio_circle, output_dir, name_rule)

        pass


class DeepAnalysis(Analysis):
    def __init__(self, data_name=None):
        super().__init__(data_name)

        self.plot_data = {"time": None,
                          "slices": [key for key in self.data.keys()],
                          "type": None}
        self.dataset = {"point": None,
                        "normal": None,
                        "pressure": None,
                        "m": None,
                        "mu": None,
                        "k": None}
        self.full_dataset = {"point": None,
                             "normal": None,
                             "pressure": None,
                             "m": None,
                             "mu": None,
                             "k": None,
                             "data": {}}
        self.pressure = {}
        self.purified_pressure = {}
        self.load_pressure()
        self.ode_data_x = []
        self.ode_data_y = []
        self.ode_data_z = []
        self.ode_data_norm = []
        self.ode_data_chen = []

    def load_pressure(self):
        """load pressure data"""
        try:
            assert self.data_name != ""
        except AssertionError:
            print("Failed to init...")
            print("Case name is indispensable for DeepAnalysis!")
            return
        import os
        import csvreader
        try:
            self.cases_pressure_dir[self.data_name] = "./data/pressure/" + self.data_name
            os.listdir(self.cases_pressure_dir[self.data_name])
            assert os.listdir(self.cases_pressure_dir[self.data_name]) is not []
        except FileNotFoundError:
            print("Pressure folder not found, generating files...")
            csvreader.convert_file(self.data_name, self.pressure_file[self.data_name])
        except AssertionError:
            print("Pressure folder is empty, generating files...")
            csvreader.convert_file(self.data_name, self.pressure_file[self.data_name])
        pressure_files = os.listdir(self.cases_pressure_dir[self.data_name])

        for pressure_file in pressure_files:
            self.pressure[int(float(pressure_file[:-4]) * 100) / 100] = \
                self.cases_pressure_dir[self.data_name] + "/" + pressure_file
        pass

    def purify_pressure(self, time_step=None):
        if time_step is None:
            time_step = [i for i in range(100)]
        import starter
        # starter.solve(solver="pressure_purifier.pyw", time_step=time_step,
        #              case_name=self.data_name, slices=list(self.data.keys()),
        #              case_dir=self.cases_dir[self.data_name], pressure=self.pressure, background="bear")
        starter.solve(solver="distance_clip_pressure_purifier.pyw", time_step=time_step,
                      case_name=self.data_name, slices=list(self.data.keys()),
                      case_dir=self.cases_dir[self.data_name], pressure=self.pressure, background="bear")
        pass

    def advanced_show(self, **kwargs):
        """**kwargs = [time=0, slices=[1,2,3], type=[pressure, m, mu, k,...]]"""
        print("Here is DataManager.Analysis.DeepAnalysis.advanced_show!")
        import csv
        import random
        for keyword in kwargs.keys():
            self.plot_data[keyword] = kwargs[keyword]
        time = "-" + str(self.plot_data["time"]) + "-"
        slices = "-" + str(self.plot_data["slices"][0]) + "-" + str(self.plot_data["slices"][-1]) + "-"
        type_ = "-"
        for i in self.plot_data["type"]:
            type_ += str(i)
            type_ += "-"
        with open(self.project_dir + "time{}_slice{}_type{}.csv".format(time, slices, type_), "w", newline="") as file:
            csv_writer = csv.writer(file)
            data = []
            for item in self.plot_data["slices"]:
                raw_points = self.data[item][self.plot_data["time"]]
                for i in range(len(raw_points) // 3):
                    temp = raw_points[i * 3:i * 3 + 3]
                    temp.append(random.random())  # TODO show specific data, now use random as a test
                    data.append(temp)
            for row in data:
                csv_writer.writerow(row)
            file.close()
        self.show_data("time{}_slice{}_type{}".format(time, slices, type_))

        pass

    def analysis_ode(self, model="a"):
        print("Here is DataManager.Analysis.DeepAnalysis.analysis_ode!")
        print("Using model {}".format(model))
        self.ode_data_x = []
        self.ode_data_y = []
        self.ode_data_z = []
        self.ode_data_norm = []
        self.ode_data_chen = []
        t = 0.01
        import numpy as np
        import os
        import csv
        for file in os.listdir("./data/pressure/" + self.data_name + "_purified/"):
            self.purified_pressure[float(file[:-4])] = "./data/pressure/" + self.data_name + "_purified/" + file
        data_index = list(self.purified_pressure.keys())
        data_index.sort()

        for step in range(len(data_index)):
            data = []
            with open(self.purified_pressure[data_index[step]], "r") as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    data.append(row)
                f.close()
            self.full_dataset["data"][step] = np.array(data, dtype=np.float32)

        def save_data(ode_data, name):
            import csv
            base_data = self.full_dataset["data"][0][:, :3]
            with open("./" + self.data_name + "/{}.csv".format(name), "w", newline="") as f:
                csv_writer = csv.writer(f)
                for row_ in zip(base_data, ode_data):
                    row_ = [float(item) for sublist in row_ for item in sublist]
                    csv_writer.writerow(row_)
                f.close()

        if model == "a":
            for index in range(self.point_number - 100):
                Ax = np.zeros([98, 1])
                Bx = np.zeros([98, 1])
                Ay = np.zeros([98, 1])
                By = np.zeros([98, 1])
                Az = np.zeros([98, 1])
                Bz = np.zeros([98, 1])
                A = np.zeros([98, 1])
                B = np.zeros([98, 1])
                for time_step in range(len(self.purified_pressure.keys()) - 2):
                    point_cloud1 = self.full_dataset["data"][time_step][index, :3]
                    point_cloud2 = self.full_dataset["data"][time_step + 1][index, :3]
                    point_cloud3 = self.full_dataset["data"][time_step + 2][index, :3]
                    pressure = self.full_dataset["data"][time_step + 1][index, 3] * 0.0001
                    normal = self.full_dataset["data"][time_step + 1][index, 4:]
                    # pressure assert to have same direction with a
                    dx1 = point_cloud2 - point_cloud1
                    dx2 = point_cloud3 - point_cloud2
                    dx = 0.5 * (dx1 + dx2)
                    v = dx / t
                    a_ = (dx2 - dx1) / t ** 2

                    Ax[time_step, 0] = abs(a_[0])
                    Bx[time_step] = abs(pressure * normal[0])
                    Ay[time_step, 0] = abs(a_[1])
                    By[time_step] = abs(pressure * normal[1])
                    Az[time_step, 0] = abs(a_[2])
                    Bz[time_step] = abs(pressure * normal[2])

                    A[time_step, 0] = np.linalg.norm(a_) * 1
                    B[time_step] = np.linalg.norm(pressure * normal @ (a_ / np.linalg.norm(a_)))
                    # A[time_step, 1] = np.linalg.norm(v) * ((a_ / np.linalg.norm(a_)) @ (v / np.linalg.norm(v)))
                    # A[time_step, 2] = np.linalg.norm(dx) * ((a_ / np.linalg.norm(a_)) @ (dx / np.linalg.norm(dx)))
                ans_x = np.linalg.solve(Ax.T @ Ax, Ax.T @ Bx)
                self.ode_data_x.append([float(ans_x[0])])  # , float(ans_x[1])])
                ans_y = np.linalg.solve(Ay.T @ Ay, Ay.T @ By)
                self.ode_data_y.append([float(ans_y[0])])  # , float(ans_y[1])])
                ans_z = np.linalg.solve(Az.T @ Az, Az.T @ Bz)
                self.ode_data_z.append([float(ans_z[0])])  # , float(ans_z[1])])
                self.ode_data_norm.append([np.sqrt(
                    ans_x[0] ** 2 + ans_y[0] ** 2 + ans_z[0] ** 2)])  # , np.sqrt(ans_x[1]**2+ans_y[1]**2+ans_z[1]**2)])
                ans_chen = np.linalg.solve(A.T @ A, A.T @ B)
                self.ode_data_chen.append([float(ans_chen[0])])
            save_data(self.ode_data_x, "m-x")
            save_data(self.ode_data_y, "m-y")
            save_data(self.ode_data_z, "m-z")
            save_data(self.ode_data_norm, "m-norm")
            save_data(self.ode_data_chen, "m-norm-c")
            pass
        elif model == "av":
            for index in range(self.point_number - 100):
                Ax = np.zeros([98, 2])
                Bx = np.zeros([98, 1])
                Ay = np.zeros([98, 2])
                By = np.zeros([98, 1])
                Az = np.zeros([98, 2])
                Bz = np.zeros([98, 1])
                A = np.zeros([98, 2])
                B = np.zeros([98, 1])
                for time_step in range(len(self.purified_pressure.keys()) - 2):
                    point_cloud1 = self.full_dataset["data"][time_step][index, :3]
                    point_cloud2 = self.full_dataset["data"][time_step + 1][index, :3]
                    point_cloud3 = self.full_dataset["data"][time_step + 2][index, :3]
                    pressure = self.full_dataset["data"][time_step + 1][index, 3] * 0.0001
                    normal = self.full_dataset["data"][time_step + 1][index, 4:]
                    # pressure assert to have same direction with a
                    dx1 = point_cloud2 - point_cloud1
                    dx2 = point_cloud3 - point_cloud2
                    dx = 0.5 * (dx1 + dx2)
                    v = dx / t
                    a_ = (dx2 - dx1) / t ** 2
                    Ax[time_step, 0] = abs(a_[0])
                    Ax[time_step, 1] = v[0]
                    Bx[time_step] = abs(pressure * normal[0])
                    Ay[time_step, 0] = abs(a_[1])
                    Ay[time_step, 1] = v[1]
                    By[time_step] = abs(pressure * normal[1])
                    Az[time_step, 0] = abs(a_[2])
                    Az[time_step, 1] = v[2]
                    Bz[time_step] = abs(pressure * normal[2])

                    A[time_step, 0] = np.linalg.norm(a_) * 1
                    A[time_step, 1] = np.linalg.norm(v) * ((a_ / np.linalg.norm(a_)) @ (v / np.linalg.norm(v)))
                    B[time_step] = np.linalg.norm(pressure * normal @ (a_ / np.linalg.norm(a_)))
                ans_x = np.linalg.solve(Ax.T @ Ax, Ax.T @ Bx)
                self.ode_data_x.append([float(ans_x[0]), float(ans_x[1])])
                ans_y = np.linalg.solve(Ay.T @ Ay, Ay.T @ By)
                self.ode_data_y.append([float(ans_y[0]), float(ans_y[1])])
                ans_z = np.linalg.solve(Az.T @ Az, Az.T @ Bz)
                self.ode_data_z.append([float(ans_z[0]), float(ans_z[1])])
                self.ode_data_norm.append([np.sqrt(
                    ans_x[0] ** 2 + ans_y[0] ** 2 + ans_z[0] ** 2),
                    np.sqrt(ans_x[1] ** 2 + ans_y[1] ** 2 + ans_z[1] ** 2)])
                ans_chen = np.linalg.solve(A.T @ A, A.T @ B)
                self.ode_data_chen.append([float(ans_chen[0]), float(ans_chen[1])])
            save_data(self.ode_data_x, "m-mu-x")
            save_data(self.ode_data_y, "m-mu-y")
            save_data(self.ode_data_z, "m-mu-z")
            save_data(self.ode_data_norm, "m-mu-norm")
            save_data(self.ode_data_chen, "m-mu-norm-c")
            pass

        pass

    def analysis_ensemble_kalman_filter(self):
        self.dataset = self.dataset
        print("Here is DataManager.Analysis.DeepAnalysis.analysis_ensemble_kalman_filter!")

        # TODO EnKF model
        # TODO save analyzed data
        pass


class Preprocessor(Analysis):
    def __init__(self, data_name):
        super().__init__(data_name)
        """
        p0=(x0,y0,z0)
        v0=(ex,ey,ez)
        p(x,y,z)
        ((p+k*v0)-p0)@v0=0
        ==>k=(p0-p)@v0/|v0|
        ==>p`=p+k*v0

        v0:(rcost,rsint,sqrt(1-r**2)), r in (0,1), t in (0,2pi)
        p0:R*v0 + C(x~,y~,z~), Any constant R > max(|p-C|)

        for p in geometry:
            calculate p`
        for all p`:
            c`=avg(p`)
        create resolution image
        find maxlikelyhood distance(spin the image and calculate costfunction)

        iteration until r and t calculated
        figure out least cost
        that direction is required

        cost function:
        cost = sum(0.5*sin(pi/180)*(cd-ab))
        a,b,c,d = p1-c`,p2-c`,p3-c`,p4-c`

        """

    def project(self):
        from PIL import Image
        import numpy as np
        raw_image = Image.open("M02shape.png")
        image = np.array(raw_image, dtype=np.int32)
        pad = np.zeros([image.shape[0] + 2, image.shape[1] + 2])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j, 0] == image[i, j, 1] == image[i, j, 2] == 255:
                    pass
                else:
                    pad[i + 1, j + 1] = 255
        kernel = np.reshape(np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]), [9, 1])
        boundary_image = np.zeros([image.shape[0], image.shape[1]])
        for i in range(boundary_image.shape[0]):
            for j in range(boundary_image.shape[1]):
                value = np.reshape(pad[i:i + 3, j:j + 3], [1, 9]) @ kernel
                boundary_image[i, j] += value

        data = [self.data[key][0] for key in self.data.keys()]
        data = [i[3 * j:3 * j + 3] for i in data for j in range(int(len(i) / 3 - 1))]
        data = np.array(data, dtype=np.float32)
        r_ = [0.1 * i for i in range(1, 11)]  # 0.1-1
        t_ = [3.141592653589793 / 30 * i for i in range(60)]  # 0-2pi
        c = np.array([np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2])])
        R = 100

        cost_ans = []
        project_log = []
        for r in r_:
            for t in t_:
                """
                def the vector vertical with v0 and have max projection on z-axis as 'base yaxis' for image
                x=-cos(t)*sqrt((1-r**2))
                y=-sin(t)*sqrt((1-r**2))
                z=r
                def the vector vertical with v0, 'base yaxis' and satisfy right hand rule
                x=-sin(t)
                y=-cos(t)
                z=0
                """
                v0 = np.array([r * np.cos(t), r * np.sin(t), np.sqrt(1 - r ** 2)])  # the other is -np.sqrt(1-a**2)
                base_y_axis = np.array([-np.cos(t) * np.sqrt(1 - r ** 2), -np.sin(t) * np.sqrt(1 - r ** 2), r])
                base_x_axis = np.array([-np.sin(t), np.cos(t), 0])

                p0 = R * v0 + c
                shadow = [p + (p0 - p) @ v0 * v0 for p in data]
                shadow = np.array(shadow, dtype=np.float32)  # all points on a plane
                center = np.array([np.mean(shadow[:, 0]), np.mean(shadow[:, 1]), np.mean(shadow[:, 2])])
                image_array = np.zeros([548, 870])  # size of operation image
                image_center = [548 / 2, 870 / 2]  # center of operation image boundary
                relative_shadow = np.array([shadow[i] - center for i in range(shadow.shape[0])],
                                           dtype=np.float32)  # delta x,y,z
                # convert 3d(x,y,z) to 2d(x,y)
                temp = np.array([[item @ base_x_axis, item @ base_y_axis] for item in relative_shadow])
                dx = np.max(temp[:, 0]) - np.min(temp[:, 0])
                dy = np.max(temp[:, 1]) - np.min(temp[:, 1])

                resolution = min(dx, dy) / max(boundary_image.shape)
                image_temp = []
                for i in range(boundary_image.shape[0]):
                    for j in range(boundary_image.shape[1]):
                        if boundary_image[i, j] >= 200:
                            image_temp.append([(j - 435) * resolution, (274 - i) * resolution])
                # clear rubbish

                opt_array = np.array(image_temp)
                start_point = np.array([np.max(opt_array[:, 0]), 0])
                clean_list = [start_point]
                while True:
                    max_distance = 0
                    selected_point = 0
                    cache = []
                    for step, point in enumerate(image_temp):
                        point = np.array(point)
                        d = np.linalg.norm(point - start_point)
                        if 4 * resolution > d:  # 4 is experience
                            cache.append(step)
                        if 4 * resolution > d > max_distance:
                            selected_point = point
                            max_distance = d
                    try:
                        assert type(selected_point) is np.ndarray
                        clean_list.append(selected_point)
                        cache.reverse()
                        for item in cache:
                            image_temp.pop(item)
                        start_point = selected_point
                    except AssertionError:
                        break
                print("clean_list_test_length:", len(clean_list))

                # move image-center to center
                image_temp = np.array(clean_list)
                movement = np.array([np.mean(image_temp[:, 0]), np.mean(image_temp[:, 1])])
                for i in range(image_temp.shape[0]):
                    image_temp[i] -= movement

                # debug
                debug = []
                for dbg in temp:
                    debug.append([dbg[0], dbg[1], 0, 1])
                for dbg in image_temp:
                    debug.append([dbg[0], dbg[1], 0, 2])
                import csv
                with open("test{}-{}.csv".format(r, t), "w", newline="") as f:
                    csv_writer = csv.writer(f)
                    for row in debug:
                        csv_writer.writerow(row)

                print("image range:", np.max(image_temp[:, 0]), np.min(image_temp[:, 0]), np.max(image_temp[:, 1]),
                      np.min(image_temp[:, 1]))

                # TODO calculate cost function
                point_pairs = []
                for i in range(image_temp.shape[0]):
                    p0 = image_temp[i]
                    x0 = p0[0]
                    y0 = p0[1]
                    max_distance = 0
                    p1 = 0
                    for item in temp:
                        x = item[0]
                        y = item[1]
                        if np.sign(x0) == np.sign(x) and np.sign(y0) == np.sign(y):
                            if abs(y0 * (x - x0) - x0 * (y - y0)) <= 3 * resolution:
                                d = np.linalg.norm(item)
                                if d > max_distance:
                                    p1 = item
                                    max_distance = d
                    try:
                        assert type(p1) is np.ndarray
                        point_pairs.append([p0, p1])
                    except AssertionError:
                        raise AssertionError
                print("pairs length:", len(point_pairs))

                # cost function:
                last = 0
                times = 0
                res = []
                log = []
                while True:
                    cost_list = []
                    scale_history = 1

                    for theta in [np.pi / 180 * i for i in range(360)]:
                        spin = np.array([[np.cos(theta), -np.sin(theta)],
                                         [np.sin(theta), np.cos(theta)]])
                        cost = 0
                        for i in range(len(point_pairs) - 1):
                            p1, p3 = point_pairs[i].copy()
                            p2, p4 = point_pairs[i + 1].copy()
                            p1 = spin @ p1
                            p2 = spin @ p2
                            cost += abs(np.sin(np.pi * 2 / len(point_pairs)) *
                                        (np.linalg.norm(p3) * np.linalg.norm(p4) -
                                         np.linalg.norm(p1) * np.linalg.norm(p2)))
                        cost_list.append(cost)
                    print(min(cost_list))
                    if times == 0:
                        last = min(cost_list)
                        times += 1
                        scale_factor = 1.1
                        scale_history *= scale_factor
                        for i in range(len(point_pairs)):
                            point_pairs[i][0] *= scale_factor
                    else:
                        if min(cost_list) >= last:
                            scale_factor = 1 / scale_factor
                        if min(cost_list) <= last:
                            scale_factor = scale_factor
                        scale_history *= scale_factor
                        for i in range(len(point_pairs)):
                            point_pairs[i][0] *= scale_factor
                        last = min(cost_list)
                        times += 1
                    res.append(last)
                    log.append([cost_list.index(min(cost_list)) * np.pi / 180, scale_history])  # theta, scale-history
                    if times == 20:
                        break

                cost = int(min(res))

                cost_ans.append(cost)
                project_log.append([r, t, resolution, movement, *log[res.index(min(res))]])
                print("min(cost_ans)")
                print(min(cost_ans))
                print(cost_ans)
        print(min(cost_ans))
        print(res)
        print(cost_list)

        # analysis cost:
        r0 = project_log[cost_list.index(min(cost_list))][0]
        t0 = project_log[cost_list.index(min(cost_list))][1]
        resolution = project_log[cost_list.index(min(cost_list))][2]
        move_vector = project_log[cost_list.index(min(cost_list))][3]
        theta = project_log[cost_list.index(min(cost_list))][4]
        scale_factor = project_log[cost_list.index(min(cost_list))][5]
        print("r", r0, "t", t0, "resolution", resolution, "move", move_vector, "spin-angle", theta, "scale",
              scale_factor)


"""
                import csv
                with open("test1.csv", "w", newline="") as f:
                    csv_writer = csv.writer(f)
                    for i in range(temp.shape[0]):
                        csv_writer.writerow(temp[i])
                    for i in range(image_temp.shape[0]):
                        csv_writer.writerow(image_temp[i])
                dx = np.max(shadow[:, 0]) - np.min(shadow[:, 0])
                dy = np.max(shadow[:, 1]) - np.min(shadow[:, 1])
                dz = np.max(shadow[:, 2]) - np.min(shadow[:, 2])
                resolution = np.sqrt(dx**2+dy**2+dz**2) / min(image_array.shape)  # length of tiny square

                theta = [np.pi*2/2001*i for i in range(2000)]
                temp = [[item @ base_x_axis, item @ base_y_axis] for item in relative_shadow]  # TODO important change 3d to 2d point cloud
                index = []
                for i in theta:
                    max_distance = 0
                    point_index = 0
                    for step, j in enumerate(temp):

                        if np.tan(i)<=j[1]/j[0]<=np.tan(i+np.pi*2/2001):
                            distance = j[0]**2+j[1]**2
                            if distance>max_distance:
                                point_index = step
                                max_distance = distance
                    index.append(point_index)
                print(index)
                print(len(index))


                votex = np.sqrt(3) / 3 * resolution
                subsample = [shadow[i*10] for i in range(shadow.shape[0]//10)]
                # up=274 down=273 left=435 right=434
                for i in range(image_array.shape[0]):

                    for j in range(image_array.shape[1]):

                        cell_center = center + base_y_axis * resolution * (274-i) + base_x_axis * resolution * (j-435)
                        min_x = cell_center[0] - votex
                        max_x = cell_center[0] + votex
                        min_y = cell_center[1] - votex
                        max_y = cell_center[1] + votex
                        min_z = cell_center[2] - votex
                        max_z = cell_center[2] + votex
                        for d in index:
                            point = shadow[d]
                            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y and min_z <= point[2] <= max_z:
                                image_array[i, j] = 255
                    if i%100==0:
                        from PIL import Image
                        image = Image.fromarray(image_array)
                        image.show()

"""


class Console:
    def __init__(self):
        self.console()
        pass

    def console(self):
        print("This is a python console, call Gui() to enter your code.")
        print("Code saved as '.text' attribute of Gui object.")
        import traceback

        while True:
            my_code = input(">>>")
            if my_code == "exit" or my_code == "quit":
                break
            try:
                execute_times = 0
                try:
                    answer = eval(my_code)
                    if answer is not None:
                        print(answer)
                    execute_times += 1
                except:
                    pass
                if execute_times == 0:
                    exec(my_code)
            except:
                traceback.print_exc()
        return


class Gui:
    def __init__(self):
        self.text = None
        import tkinter
        root = tkinter.Tk()
        root.geometry("300x450")
        root.title("Text Editor")
        self.inbox = tkinter.Text(root, width=42, height=32)
        button1 = tkinter.Button(root, text="apply", width=41)
        self.inbox.place(x=0, y=0)
        button1.place(x=0, y=420)
        button1.bind('<Button-1>', lambda event: self.get_data(event))
        root.mainloop()

    def get_data(self, event):
        self.text = self.inbox.get('1.0', 'end')

    def restart(self):
        import tkinter
        root = tkinter.Tk()
        root.geometry("300x450")
        root.title("Text Editor")
        self.inbox = tkinter.Text(root, width=42, height=32)
        button1 = tkinter.Button(root, text="apply", width=41)
        self.inbox.place(x=0, y=0)
        button1.place(x=0, y=420)
        button1.bind('<Button-1>', lambda event: self.get_data(event))
        root.mainloop()


a = Preprocessor("Mie02")
a.project()

Console()
