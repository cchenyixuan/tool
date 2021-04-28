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
                              "Mie02_10": "M02_Pressure_100.csv"
                              }

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
            self.point_number += len(a.data[i][0])/3
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


class Test:
    def __init__(self):
        print(1)
        self.book = {1: 2}

    def test(self):
        print("This is Test.test!")


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
        #starter.solve(solver="pressure_purifier.pyw", time_step=time_step,
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

    def analysis_ode(self):
        print("Here is DataManager.Analysis.DeepAnalysis.analysis_ode!")
        t = 0.01
        import numpy as np
        import os
        import csv
        for file in os.listdir("./data/pressure/" + self.data_name + "_purified/"):
            self.purified_pressure[float(file[:-4])] = "./data/pressure/" + self.data_name + "_purified/" + file
        data_index = list(self.purified_pressure.keys())
        data_index.sort()

        if self.full_dataset["data"] != {}:
            pass
        else:
            for step in range(len(data_index)):
                data = []
                with open(self.purified_pressure[data_index[step]], "r") as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        data.append(row)
                    f.close()
                self.full_dataset["data"][step] = np.array(data, dtype=np.float32)
        for index in range(self.point_number):
            Ax = np.zeros([98, 2])
            Bx = np.zeros([98, 1])
            Ay = np.zeros([98, 2])
            By = np.zeros([98, 1])
            Az = np.zeros([98, 2])
            Bz = np.zeros([98, 1])
            for time_step in range(len(self.purified_pressure.keys())-2):
                point_cloud1 = self.full_dataset["data"][time_step][index, :3]
                point_cloud2 = self.full_dataset["data"][time_step+1][index, :3]
                point_cloud3 = self.full_dataset["data"][time_step+2][index, :3]
                pressure = self.full_dataset["data"][time_step+1][index, 3] * 0.0001
                normal = self.full_dataset["data"][time_step+1][index, 4:]
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
                #A[time_step, 0] = np.linalg.norm(a_) * 1
                #A[time_step, 1] = np.linalg.norm(v) * ((a_ / np.linalg.norm(a_)) @ (v / np.linalg.norm(v)))
                #A[time_step, 2] = np.linalg.norm(dx) * ((a_ / np.linalg.norm(a_)) @ (dx / np.linalg.norm(dx)))
                #B[time_step] = np.linalg.norm(pressure * normal @ (a_ / np.linalg.norm(a_)))
            ans_x = np.linalg.solve(Ax.T @ Ax, Ax.T @ Bx)
            self.ode_data_x.append([float(ans_x[0]), float(ans_x[1])])
            ans_y = np.linalg.solve(Ay.T @ Ay, Ay.T @ By)
            self.ode_data_y.append([float(ans_y[0]), float(ans_y[1])])
            ans_z = np.linalg.solve(Az.T @ Az, Az.T @ Bz)
            self.ode_data_z.append([float(ans_z[0]), float(ans_z[1])])
            self.ode_data_norm.append([np.sqrt(ans_x[0]**2+ans_y[0]**2+ans_z[0]**2), np.sqrt(ans_x[1]**2+ans_y[1]**2+ans_z[1]**2)])
        import csv
        base_data = self.full_dataset["data"][0][:, :3]
        with open("./"+self.data_name+"/mass_redu-x.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            for step, row in enumerate(zip(base_data, self.ode_data_x)):
                row = [float(item) for sublist in row for item in sublist]
                csv_writer.writerow(row)
            f.close()
        base_data = self.full_dataset["data"][0][:, :3]
        with open("./" + self.data_name + "/mass_redu-y.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            for step, row in enumerate(zip(base_data, self.ode_data_y)):
                row = [float(item) for sublist in row for item in sublist]
                csv_writer.writerow(row)
            f.close()
        base_data = self.full_dataset["data"][0][:, :3]
        with open("./" + self.data_name + "/mass_redu-z.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            for step, row in enumerate(zip(base_data, self.ode_data_z)):
                row = [float(item) for sublist in row for item in sublist]
                csv_writer.writerow(row)
            f.close()
        base_data = self.full_dataset["data"][0][:, :3]
        with open("./" + self.data_name + "/mass_redu-norm.csv", "w", newline="") as f:
            csv_writer = csv.writer(f)
            for step, row in enumerate(zip(base_data, self.ode_data_norm)):
                row = [float(item) for sublist in row for item in sublist]
                csv_writer.writerow(row)
            f.close()

            
            


        # TODO ode model
        pass

    def analysis_ensemble_kalman_filter(self):
        self.dataset = self.dataset
        print("Here is DataManager.Analysis.DeepAnalysis.analysis_ensemble_kalman_filter!")

        # TODO EnKF model
        # TODO save analyzed data
        pass


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


a = DeepAnalysis("Mie02")


Console()
