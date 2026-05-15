"""
Created on Mon Jun 19 13:22:02 2023

@author: Paul Dräger

Object-oriented framework for the analysis of PRIME simulation data.
Currently supports fluid, ellipsoid, and bubble data.
"""

import os
import re
import shutil
import time
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from ..hpcsync.ssh_functions import ssh_download_sim, ssh_get_files


class sim:
    """Simulation object containing data for a single run."""

    def __init__(
        self,
        path,
        newbinary=True,
        onlyInfo=False,
        force_reading=True,
        ssh_pass="none",
        ssh_path="local",
        ssh_download_fluid=False,
        ssh_update=True,
        with_coll_force=False,
        reduce_tell=1,
        reduce_tbub=1,
    ):
        # path         - path to the simulation result directory
        # reduce_tell  - every nth time step from ellipsoid monitor files is saved
        # reduce_tbub  - every nth time step from bubble monitor files is saved
        # newbinary    - if True, new numpy binary files are generated
        # onlyInfo     - object contains only minimal information

        rpath = ""

        if not ssh_pass == "local":
            path, rpath = ssh_download_sim(
                path,
                ssh_pass,
                ssh_download_fluid,
                ssh_update,
                bub_fp=False,
            )

        self.ssh_fluid = ssh_download_fluid
        self.ssh_pass = ssh_pass

        print("Creating Simulation-Object ...")
        print("1/3 Reading infos.dat")

        T = pd.read_csv(path + "/rst_info.dat", sep=r"\s+")

        self.with_coll_force = with_coll_force
        self.ecount = T["ell_counter"].values
        self.bcount = T["bub_counter"].values

        if self.ecount[-1] > 0:
            with_ell = True
        if self.bcount[-1] > 0:
            with_bub = True

        self.onlyInfo = onlyInfo
        self.resultpath = path
        self.rpath = rpath

        self.tf = T["t"].values
        self.tei = len(self.tf)
        self.tee = max(self.tf)
        self.tfa = min(self.tf)

        nt = T["nt"].values
        self.nt = nt

        sim_counter = T["counter"].values
        self.nf = sim_counter[-1]

        latest_out_prime_file = find_latest_out_prime_file(self.resultpath)
        self.out_prime = latest_out_prime_file

        cfl_con_values = extract_cfl_values(self.resultpath)
        cfl_diff_values = extract_cfl_diff_values(self.resultpath)

        self.cfl = cfl_con_values
        self.cfl_diff = cfl_diff_values

        if latest_out_prime_file and not onlyInfo:
            try:
                self.vis = extract_value_from_out(latest_out_prime_file, "vis")
                self.rhof = extract_value_from_out(latest_out_prime_file, "rho_f")
                self.gx = extract_value_from_out(latest_out_prime_file, "grx")
                self.gy = extract_value_from_out(latest_out_prime_file, "gry")
                self.gz = extract_value_from_out(latest_out_prime_file, "grz")
                self.g = np.sqrt(self.gx**2 + self.gy**2 + self.gz**2)
            except:
                print(
                    "ATTENTION: got a problem with reading in sim. param. "
                    "-> no phyiscal parameters (sim) available!"
                )

            try:
                if with_bub:
                    r_out = extract_value_from_out(latest_out_prime_file, "radius")
                    d_out = 2 * r_out
                    self.d_out = d_out

                    
                    Eo = extract_value_from_out(latest_out_prime_file, "Eo")
                    Ar = extract_value_from_out(latest_out_prime_file, "Ar")
                    self.sig = Ar * self.rhof * self.vis**2 / (Eo * d_out)
            except:
                print(
                    "ATTENTION: got a problem with reading in bub. param. "
                    "-> no phyiscal parameters (bubble) available!"
                )

        if with_ell and not onlyInfo:
            print("2/3a Reading input_ellipsoid_data.dat")

            Elli = read_input(path + "/input_ellipsoid_data.dat")
            self.eNr = len(Elli)

            print(f"Number of Ellipsoids: {self.eNr}")

            d = Elli.iloc[:, 2].values
            self.ed = d * 2

        if with_bub and not onlyInfo:
            print("2/3b Reading input_bubble_data.dat")

            Bub = read_input(path + "/input_bubble_data.dat")
            self.bNr = len(Bub)

            print(f"Number of Bubbles: {self.bNr}")

            req = Bub.iloc[:, 3].values
            self.bd = req[0] * 2

        Gpath = path + "/gridbinary"
        self.Gpath = Gpath

        print("3/3 Grid data")
        if not os.path.exists(Gpath + "/gridinfo.npz"):
            try:
                os.makedirs(Gpath)
            except FileExistsError:
                pass

            print("Reading Grid Data (fluid/x.dat) \nmight take some time for large grids...")

            Tx = pd.read_csv(
                path + "/fluid/x.dat",
                sep=r"\s+",
                engine="python",
                header=None,
            )
            Ty = pd.read_csv(
                path + "/fluid/y.dat",
                sep=r"\s+",
                engine="python",
                header=None,
            )
            Tz = pd.read_csv(
                path + "/fluid/z.dat",
                sep=r"\s+",
                engine="python",
                header=None,
            )

            # Every grid dimension has two coordinate arrays: centered and staggered.
            xce = Tx.iloc[:, 0].values
            xst = Tx.iloc[:, 1].values

            yce = Ty.iloc[:, 0].values
            yst = Ty.iloc[:, 1].values

            zce = Tz.iloc[:, 0].values
            zst = Tz.iloc[:, 1].values

            dx = xst[1] - xst[0]

            nx = len(xst)
            ny = len(yst)
            nz = len(zst)

            # Domain size from .dat; only works this way for periodic domains.
            Lx = max(xst)
            Ly = max(yst)
            Lz = max(zst)

            print("Saving Meshgrids")

            Xp, Yp, Zp = np.meshgrid(xce, yce, zce)
            np.save(Gpath + "/Xp.npy", Xp)
            np.save(Gpath + "/Yp.npy", Yp)
            np.save(Gpath + "/Zp.npy", Zp)

            Xu, Yu, Zu = np.meshgrid(xst, yce, zce)
            np.save(Gpath + "/Yu.npy", Yu)
            np.save(Gpath + "/Xu.npy", Xu)
            np.save(Gpath + "/Zu.npy", Zu)

            Xv, Yv, Zv = np.meshgrid(xce, yst, zce)
            np.save(Gpath + "/Xv.npy", Xv)
            np.save(Gpath + "/Yv.npy", Yv)
            np.save(Gpath + "/Zv.npy", Zv)

            Xw, Yw, Zw = np.meshgrid(xce, yce, zst)
            np.save(Gpath + "/Xw.npy", Xw)
            np.save(Gpath + "/Yw.npy", Yw)
            np.save(Gpath + "/Zw.npy", Zw)

            np.savez(
                Gpath + "/gridinfo.npz",
                dx=dx,
                nx=nx,
                ny=ny,
                nz=nz,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                xce=xce,
                xst=xst,
                yce=yce,
                yst=yst,
                zce=zce,
                zst=zst,
            )

        grid = np.load(Gpath + "/gridinfo.npz")

        self.dx = grid["dx"]

        self.nx = grid["nx"]
        self.ny = grid["ny"]
        self.nz = grid["nz"]

        self.Lx = grid["Lx"]
        self.Ly = grid["Ly"]
        self.Lz = grid["Lz"]
        print("please be careful with sim.Lx, sim.Ly, sim.Lz if the domain is not periodic!")

        try:
            self.Vges = self.Lx * self.Ly * self.Lz
        except:
            pass

        self.xst = grid["xst"]
        self.xce = grid["xce"]

        self.yst = grid["yst"]
        self.yce = grid["yce"]

        self.zst = grid["zst"]
        self.zce = grid["zce"]

        self.xp = self.xst
        self.yp = self.yst
        self.zp = self.zst

        if with_bub and newbinary and not onlyInfo:
            print("(+) Translate Bubble data to numpy binary (deactivate newbinary to save time if already done once)")

            Epath = self.resultpath + "/bubble"
            Bpath = Epath + "/bub_binary"

            try:
                os.makedirs(Bpath)
            except FileExistsError:
                pass

            boolpath = Bpath + "/writing.dat"

            if not read_bool(boolpath, force_reading):
                write_bool(boolpath, True, force_reading)

                for i in range(1, self.bNr + 1):
                    print(f"Bubble {i}/{self.bNr}")

                    file_path = f"{Epath}/bub_{i:03d}_monitor.dat"

                    E = pd.read_csv(
                        file_path,
                        sep=r"\s+",
                        header=None,
                        names=[
                            "i",
                            "t",
                            "x1",
                            "x2",
                            "x3",
                            "u1",
                            "u2",
                            "u3",
                        ],
                        usecols=range(8),
                        dtype=str,
                    )

                    for col in E.columns:
                        E[col] = pd.to_numeric(E[col], errors="coerce")

                    Etxu = E.iloc[:, 1:8].values
                    Efile = os.path.join(Bpath, f"bub_{i}.npy")

                    np.save(Efile, Etxu[::reduce_tbub])

                    print("saved binary file!")

                write_bool(boolpath, False, force_reading)
            else:
                print("Externen Einlesevorgang erkannt: Warten auf Fertigstellung...")

                while read_bool(boolpath, force_reading):
                    time.sleep(1)

                print("Fertigstellung erkannt.")

        if with_ell and newbinary and not onlyInfo:
            print("(+) Translate Ellipsoid data to numpy binary (deactivate newbinary to save time if already done once)")

            Epath = self.resultpath + "/ellipsoid"
            Bpath = Epath + "/ell_binary"

            try:
                os.makedirs(Bpath)
            except FileExistsError:
                pass

            boolpath = Bpath + "/writing.dat"

            if not read_bool(boolpath, force_reading):
                write_bool(boolpath, True, force_reading)

                for i in range(1, self.eNr + 1):
                    print("Ellipsoid " + format(i, "d") + "/" + format(self.eNr, "d"))

                    if with_coll_force:
                        E = pd.read_csv(
                            Epath + "/ell_" + format(i, "03d") + "_monitor.dat",
                            sep=r"\s+",
                            header=None,
                            names=[
                                "i",
                                "t",
                                "x1",
                                "x2",
                                "x3",
                                "u1",
                                "u2",
                                "u3",
                                "o1",
                                "o2",
                                "o3",
                                "f1",
                                "f2",
                                "f3",
                            ],
                            usecols=range(14),
                        )
                        Etxu = E.iloc[:, 1:14].values
                    else:
                        E = pd.read_csv(
                            Epath + "/ell_" + format(i, "03d") + "_monitor.dat",
                            sep=r"\s+",
                            header=None,
                            names=[
                                "i",
                                "t",
                                "x1",
                                "x2",
                                "x3",
                                "u1",
                                "u2",
                                "u3",
                                "o1",
                                "o2",
                                "o3",
                            ],
                            usecols=range(11),
                        )
                        Etxu = E.iloc[:, 1:11].values

                    Efile = Bpath + "/ell_" + format(i, "d") + ".npy"
                    np.save(Efile, Etxu[::reduce_tell])

                    print("saved binary file!")

                write_bool(boolpath, False, force_reading)
            else:
                print("externen Einlesevorgang erkannt: Warten auf Fertigstellung...")

                while read_bool(boolpath):
                    time.sleep(1)

                print("Fertigstellung erkannt.")

        self.force_reading = force_reading

        print("Done")

    @property
    def Xp(self):
        return np.load(self.Gpath + "/Xp.npy")

    @property
    def Yp(self):
        return np.load(self.Gpath + "/Yp.npy")

    @property
    def Zp(self):
        return np.load(self.Gpath + "/Zp.npy")

    @property
    def Xu(self):
        return np.load(self.Gpath + "/Xu.npy")

    @property
    def Yu(self):
        return np.load(self.Gpath + "/Yu.npy")

    @property
    def Zu(self):
        return np.load(self.Gpath + "/Zu.npy")

    @property
    def Xv(self):
        return np.load(self.Gpath + "/Xv.npy")

    @property
    def Yv(self):
        return np.load(self.Gpath + "/Yv.npy")

    @property
    def Zv(self):
        return np.load(self.Gpath + "/Zv.npy")

    @property
    def Xw(self):
        return np.load(self.Gpath + "/Xw.npy")

    @property
    def Yw(self):
        return np.load(self.Gpath + "/Yw.npy")

    @property
    def Zw(self):
        return np.load(self.Gpath + "/Zw.npy")

    @property
    def dt(self):
        if (self.nt[-2] - self.nt[-3])>1:
            print(
                "(requested dt): time step might deviate if adaptive!"
            )

        dt = (self.tf[-2] - self.tf[-3]) / (self.nt[-2] - self.nt[-3])

        return dt

    def f(self, i):
        print("Reading Fluid Data of " + format(i, "d") + "th datapoint")
        flfield = fluid(self, i)
        print("Done")
        return flfield

    def e(self, i):
        elli = ellipsoid(self, i)
        return elli

    def b(self, i, check=False, old_cnm=False):
        bub = bubble(self, i, check=check, old_cnm=old_cnm)
        return bub


class fluid:
    """Fluid-field data for one time index."""

    def __init__(self, sim, i):
        path = sim.resultpath + "/fluid"

        self.fpath = path
        self.rpath = sim.resultpath

        fluid_path = path + "/uvw_" + format(i, "06d") + ".bin"

        if not sim.ssh_pass == "local":
            if not os.path.exists(fluid_path):
                from ..hpcsync.ssh_functions import ssh_get_fluid

                print("Download Fluid data via ssh...")
                ssh_get_fluid(i, sim.ssh_pass, sim.resultpath, sim.rpath)
                print("...done")

        with open(path + "/uvw_" + format(i, "06d") + ".bin", "rb") as fin:
            vel = np.fromfile(fin, dtype=">d")
            vel = np.delete(vel, 0)
            vel = np.reshape(vel, [3, sim.nx, sim.ny, sim.nz], order="f")
            vel = np.transpose(vel, (2, 1, 3, 0))

        with open(path + "/pre_" + format(i, "06d") + ".bin", "rb") as fin:
            pel = np.fromfile(fin, dtype=">d")
            pel = np.delete(pel, 0)
            pel = np.reshape(pel, [sim.nx, sim.ny, sim.nz], order="f")
            self.p = np.transpose(pel, (1, 0, 2))

        self.u = vel[:, :, :, 0]
        self.v = vel[:, :, :, 1]
        self.w = vel[:, :, :, 2]

        self.i = i

    def save_slice(self, Z):
        print("Save Slice in the X-Y Plane of fluid data")

        Spath = self.rpath + "/fl_slices"

        try:
            os.makedirs(Spath)
        except FileExistsError:
            pass

        np.save(Spath + "/u_" + format(self.i, "d") + ".npy", self.u[:, :, Z])
        np.save(Spath + "/v_" + format(self.i, "d") + ".npy", self.v[:, :, Z])
        np.save(Spath + "/w_" + format(self.i, "d") + ".npy", self.w[:, :, Z])
        np.save(Spath + "/p_" + format(self.i, "d") + ".npy", self.p[:, :, Z])

        print("Done")


class bubble:
    """Bubble monitor data."""

    def __init__(self, sim, i, check=False, old_cnm=False):
        path = sim.resultpath + "/bubble/bub_binary/bub_" + format(i, "d") + ".npy"

        self.path_sh = sim.resultpath + "/bubble/bub_binary/bub_" + format(i, "d") + "_sh.npy"
        self.spath = sim.resultpath

        boolpath = sim.resultpath + "/bubble/bub_binary/writing.dat"

        if old_cnm:
            self.cnm_path = sim.resultpath + f"/bubble/bub_{i:03d}_cnm.bin"
        else:
            self.cnm_path = sim.resultpath + f"/bubble/bub_{i:03d}_cnm_pnt.bin"

        if read_bool(boolpath, sim.force_reading):
            print("externen Einlesevorgang erkannt: Warten auf Fertigstellung...")

            while read_bool(boolpath):
                time.sleep(1, False)

            print("Fertigstellung erkannt.")

        self.i = i

        if not sim.onlyInfo:
            B = np.load(path, allow_pickle=True)

            self.nb = sim.bNr

            self.t = B[:, 0]

            self.x = B[:, 1]
            self.y = B[:, 2]
            self.z = B[:, 3]

            self.u = B[:, 4]
            self.v = B[:, 5]
            self.w = B[:, 6]

            self.ui = interpolate.interp1d(self.t, self.u, fill_value="extrapolate")
            self.vi = interpolate.interp1d(self.t, self.v, fill_value="extrapolate")
            self.wi = interpolate.interp1d(self.t, self.w, fill_value="extrapolate")

            self.uges = np.sqrt(self.u**2 + self.v**2 + self.w**2)

            try:
                self.Re = self.uges * sim.vis
                self.Rei = interpolate.interp1d(self.t, self.Re, fill_value="extrapolate")
            except:
                pass

    def Xi(self, ti, I=False):
        x = np.atleast_1d(self.xi(ti, I=I))
        y = np.atleast_1d(self.yi(ti, I=I))
        z = np.atleast_1d(self.zi(ti, I=I))

        X = np.column_stack([x, y, z])

        if X.shape[0] == 1:
            X = X[0]

        return X

    def Ui(self, ti, I=False):
        x = np.atleast_1d(self.ui(ti, I=I))
        y = np.atleast_1d(self.vi(ti, I=I))
        z = np.atleast_1d(self.wi(ti, I=I))

        X = np.column_stack([x, y, z])

        if X.shape[0] == 1:
            X = X[0]

        return X

    def get_cnm(self, NF=8):
        with open(self.cnm_path, "rb") as fin:
            A = np.fromfile(fin, dtype=np.float64, offset=4)

        NL1 = (NF + 1) * (NF + 2) / 2
        NL = NL1 * 2

        NL = round(NL) + 2
        Ncof = round(NL * 3)

        lA = len(A)
        maxcoef = lA % Ncof
        A = A[0:-maxcoef]

        cnm = np.reshape(A, [-1, Ncof])

        cnmx = cnm[:, 0:NL]
        cnmy = cnm[:, NL : (2 * NL)]
        cnmz = cnm[:, (2 * NL) : (3 * NL)]

        NLh = round(NL / 2)

        anmxR = cnmx[:, 0:NLh]
        anmxI = cnmx[:, NLh : (2 * NLh)]

        anmyR = cnmy[:, 0:NLh]
        anmyI = cnmy[:, NLh : (2 * NLh)]

        anmzR = cnmz[:, 0:NLh]
        anmzI = cnmz[:, NLh : (2 * NLh)]

        A = np.concatenate(
            (
                anmxR[:, 1:],
                anmxI[:, 1:],
                anmyR[:, 1:],
                anmyI[:, 1:],
                anmzR[:, 1:],
                anmzI[:, 1:],
            ),
            axis=1,
        )

        return A


class ellipsoid:
    """Ellipsoid monitor data."""

    def __init__(self, sim, i):
        path = sim.resultpath + "/ellipsoid/ell_binary/ell_" + format(i, "d") + ".npy"
        boolpath = sim.resultpath + "/ellipsoid/ell_binary/writing.dat"

        if read_bool(boolpath, sim.force_reading):
            print("externen Einlesevorgang erkannt: Warten auf Fertigstellung...")

            while read_bool(boolpath, False):
                time.sleep(1)

            print("Fertigstellung erkannt.")

        E = np.load(path)

        self.id = i
        self.with_coll = sim.with_coll_force
        self.d = sim.ed[i - 1]

        self.t = E[:, 0]

        self.x = E[:, 1]
        self.y = E[:, 2]
        self.z = E[:, 3]

        self.u = E[:, 4]
        self.v = E[:, 5]
        self.w = E[:, 6]

        if self.with_coll:
            self.cx = E[:, 10]
            self.cy = E[:, 11]
            self.cz = E[:, 12]

        self.Lx = sim.Lx
        self.Ly = sim.Ly
        self.Lz = sim.Lz

        self.ui = interpolate.interp1d(self.t, self.u, fill_value="extrapolate")
        self.vi = interpolate.interp1d(self.t, self.v, fill_value="extrapolate")
        self.wi = interpolate.interp1d(self.t, self.w, fill_value="extrapolate")

        self.uges = np.sqrt(self.u**2 + self.v**2 + self.w**2)

        try:
            self.Re = self.uges * sim.vis
            self.Rei = interpolate.interp1d(self.t, self.Re, fill_value="extrapolate")
        except:
            pass


# Helper functions


def extract_numeric(filename, out=False, type="int"):
    if out:
        print('Ermittle Index aus " ' + filename + ' "')

    parts = filename.split("_")
    numeric_part = "".join(
        filter(
            lambda x: x.isdigit() or x == "." or "e" in x or "-" in x,
            parts[-1],
        )
    )

    try:
        if type == "int":
            numeric_value = int(numeric_part)
        if type == "float":
            numeric_value = float(numeric_part)

        if out:
            print("-> " + numeric_part)

        return numeric_value
    except ValueError:
        if out:
            print("no content")

        return 10000


def write_bool(file_path, value, fr):
    if not fr:
        with open(file_path, "w") as file:
            file.write("1" if value else "0")


def read_bool(file_path, fr):
    if fr:
        return False
    else:
        try:
            with open(file_path, "r") as file:
                content = file.read()
                bool = content.strip() == "1"

                if bool:
                    if check_recent(file_path):
                        print(
                            "Einlese-Markierung liegt mehr als 10 min zurück "
                            "und wird Ignoriert"
                        )
                        bool = False
                        write_bool(file_path, False, False)

                return bool
        except FileNotFoundError:
            return False


def check_recent(file_path):
    try:
        last_modification_time = os.path.getmtime(file_path)
        current_time = time.time()

        return current_time - last_modification_time > 600
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking file modification time: {e}")
        return False


def extract_cfl_con_values(filename):
    values = []
    pattern = r"cfl_con\s*=\s*([+-]?\d+\.\d+E[+-]?\d+)"

    with open(filename, "r") as file:
        for line in file:
            match = re.search(pattern, line)

            if match:
                values.append(float(match.group(1)))

    return np.array(values)


def extract_cfl_values(folder_path):
    cfl_values = []
    file_pattern = re.compile(r"out_prime(?:_(\d+)_(\d+))?\.log")

    files = []

    for filename in os.listdir(folder_path):
        match = file_pattern.match(filename)

        if match:
            num = int(match.group(1)) if match.group(1) is not None else -1
            files.append((num, filename))

    files.sort()

    cfl_pattern = re.compile(r"cfl_con =\s+([\d\.E\+\-]+)")
    cfl_values = []

    for _, filename in files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as file:
#            print(f"CFL: searching {file_path}")

            for line in file:
                match = cfl_pattern.search(line)

                if match:
                    cfl_values.append(float(match.group(1)))

    return cfl_values


def extract_cfl_diff_values(folder_path):
    cfl_values = []
    file_pattern = re.compile(r"out_prime(?:_(\d+)_(\d+))?\.log")

    files = []

    for filename in os.listdir(folder_path):
        match = file_pattern.match(filename)

        if match:
            num = int(match.group(1)) if match.group(1) is not None else -1
            files.append((num, filename))

    files.sort()

    cfl_pattern = re.compile(r"cfl_dif =\s+([\d\.E\+\-]+)")
    cfl_values = []

    for _, filename in files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r") as file:
#            print(f"CFL: searching {file_path}")

            for line in file:
                match = cfl_pattern.search(line)

                if match:
                    cfl_values.append(float(match.group(1)))

    return cfl_values


def extract_value_from_out(file_path, name):
    pattern = re.compile(rf"{name}\s*=\s*([^ ]+)")

    if name == "radius":
        pattern = re.compile(
            r"initial bubble radius is\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )

    with open(file_path, "r") as file:
        for line in file:
            match = pattern.search(line)

            if match:
                #print(match.group(1))
                number_str = match.group(1).replace("d", "e").replace("D", "e")
                return float(number_str)

    print(f"VALUE NOT FOUND IN OUT FILE: {name}")

    return 0.5


def find_latest_out_prime_file(directory):
    out_prime_files = []
    mons_out = "out_prime.log"

    for filename in os.listdir(directory):
        if filename.startswith("out_prime") and filename.endswith(".log"):
            if filename == mons_out:
                return os.path.join(directory, mons_out)

            try:
                numerical_value = int(
                    re.findall(r"out_prime_(\d+)_\d+\.log", filename)[0]
                )
                out_prime_files.append((numerical_value, filename))
            except IndexError:
                continue

    if out_prime_files:
        latest_file = min(out_prime_files, key=lambda x: x[0])[1]
#        print(latest_file)
        return os.path.join(directory, latest_file)

    return None


def read_input(path):
    """
    Read input table from input_ellipsoid_data.dat or input_bubble_data.dat.

    Stops reading at the first empty line.
    """

    fname = path

    with open(fname, "r") as f:
        lines = f.readlines()

    try:
        icut = next(i for i, line in enumerate(lines) if not line.strip())
    except StopIteration:
        icut = len(lines)

    data = "".join(lines[:icut])

    Elli = pd.read_csv(
        StringIO(data),
        delimiter=",",
        header=None,
    )

    return Elli
