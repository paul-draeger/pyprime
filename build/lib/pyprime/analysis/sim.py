"""
Created on Mon Jun 19 13:22:02 2023

@author: Paul Dräger

"""
#object oriented framework for the analysis of PRIME simulation Data
#currently fluid & ellipsoid data


import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import os
import shutil
import time
from .ssh_functions import ssh_get_files
import re
from .physics import prop


class sim:
    # sim-Object contains data related to a single simulation
    def __init__(self, path, reduce_tell=3,reduce_tbub=1, bNr=1, d_b=-1, read_ell=True, ell_type='bubble', newbinary=1, newbubble=1, onlyInfo=False, force_reading=False, no_cmn=False,ssh_pass='none',ssh_path='local', ssh_download_fluid=False, ssh_update=True, get_bub_fp=False,with_coll_force=False,bnt_plot=50,eNr=0,noslip=False, nogridconv=False, extra_var_bub=False):

        # path        - path to the simulation result dir.
        # reduce_tell - Every nth time step from ellipsoid monitor files is
        #               saved in numpy binary.
        # newbinary   - if 1, new numpy binary files are generated from ellipsoid
        #               monitor files (takes longer).
        # newbubble   - if 1, new numpy binary files are generated from bubble
        #               monitor files (takes longer).
        # onlyInfo    - object contains only minimal information
        rpath = ''
        if not ssh_pass=='local':
            from .ssh_functions import ssh_download_sim
            #if newbubble==1 or newbinary==1:
                #print('HEY')
            path, rpath = ssh_download_sim(path,ssh_pass,ssh_download_fluid, ssh_update, bub_fp = get_bub_fp)

        self.ssh_fluid = ssh_download_fluid
        self.ssh_pass = ssh_pass
        print('Creating Simulation-Object ...')
        #Read info.dat File
        print('1/3 Reading infos.dat')
        T = pd.read_csv(path+'/rst_info.dat', sep=r'\s+')
        #general information
        with_ell=False
        wit_bub =False
        self.with_coll_force = with_coll_force
        self.ecount = T['ell_counter'].values
        if self.ecount[-1]>0 and read_ell:
            with_ell=True
            self.ell_type = ell_type
        self.bcount = T['bub_counter'].values
        with_bub=False
        if self.bcount[-1]>0: with_bub=True

        self.extra_bub_var = extra_var_bub
        self.onlyInfo = onlyInfo
        self.resultpath = path
        self.rpath = rpath
        self.tf = T['t'].values
        self.tei = len(self.tf)
        self.tee = max(self.tf)
        self.tfa = min(self.tf)
        self.eNr = eNr
        #self.dt = self.tf[1]-self.tf[0]
        nt = T['nt'].values
        self.nt = nt
        sim_counter = T['counter'].values
        self.Nf = sim_counter[-1]

        #Read in the out_file
        latest_out_prime_file = find_latest_out_prime_file(self.resultpath)
        self.out_prime = latest_out_prime_file
        cfl_con_values = extract_cfl_values(self.resultpath)
        cfl_diff_values = extract_cfl_diff_values(self.resultpath)
        self.cfl = cfl_con_values
        self.cfl_diff = cfl_diff_values
        if latest_out_prime_file and not onlyInfo:
            # Extract CFL values
            #cfl_con_values = extract_cfl_con_values(latest_out_prime_file)

            # Extract Sim. Parameters
            if ell_type=='bubble':
                try:
                    self.vis = extract_value_from_out(latest_out_prime_file,'vis')
                    self.rhof = extract_value_from_out(latest_out_prime_file,'rho_f')
                    r_out = extract_value_from_out(latest_out_prime_file,'radius')
                    d_out = 2 * r_out
                    self.d_out = d_out
                    print(f"Durchmesser: {d_out:.2f}")
                    Eo = extract_value_from_out(latest_out_prime_file,'Eo')
                    Ar = extract_value_from_out(latest_out_prime_file,'Ar')
                    self.gx = extract_value_from_out(latest_out_prime_file,'grx')
                    self.gy = extract_value_from_out(latest_out_prime_file,'gry')
                    self.gz = extract_value_from_out(latest_out_prime_file,'grz')
                    self.g = np.sqrt( self.gx**2 + self.gy**2 + self.gz**2 )

                    self.sig = Ar * self.rhof * self.vis**2 / ( Eo * d_out )
                    drho = Eo * self.sig / ( self.g * d_out**2 )
                    self.rhog = self.rhof - drho

                    self.props = prop(d=self.d_out,g=self.g,vis=self.vis,rhof=self.rhof,rhog=self.rhog,sig=self.sig)
                except:
                    print('ATTENTION: got a problem with reading in sim. param. -> no props object created! -> using standard prop object')
            else:
                self.vis = 3.16955e-02
                self.rhof = 1
                self.rhog = 1e-03
                self.g = 9.81
                self.sigma = 72.75


        try:
            self.dt = (self.tf[-2]-self.tf[-3]) / (sim_counter[-2] - sim_counter[-3])
            self.Nt = len(self.tf)
            label = os.path.dirname(path)
            if not label[:2].isdigit(): label = os.path.basename(path)
            self.dirname = label
            self.label = label[3:]

            self.e_g = extract_numeric(self.label)/100
        except Exception as error:
            print(error)

        #Fluid
        file_path = self.resultpath + '/input.dat'
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                vis_line = lines[9].strip()
                vis = extract_numeric(vis_line,type='float')
        except Exception as error23c:
            print(error23c)
            vis = 1


        self.polyd = np.array( [1.022, 1.573, 2.125] )

        #read ellipsoid_input
        if with_ell and ell_type=='bubble' and not onlyInfo:
            print('2/3a Reading input_ellipsoid_data.dat')
            Elli = pd.read_csv(path+'/input_ellipsoid_data.dat',delimiter = ',', header=None)
            self.Elli = Elli
            self.eNr = len(Elli)
            d = Elli.iloc[:, 2].values
            self.d = d * 2
            self.du, protoE = np.unique(self.d, return_index=True)
            self.protoE = protoE + 1
            self.gkN = np.size(self.du)
            self.n1 = 0
            self.n2 = 0
            self.n3 = 0
            self.d1 = 0
            self.d2 = 0
            self.d3 = 0
            self.Vg = np.sum(self.d**3/6*np.pi)
            if len(self.du)>3:
                self.continous = True
            else:
                self.continous = False
                if self.gkN==1:
                    indf = np.abs(self.du[0]-np.array([1,1.5,2]))
                    gk = np.argmin(indf)+1
                    if gk==1:
                        self.d1 = self.du[0]
                        self.n1 = self.eNr
                    if gk==2:
                        self.d2 = self.du[0]
                        self.n2 = self.eNr
                    if gk==3:
                        self.d3 = self.du[0]
                        self.n3 = self.eNr
                if self.gkN==2:
                    self.n1 = np.size(np.nonzero(self.d==self.du[0]))
                    self.n3 = np.size(np.nonzero(self.d==self.du[1]))
                    self.d1 = self.du[0]
                    self.d3 = self.du[1]
                if self.gkN==3:
                    self.n1 = np.size(np.nonzero(self.d==self.du[0]))
                    self.n2 = np.size(np.nonzero(self.d==self.du[1]))
                    self.n3 = np.size(np.nonzero(self.d==self.du[2]))
                    self.d1 = self.du[0]
                    self.d2 = self.du[1]
                    self.d3 = self.du[2]



            self.GKi = 1*(self.d==self.d1) + 2*(self.d==self.d2) + 3*(self.d==self.d3)
            self.GKu = np.unique(self.GKi)

            self.gkn = np.array([np.sum(self.GKi==1), np.sum(self.GKi==2), np.sum(self.GKi==3)])
            self.gknu = self.gkn[self.gkn>0]

        if with_bub:
            print('2/3b Reading input_bubble_data.dat')
            Bub = pd.read_csv(path+'/input_bubble_data.dat',delimiter = ',', header=None)
            end_not_found = True
            ibub=0
            lenb = len(Bub)
            while 1==1:
                ibub+=1
                bubid = Bub.iloc[ibub-1, 0]
                if bubid=='bubbleID':
                    self.bNr = ibub-1
                    break
                if ibub>=lenb:
                    self.bNr = ibub
                    break
            if bNr!=-1:
                self.bNr = bNr
            print(f'Anzahl Blasen: {self.bNr}')
            req = Bub.iloc[:, 3].values
            nlb = Bub.iloc[:, 7].values
            self.d_b = req[0] * 2
            self.bRI = req
            self.nlRI = nlb

            print('2/3c Reading input.dat')
            file_path = self.resultpath + '/input.dat'
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    bnt_line = lines[56].strip()
                    bnt_plt = int(extract_numeric(bnt_line))
            except Exception as error23c:
                print(error23c)
                bnt_plt = 1
            print(f'Bubble fp time step: {bnt_plt}')
            self.bnt_plot = bnt_plot#40#40 #bnt_plt #2#20#bnt_plt


        if with_ell and ell_type=='electrode':
            file_path = self.resultpath + '/input.dat'
            try:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    elec_line = lines[74].strip()
                    elec_plt = int(extract_numeric(elec_line))
            except Exception as error23c:
                print(error23c)
                elec_plt = 1
            print(f'Electrode fp time step: {elec_plt}')
            self.elec_plot = elec_plt

        Gpath = path + '/gridbinary'
        self.Gpath = Gpath

        if not os.path.exists(Gpath+'/gridinfo.npz') and not nogridconv:
            try:
                os.makedirs(Gpath)
            except FileExistsError:
                pass

            print('Reading Grid Data (fluid/x.dat)')
            Tx = pd.read_csv(path+'/fluid/x.dat', sep=r'\s+',engine="python",header=None)
            Ty = pd.read_csv(path+'/fluid/y.dat', sep=r'\s+',engine="python",header=None)
            Tz = pd.read_csv(path+'/fluid/z.dat', sep=r'\s+',engine="python",header=None)

            #every Grid dimension has two coordinate arrays (staggered grid)
            xce = Tx.iloc[:, 0].values
            xst = Tx.iloc[:, 1].values

            yce = Ty.iloc[:, 0].values
            yst = Ty.iloc[:, 1].values

            zce = Tz.iloc[:, 0].values
            zst = Tz.iloc[:, 1].values


            # #Numerical parameters
            dx = xst[1]-xst[0]

            #Domain
            nx = len(xst)
            ny = len(yst)
            nz = len(zst)
            #Domain-Size form .dat - only works this way for periodic domains
            Lx = max(xst)
            Ly = max(yst)
            Lz = max(zst)
            print('Saving Meshgrids')
            Xp, Yp, Zp = np.meshgrid(xce,yce,zce)
            np.save(Gpath+'/Xp.npy',Xp)
            np.save(Gpath+'/Yp.npy',Yp)
            np.save(Gpath+'/Zp.npy',Zp)
            Xu, Yu, Zu = np.meshgrid(xst,yce,zce)
            np.save(Gpath+'/Yu.npy',Yu)
            np.save(Gpath+'/Xu.npy',Xu)
            np.save(Gpath+'/Zu.npy',Zu)
            Xv, Yv, Zv = np.meshgrid(xce,yst,zce)
            np.save(Gpath+'/Xv.npy',Xv)
            np.save(Gpath+'/Yv.npy',Yv)
            np.save(Gpath+'/Zv.npy',Zv)
            Xw, Yw, Zw = np.meshgrid(xce,yce,zst)
            np.save(Gpath+'/Xw.npy',Xw)
            np.save(Gpath+'/Yw.npy',Yw)
            np.save(Gpath+'/Zw.npy',Zw)

            np.savez(Gpath+'/gridinfo.npz',dx=dx,nx=nx,ny=ny,nz=nz,Lx=Lx,Ly=Ly,Lz=Lz,xce=xce,xst=xst,yce=yce,yst=yst,zce=zce,zst=zst)

        grid = np.load(Gpath+'/gridinfo.npz')
        self.dx = grid['dx']
        if with_ell and ell_type=='bubble': self.dcorr = self.d + 0.6*self.dx
        self.nx = grid['nx']
        self.ny = grid['ny']
        self.nz = grid['nz']

        self.Lx = grid['Lx']
        self.Ly = grid['Ly']
        self.Lz = grid['Lz']
        try:
            self.Vges = self.Lx*self.Ly*self.Lz
        except:
            pass
        self.xst = grid['xst']
        self.xce = grid['xce']

        self.yst = grid['yst']
        self.yce = grid['yce']

        self.zst = grid['zst']
        self.zce = grid['zce']

        self.xp = self.xst
        self.yp = self.yst
        self.zp = self.zst

        if with_bub and newbubble==1 and not onlyInfo:
            print('Translate Bubble Data to Numpy Binary (takes the longest time)')
            Epath = self.resultpath + '/bubble'
            Bpath = Epath + '/bub_binary'
            try:
                os.makedirs(Bpath)
            except FileExistsError:
                pass
            boolpath = Bpath+'/writing.dat'

            if not read_bool(boolpath, force_reading):
                write_bool(boolpath, True, force_reading)
                for i in range(1, self.bNr + 1):
                    print(f'Bubble {i+1}/{self.bNr+1}')
                    file_path = f"{Epath}/bub_{i:03d}_monitor.dat"

                    if extra_var_bub:
                        # Try reading as if 10 columns
                        E = pd.read_csv(
                            file_path,
                            sep=r'\s+',
                            header=None,
                            names=['i', 't', 'x1', 'x2', 'x3', 'u1', 'u2', 'u3','v1','v2','v3','v4','v5'],
                            usecols=range(13),
                            dtype=str   # read everything as strings first
                        )
                        for col in E.columns:
                            E[col] = pd.to_numeric(E[col], errors='coerce')
                        Etxu = E.iloc[:, 1:13].values  # from 't' to 'WD'
                        Efile = os.path.join(Bpath, f"bub_{i}.npy")
                        np.save(Efile, Etxu[::reduce_tbub])
                        print("saved bubble file!")
                    else:
                        # Try reading as if 10 columns
                        E = pd.read_csv(
                            file_path,
                            sep=r'\s+',
                            header=None,
                            names=['i', 't', 'x1', 'x2', 'x3', 'u1', 'u2', 'u3'],
                            usecols=range(8),
                            dtype=str   # read everything as strings first
                        )
                        for col in E.columns:
                            E[col] = pd.to_numeric(E[col], errors='coerce')

                        # Select the time + positions + velocities + optional WS, WD
                        Etxu = E.iloc[:, 1:8].values  # from 't' to 'WD'
                        Efile = os.path.join(Bpath, f"bub_{i}.npy")
                        np.save(Efile, Etxu[::reduce_tbub])
                        print("saved bubble file!")
                    if not no_cmn:
                        Esh = pd.read_csv(
                            f"{Epath}/bub_{i:03d}_cnm.dat",
                            sep=r'\s+',
                            header=None
                        )
                        Esh_mat = Esh.values
                        Efile_sh = os.path.join(Bpath, f"bub_{i}_sh.npy")
                        np.save(Efile_sh, Esh_mat)

                write_bool(boolpath, False, force_reading)
            else:
                print('Externen Einlesevorgang erkannt: Warten auf Fertigstellung...')
                while read_bool(boolpath, force_reading):
                    time.sleep(1)
                print('Fertigstellung erkannt.')



        if with_ell and newbinary==1 and ell_type=='ellipsoid' and not onlyInfo:
            print('Translate Ellipsoid Data to Numpy Binary (takes the longest time)')
            Epath = self.resultpath + '/ellipsoid'
            Bpath = Epath + '/ell_binary'
            try:
                os.makedirs(Bpath)
            except FileExistsError:
                pass
            boolpath = Bpath+'/writing.dat'

            if not read_bool(boolpath,force_reading):
                write_bool(boolpath,True,force_reading)
                for i in range(1,self.eNr+1):
                    print('Elliposid '+format(i+1,"d")+'/'+format(self.eNr+1,"d"))
                    if with_coll_force:
                        E = pd.read_csv(Epath+'/ell_'+ format(i, "03d") +'_monitor.dat', sep=r'\s+', header=None, names=['i','t','x1','x2','x3','u1','u2','u3','o1','o2','o3','f1','f2','f3'],usecols=range(14))
                        Etxu = E.iloc[:,1:14].values
                    else:
                        E = pd.read_csv(Epath+'/ell_'+ format(i, "03d") +'_monitor.dat', sep=r'\s+', header=None, names=['i','t','x1','x2','x3','u1','u2','u3','o1','o2','o3'],usecols=range(11))
                        Etxu = E.iloc[:,1:11].values

                    if 1==2:
                        try:
                            t = Etxu[:,0]
                            td = np.diff(t)
                            dt = t[-3]-t[-2]
                            indt = np.argwhere(td < -dt*1.5)
                            if np.any(indt):
                                print('corrected Ellipsoid data')
                                print(t[indt])
                            mask = np.ones(t.shape, dtype=bool)

                            for idt in indt:
                                orig_t = t[idt+1]
                                iit = 0
                                ti = 0
                                while ti>orig_t or iit==0:
                                    ti = t[idt-iit]
                                    mask[idt-iit] = False
                                    iit+=1
                            Etxu = Etxu[mask,:]
                        except Exception as e:
                            print(f"Fehler bei der Ell-Monitor-Korrektur: {e}")

                    Efile = Bpath + '/ell_' + format(i,"d") + '.npy'

                    np.save(Efile, Etxu[::reduce_tell])
                write_bool(boolpath,False,force_reading)
            else:
                print('externen Einlesevorgang erkannt: Warten auf Fertigstellung...')
                while read_bool(boolpath):
                    time.sleep(1)
                print('Fertigstellung erkannt.')

        self.tkepath = path + '/../postproc_bubble/tke_results'
        with_tke = os.path.exists(self.tkepath)
        self.fluidpath = path + '/fluid'
        if with_tke and not onlyInfo:
            tkeI = []
            fI = []
            for filename in os.listdir(self.tkepath):
                i = extract_numeric(filename)
                if i<10000 and isinstance(i,int): tkeI = np.append(tkeI,i)
            self.tkeI = tkeI
            #print(tkeI)

            for filename in os.listdir(self.fluidpath):
                i = extract_numeric(filename)
                if i<10000 and isinstance(i,int): fI = np.append(fI,i)
            self.fI = fI

        self.force_reading = force_reading
        print('Done')



    @property
    def Xp(self):
        return np.load(self.Gpath+'/Xp.npy')
    @property
    def Yp(self):
        return np.load(self.Gpath+'/Yp.npy')
    @property
    def Zp(self):
        return np.load(self.Gpath+'/Zp.npy')
    @property
    def Xu(self):
        return np.load(self.Gpath+'/Xu.npy')
    @property
    def Yu(self):
        return np.load(self.Gpath+'/Yu.npy')
    @property
    def Zu(self):
        return np.load(self.Gpath+'/Zu.npy')
    @property
    def Xv(self):
        return np.load(self.Gpath+'/Xv.npy')
    @property
    def Yv(self):
        return np.load(self.Gpath+'/Yv.npy')
    @property
    def Zv(self):
        return np.load(self.Gpath+'/Zv.npy')
    @property
    def Xw(self):
        return np.load(self.Gpath+'/Xw.npy')
    @property
    def Yw(self):
        return np.load(self.Gpath+'/Yw.npy')
    @property
    def Zw(self):
        return np.load(self.Gpath+'/Zw.npy')

    @property
    def t_conv(self):
        conv_dict = convergence_times()
        if self.label in conv_dict:
            t_conv = conv_dict[self.label]
            print('Konvergez-Zeit festgelegt auf: '+format(t_conv,'.2f'))
            return t_conv
        else:
            print('keine End-Zeit gefunden, setze auf 0')
            return 0

    @property
    def te(self):
        return self.tf[-1]
        # conv_dict = end_time_step()
        # if self.label in conv_dict:
        #     t_conv = conv_dict[self.label]
        #     print('Endzeit festgelegt auf: '+format(t_conv,'.2f'))
        #     return t_conv
        # else:
        #     print('keine Konvergenz-Zeit gefunden -> letzter Wert')
        #     return self.tee

    @property
    def ne_conv(self):
        e = self.e(1)
        ne = np.argmin( np.abs( e.t - self.t_conv )) + 1
        return int(ne) #First Ellipsoid Index inside analysis interval

    @property
    def t_corr(self):
        corr_dict = correction_times()
        if self.label in corr_dict:
            t_corr = corr_dict[self.label]
            print('Korrektur-Zeitpunkt: '+format(t_corr,'.2f'))
            return t_corr
        else:
            print('keine Korrektur während Sim')
            return -1



#Methods
    #Fluid-Field-Data
    # a fluid object is created by this method and binary data is read
    def f(self,i):
        print('Reading Fluid Data of '+format(i,"d")+'th datapoint')
        flfield = fluid(self,i);
        print('Done')
        return flfield

    #Ellipsoid-Data
    def e(self,i):
        elli = ellipsoid(self,i);
        return elli

    #Bubble-Data
    def b(self,i,check=False):
        bub = bubble(self,i,check=check);
        return bub


    def tke(self,newbinary=False):
        return TKE(self,new=newbinary)

    def dfE(self):
        df = []
        for i in range(1,self.eNr+1):
            print('Ell-'+format(i,'d'))
            if i==1:
                df = self.e(i).df()
            else:
                df = pd.concat([df, self.e(i).df()], axis=0).reset_index(drop=True)   #,Nt=1000, NsubT=20, Nr=5000,
        return df

    def elec(self,i):
        elec = electrode(self,i);
        return elec

    def voro(self,newbinary=False,newrandom=False,read_sim=True,read_random=True, random_type='no', cluster_strength=0.5, Nt=-1, NsubT=20, Nr=-1, NsubR=20, Naw=200, additive=True, Nbin=20):
        if Nt==-1:
            conv_dict = VoroNt()
            if self.label in conv_dict:
                Nt = conv_dict[self.label]
                print('Vornoi Nt festgelegt: '+format(Nt,'.0f'))
            else:
                print('kein Voro Nt gefunden, setze auf 5000')
                Nt =  5000
        if Nr==-1:
            conv_dict = VoroNR()
            if self.label in conv_dict:
                NR = conv_dict[self.label]
                print('Vornoi Random NR festgelegt: '+format(NR,'.0f'))
            else:
                print('kein Voro NR gefunden, setze auf 5000')
                NR =  5000
        return voro(self, newbinary=newbinary, newrandom=newrandom,read_sim=read_sim,read_random=read_random, random_type=random_type, cluster_strength=cluster_strength, Nt=Nt, NsubT=NsubT, Nr=NR, NsubR=NsubR, Naw=Naw, additive=additive, Nbin=Nbin)

    @property
    def dfTKE(self):
        df = []
        o = 0
        for i in self.tkeI:
            print('TKE-'+format(o,'d'))
            o += 1
            dfi = self.tke(i).df
            if isinstance(dfi, pd.DataFrame):
                dfi['t'] = np.repeat(self.tf[int(i-1)],len(dfi))
                if o==1:
                    df = dfi
                else:
                    df = pd.concat([df, dfi], axis=0).reset_index(drop=True)
        return df

    def exyz(self,t,GK=0):
        if GK==0:
            xyz = np.zeros((self.eNr,3))
            for i in range(1,self.eNr+1):
                e = self.e(i)
                xyz[i-1,0] = e.xi(t)
                xyz[i-1,1] = e.yi(t)
                xyz[i-1,2] = e.zi(t)
        else:
            xyz = np.zeros(np.sum(self.GKi==GK),3)
            p = 0
            for i in range(1,self.eNr+1):
                e = self.e(i)
                if e.GK==GK:
                    xyz[p,0] = e.ui(t)
                    xyz[p,1] = e.vi(t)
                    xyz[p,2] = e.wi(t)
                    p += 1
        return xyz

    def euvw(self,t,GK=0):
        if GK==0:
            uvw = np.zeros((self.eNr,3,len(t)))
            for i in range(1,self.eNr+1):
                e = self.e(i)
                uvw[i-1,0,:] = e.ui(t)
                uvw[i-1,1,:] = e.vi(t)
                uvw[i-1,2,:] = e.wi(t)
        else:
            if not self.continous:
                uvw = np.zeros((np.sum(self.GKi==GK),3,len(t)))
                p = 0
                for i in range(1,self.eNr+1):
                    e = self.e(i)
                    if e.GK==GK:
                        uvw[p,0,:] = e.ui(t)
                        uvw[p,1,:] = e.vi(t)
                        uvw[p,2,:] = e.wi(t)
                        p += 1
            else:
                if GK<=self.eNr:
                    uvw = np.zeros((3,len(t)))
                    e = self.e(GK)
                    uvw[0,:] = e.ui(t)
                    uvw[1,:] = e.vi(t)
                    uvw[2,:] = e.wi(t)
                else:
                    uvw = 0
        return uvw

    def eRe(self,t,GK=0):
        if GK==0:
            Re = np.zeros((self.eNr,len(t)))
            for i in range(1,self.eNr+1):
                e = self.e(i)
                Re[i-1,:] = e.Rei(t)
        else:
            if not self.continous:
                Re = np.zeros((np.sum(self.GKi==GK),len(t)))
                p = 0
                for i in range(1,self.eNr+1):
                    e = self.e(i)
                    if e.GK==GK:
                        Re[p,:] = e.Rei(t)
                        p += 1
            else:
                if GK<=self.eNr:
                    e = self.e(GK)
                    Re = e.Rei(t)
                else:
                    Re = 0
        return Re




# Fluid class
class fluid:
    #Constructor
    def __init__(self,sim,i):
        path = sim.resultpath + '/fluid'
        self.fpath = path
        self.rpath = sim.resultpath
        fluid_path = path+"/uvw_"+format(i, "06d")+".bin"

        if not sim.ssh_pass=='local':
            if not os.path.exists(fluid_path):
                from .ssh_functions import ssh_get_fluid
                print('Download Fluid data via ssh...')
                ssh_get_fluid(i, sim.ssh_pass, sim.resultpath, sim.rpath)
                print('...done')

        # Read Binary - Velocity
        with open(path+"/uvw_"+format(i, "06d")+".bin", 'rb') as fin:
                vel = np.fromfile(fin, dtype='>d')                              # Reads Binary to numpy Array
                vel = np.delete(vel, 0)                                         # first Entry is rubbish (i think its a timestamp)
                vel = np.reshape(vel, [3, sim.nx, sim.ny, sim.nz], order='f')   # Reshape into 3D Matrix
                vel = np.transpose(vel, (2, 1, 3, 0))                           # Change order
        # Read Binary - Pressure
        with open(path+"/pre_"+format(i, "06d")+".bin", 'rb') as fin:
                pel = np.fromfile(fin, dtype='>d')
                pel = np.delete(pel, 0)
                pel = np.reshape(pel, [sim.nx, sim.ny, sim.nz], order='f')
                self.p = np.transpose(pel, (1, 0, 2))
        # Create Variables
        self.u = vel[:,:,:,0]
        self.v = vel[:,:,:,1]
        self.w = vel[:,:,:,2]

        self.i = i

    def save_slice(self,Z):
        print('Save Slice in the X-Y Plane of fluid data')
        Spath = self.rpath + '/fl_slices'
        try:
            os.makedirs(Spath)
        except FileExistsError:
            pass

        np.save(Spath + '/u_' + format(self.i,"d") + '.npy', self.u[:,:,Z])
        np.save(Spath + '/v_' + format(self.i,"d") + '.npy', self.v[:,:,Z])
        np.save(Spath + '/w_' + format(self.i,"d") + '.npy', self.w[:,:,Z])
        np.save(Spath + '/p_' + format(self.i,"d") + '.npy', self.p[:,:,Z])
        print('Done')


#Bubble class
class bubble:
    #Constructor
    def __init__(self,sim,i,check=False):
        # Read monitor.dat file
        path = sim.resultpath + '/bubble/bub_binary/bub_' + format(i,"d") + '.npy'
        self.path_sh = sim.resultpath + '/bubble/bub_binary/bub_' + format(i,"d") + '_sh.npy'
        self.spath = sim.resultpath
        boolpath = sim.resultpath + '/bubble/bub_binary/writing.dat'
        if read_bool(boolpath,sim.force_reading):
            print('externen Einlesevorgang erkannt: Warten auf Fertigstellung...')
            while read_bool(boolpath):
                time.sleep(1,False)
            print('Fertigstellung erkannt.')
        self.i = i
        if not sim.onlyInfo:
            B = np.load(path, allow_pickle=True)
            self.nb = sim.bNr
            self.bnt = sim.bnt_plot
            try:
                self.r = sim.bRI[i-1]
                self.nl = sim.nlRI[i-1]
            except:
                print('no radius for bubble!')

            self.t = B[:,0]

            self.x = B[:,1]
            self.y = B[:,2]
            self.z = B[:,3]

            self.u = B[:,4]
            self.v = B[:,5]
            self.w = B[:,6]

            try:
                surfpath = sim.resultpath + f'/bubble/bub_{i:03d}_monitor_S.dat'
                Smat = np.loadtxt(surfpath)
                self.Smat = Smat
            except:
                print('no Surface Area file found (no worries)')

            if sim.extra_bub_var:
                self.v1 = B[:,7]
                self.v2 = B[:,8]
                self.v3 = B[:,9]
                self.v4 = B[:,10]
                self.v5 = B[:,11]
                self.v1i = interpolate.interp1d(self.t, self.v1,fill_value="extrapolate")
                self.v2i = interpolate.interp1d(self.t, self.v2,fill_value="extrapolate")
                self.v3i = interpolate.interp1d(self.t, self.v3,fill_value="extrapolate")
                self.v4i = interpolate.interp1d(self.t, self.v4,fill_value="extrapolate")
                self.v5i = interpolate.interp1d(self.t, self.v5,fill_value="extrapolate")

            if check:
                xj = np.where(np.abs( np.diff(self.x) ) >sim.Lx/2)[0]
                jx = np.zeros((np.size(xj),2))
                jxt = np.zeros((np.size(xj),2))

                jx[:,0]  = self.x[xj]
                jx[:,1]  = self.x[xj+1]
                jxt[:,0] = self.t[xj]
                jxt[:,1] = self.t[xj+1]
                self.jx  = np.array(jx)
                self.jxt = np.array(jxt)

                yj = np.where(np.abs( np.diff(self.y) )>sim.Ly/2)[0]
                jy = np.zeros((np.size(yj),2))
                jyt = np.zeros((np.size(yj),2))

                jy[:,0]  = self.y[yj]
                jy[:,1]  = self.y[yj+1]
                jyt[:,0] = self.t[yj]
                jyt[:,1] = self.t[yj+1]
                self.jy  = np.array(jy)
                self.jyt = np.array(jyt)

                zj = np.where(np.abs( np.diff(self.z) )>sim.Lz/2)[0]
                jz = np.zeros((np.size(zj),2))
                jzt = np.zeros((np.size(zj),2))

                jz[:,0]  = self.z[zj]
                jz[:,1]  = self.z[zj+1]
                jzt[:,0] = self.t[zj]
                jzt[:,1] = self.t[zj+1]
                self.jz  = jz
                self.jzt = jzt

            self.xiI = interpolate.interp1d(self.t, self.x,fill_value="extrapolate")
            self.yiI = interpolate.interp1d(self.t, self.y,fill_value="extrapolate")
            self.ziI = interpolate.interp1d(self.t, self.z,fill_value="extrapolate")

            self.check = check

            self.uiI = interpolate.interp1d(self.t, self.u,fill_value="extrapolate")
            self.viI = interpolate.interp1d(self.t, self.v,fill_value="extrapolate")
            self.wiI = interpolate.interp1d(self.t, self.w,fill_value="extrapolate")

            self.uges = np.sqrt(self.u**2 + self.v**2 + self.w**2)
            try:
                self.Re = self.uges * sim.d_out / sim.vis
                self.ReiI = interpolate.interp1d(self.t, self.Re, fill_value="extrapolate")
            except:
                pass


    def Xi(self,ti,I=False):
        x = np.atleast_1d( self.xi(ti,I=I) )
        y = np.atleast_1d( self.yi(ti,I=I) )
        z = np.atleast_1d( self.zi(ti,I=I) )
        X = np.column_stack([x,y,z])
        if X.shape[0] == 1:
            X = X[0]
        return X

    def xi(self,ti,I=False):
        if self.check and not I:
            res = self.xiI(ti)
            for k,t in enumerate(ti):
                for gi,gx in zip(self.jxt, self.jx):
                    if (t>gi[0] and t<gi[1]):
                        dt = gi[1]-gi[0]
                        if (gi[1]-t)>dt/2:
                            ind = 0
                        else:
                            ind = 1
                        res[k] = gx[ind]
            return res
        if I:
            ind = np.array(ti)*self.bnt
            return self.x[ind]
        return self.xiI(ti)

    def yi(self,ti,I=False):
        if self.check and not I:
            res = self.yiI(ti)
            for k,t in enumerate(ti):
                for gi,gx in zip(self.jyt, self.jy):
                    if (t>gi[0] and t<gi[1]):
                        dt = gi[1]-gi[0]
                        if (gi[1]-t)>dt/2:
                            ind = 0
                        else:
                            ind = 1
                        res[k] = gx[ind]
            return res
        if I:
            ind = np.array(ti)*self.bnt
            return self.y[ind]
        return self.yiI(ti)

    def zi(self,ti,I=False):
        if self.check and not I:
            res = self.ziI(ti)
            for k,t in enumerate(ti):
                for gi,gx in zip(self.jzt, self.jz):
                    if (t>gi[0] and t<gi[1]):
                        dt = gi[1]-gi[0]
                        if (gi[1]-t)>dt/2:
                            ind = 0
                        else:
                            ind = 1
                        res[k] = gx[ind]
            return res
        if I:
            ind = np.array(ti)*self.bnt
            return self.z[ind]
        return self.ziI(ti)


    def ui(self,ti,I=False):
        if I:
            ind = np.array(ti)*self.bnt
            return self.u[ind]
        return self.uiI(ti)

    def vi(self,ti,I=False):
        if I:
            ind = np.array(ti)*self.bnt
            return self.v[ind]
        return self.viI(ti)

    def wi(self,ti,I=False):
        if I:
            ind = np.array(ti)*self.bnt
            return self.w[ind]
        return self.wiI(ti)

    def Rei(self,ti,I=False):
        if I:
            ind = np.array(ti)*self.bnt
            return self.Re[ind]
        return self.ReiI(ti)


    def sh(self,ti,res=200):

        def get_Nsh(x): # calculate Nsh from len
            if x==1: return 1
            n=0
            s=0
            while n<1000:
                n+=1
                s+=n
                if x==s: return n

        Bsh = np.load(self.path_sh)

        nphi = round(res/2)
        ntheta = nphi*2
        a_t = np.linspace(0,2*np.pi,ntheta+1)[0:-1]
        a_p = np.linspace(0,np.pi,nphi+2)[1:-1]
        Th,Ph = np.meshgrid(a_t,a_p)
        theta = np.append(Th.ravel(),0)
        phi = np.append(Ph.ravel(),0)

        xi = np.zeros([len(ti),3])

        ind0 = nphi*ntheta
        i = np.arange(0,ind0)
        Ind = np.reshape(i,np.shape(Ph))
        print(np.shape(Ind))
        Ind = Ind.T
        faces = []
        for i in range(1,ntheta-1):
            fac = [Ind[i,0], Ind[i+1,0], ind0]
            faces.append(fac)
            for j in range(1,nphi-1):
                fac = [Ind[i,j], Ind[i+1,j], Ind[i,j+1], Ind[i+1,j+1]]
                faces.append(fac)

        lenB = Bsh.shape()[1]
        Nsh = get_Nsh(lenB)
        amn_x = B[::3,:]
        amn_y = B[1::3,:]
        amn_z = B[2::3,:]

        for n in range(0,Nsh):
            for m in range(-n,n):
                Ymn = sph_harm(m,n,theta,phi)
                if m==0:
                    Z = n*(n+1)/2 + 0 + 1
                else:
                    Z = n*(n+1)/2 + np.abs(m) + 1
                xi[:,0] = xi[:,0] + amn_x[Z]*Ymn
                xi[:,1] = xi[:,1] + amn_y[Z]*Ymn
                xi[:,2] = xi[:,2] + amn_z[Z]*Ymn

        return xi,faces

    def fp(self,j,sim,old=False):
        j = j * self.bnt
        i = self.i
        print(j)
        if old:
            fpname = f"bubble/bubble_fp/bub_fp_0{j:05d}.bin"
        else:
            fpname = f"bubble/bubble_fp/bub_{i:03d}_fp_0{j:05d}.bin"
            print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fp = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fp)
            fp = np.delete(fp,[round(lf/3-1),round(2*lf/3)])
            fpx = np.reshape(fp,[3,-1]).T
        return fpx

    def tc(self,j,sim,old=False):
        j = j * self.bnt
        i = self.i
        print(j)
        if old:
            fpname = f"bubble/bubble_tc/bub_tc_0{j:05d}.bin"
        else:
            fpname = f"bubble/bubble_tc/bub_{i:03d}_tc_0{j:05d}.bin"
            print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fp = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fp)
            fp = np.delete(fp,[round(lf/3-1),round(2*lf/3)])
            fpx = np.reshape(fp,[3,-1]).T
        return fpx


    def Si(self,j,sim,old=False):
        j = j * self.bnt
        i = self.i
        print(j)
        if old:
            fpname = f"bubble/bubble_Si/bub_tan_0{j:05d}.bin"
        else:
            fpname = f"bubble/bubble_Si/bub_{i:03d}_Si1_0{j:05d}.bin"
            print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fp = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fp)
            fp = np.delete(fp,[round(lf/3-1),round(2*lf/3)])
            fpx = np.reshape(fp,[3,-1]).T
        return fpx


    def jp(self,j,sim):
        j = j * self.bnt
        i = self.i
        fpname = f"bubble/bubble_fp_jp/bub_{i:03d}_fp_jp_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps


    def pri(self,j,sim):
        j = j * self.bnt
        i = self.i
        fpname = f"bubble/bubble_pri/bub_{i:03d}_pri_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps


    def pri1(self,j,sim):
        j = j * self.bnt
        i = self.i
        fpname = f"bubble/bubble_pri1/bub_{i:03d}_pri1_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps


    def pri2(self,j,sim):
        j = j * self.bnt
        i = self.i
        fpname = f"bubble/bubble_pri2/bub_{i:03d}_pri2_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps


    def pri_sk(self,j,sk,sim):
        j = j * self.bnt
        i = self.i
        fpname = f"bubble/bubble_pri/bub_{i:03d}_{sk:03d}_fp_pri_sk_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps

    def fp_sk(self,j,sk,sim):
        j = j * self.bnt
        i = self.i
        fpname = f"bubble/bubble_fp/bub_{i:03d}_{sk:03d}_fp_sk_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fp = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fp)
            fp = np.delete(fp,[round(lf/3-1),round(2*lf/3)])
            fpx = np.reshape(fp,[3,-1]).T
        return fpx

    def fnop(self,j,sim):
        j = j * self.bnt
        i = self.i
        fpname = f"bubble/bubble_fnop/bub_{i:03d}_fnop_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps

    def ph(self,j,sim):
        j = j * self.bnt
        i = self.i
        fpname = f"bubble/bubble_ph/bub_{i:03d}_ph_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps

    def fpcoll(self,j,sim,old=False):
        j = j * self.bnt
        i = self.i
        print(i)
        fpname = f"/bubble/bubble_fp_coll/bub_{i:03d}_fp_coll_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps

    def fpdc(self,j,sim,old=False):
        j = j * self.bnt
        i = self.i
        print(i)
        fpname = f"/bubble/bubble_fp_dc/bub_{i:03d}_fp_dc_0{j:05d}.bin"
        print(fpname)
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps


    def un(self,j,sim,old=False):
        j = j * self.bnt
        i = self.i
        fpname = f"/bubble/bubble_fp_un/bub_{i:03d}_fp_un_0{j:05d}.bin"
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fps = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fps)
            #fp = np.delete(fp,[round(lf/3-2),round(2*lf/3)])
        return fps

    def un_sk(self,j,sk,sim,old=False):
        j = j * self.bnt
        i = self.i
        fpname = f"/bubble/bubble_fp_un/bub_{i:03d}_{sk:03d}_fp_un_0{j:05d}.bin"
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fp = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fp)
            fp = np.delete(fp,[round(lf/3-1),round(2*lf/3)])
            fpx = np.reshape(fp,[3,-1]).T
        return fpx


    def n(self,j,sim,old=False):
        j = j * self.bnt
        i = self.i
        if old:
            fpname = f"bubble/bubble_fpn/bub_fp_n_0{j:05d}.bin"
        else:
            fpname = f"bubble/bubble_fp_n/bub_{i:03d}_fp_n_0{j:05d}.bin"
        pathname = self.spath + '/' + fpname
        if not os.path.exists(pathname):
            ssh_get_files(sim,fpname)
        with open(pathname, 'rb') as fin:
            fp = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fp)
            fp = np.delete(fp,[round(lf/3-1),round(2*lf/3)])
            fpx = np.reshape(fp,[3,-1]).T
        return fpx




class electrode:
    #Constructor
    def __init__(self,sim,i):
        # Read monitor.dat file
        path = sim.resultpath + '/ellipsoid/ell_binary/ell_' + format(i,"d") + '.npy'
        self.ent = sim.elec_plot
        self.spath = sim.resultpath

    def fp(self,j,sim, path='none'):
        if path=='none':
            j = j * self.ent
            fpname = f"ellipsoid/ellipsoid_fp/ell_fp_0{j:05d}.bin"
            pathname = self.spath + '/' + fpname
        else:
            fpname = f"ellipsoid/ellipsoid_fp/{path}"
            pathname = self.spath + '/' + fpname
            if not os.path.exists(pathname):
                ssh_get_files(sim,fpname)

        with open(pathname, 'rb') as fin:
            fp = np.fromfile(fin, dtype=np.float64,offset=4)
            lf = np.size(fp)
            print(lf)
            fp = np.delete(fp,[round(lf/3-1),round(2*lf/3)])
            fpx = np.reshape(fp,[3,-1]).T
        return fpx


# Ellipsoid class
class ellipsoid:
    #Constructor
    def __init__(self,sim,i):
        # Read monitor.dat file
        path = sim.resultpath + '/ellipsoid/ell_binary/ell_' + format(i,"d") + '.npy'
        boolpath = sim.resultpath + '/ellipsoid/ell_binary/writing.dat'
        if read_bool(boolpath,sim.force_reading):
            print('externen Einlesevorgang erkannt: Warten auf Fertigstellung...')
            while read_bool(boolpath,False):
                time.sleep(1)
            print('Fertigstellung erkannt.')

        E = np.load(path)

        self.id = i
        self.with_coll = sim.with_coll_force
        # Get diameter from sim object
        self.d = 1#sim.d[i-1]
        self.dcorr = 1#sim.dcorr[i-1]
        sim.dcorr = [1.0]
        # Groeßenklasse
        if 1==2:#sim.continous:
            GK = np.where(sim.du == self.d)[0] + 1
            self.GK = GK[0]
        else:
            self.GK = 1#1*(self.d==sim.d1) + 2*(self.d==sim.d2) + 3*(self.d==sim.d3)
        # Lagrange-Data from pandas to numpy

        self.t = E[:,0]


        self.x = E[:,1]
        self.y = E[:,2]
        self.z = E[:,3]

        self.u = E[:,4]
        self.v = E[:,5]
        self.w = E[:,6]
        if self.with_coll:
            self.cx = E[:,10]
            self.cy = E[:,11]
            self.cz = E[:,12]

        # Domian Length (sometimes needed)
        self.Lx = sim.Lx
        self.Ly = sim.Ly
        self.Lz = sim.Lz

        # Interpolation Functions
        self.xi = interpolate.interp1d(self.t, self.x,fill_value="extrapolate")
        self.yi = interpolate.interp1d(self.t, self.y,fill_value="extrapolate")
        self.zi = interpolate.interp1d(self.t, self.z,fill_value="extrapolate")

        self.ui = interpolate.interp1d(self.t, self.u,fill_value="extrapolate")
        self.vi = interpolate.interp1d(self.t, self.v,fill_value="extrapolate")
        self.wi = interpolate.interp1d(self.t, self.w,fill_value="extrapolate")

        if self.with_coll:
            self.cxi = interpolate.interp1d(self.t, self.cx,fill_value="extrapolate")
            self.cyi = interpolate.interp1d(self.t, self.cy,fill_value="extrapolate")
            self.czi = interpolate.interp1d(self.t, self.cz,fill_value="extrapolate")

        # Overall velocity and Reynolds-Number
        self.uges = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.Re = self.uges * sim.vis
        self.Rei = interpolate.interp1d(self.t, self.Re, fill_value="extrapolate")

    def df(self):
        # this will create a df that contains all attributes for the reduced timestep
        l = np.size(self.t)
        i = np.repeat(self.id,l)
        d = np.repeat(self.d,l)
        gk = np.repeat(self.GK,l)
        data = np.column_stack([i,d,gk,self.t,self.x,self.y,self.z,self.u,self.v,self.w,self.Re,self.cx,self.cy,self.cz])
        param = ['e', 'd', 'GK','t','x','y','z','u','v','w','Re','cx','cy','cz']
        return pd.DataFrame(data, columns=param)

    # Is-in-plane-Check
    def inplane(self, t, x, direction):
        if direction=='x':
            zt = self.xi(t)
            zt = [zt-self.Lx, zt, zt+self.Lx]
            dist = min(np.absolute(zt-x))
            if dist<self.d/2:
                dB = 2*np.sqrt((self.d/2)**2 - dist**2)
                return 1, [self.yi(t)-self.Ly, self.yi(t), self.yi(t)+self.Ly] , [self.zi(t)- self.Lz, self.zi(t), self.zi(t)+self.Lz] , dB
            else:
                return 0, 0, 0, 0

        if direction=='y':
            zt = self.yi(t)
            dist = np.absolute(zt-x)
            if dist<self.d/2:
                dB = 2*np.sqrt((self.d/2)**2 - dist**2)
                return 1, [self.xi(t)- self.Lx, self.xi(t), self.xi(t)+self.Lx], [self.zi(t)- self.Lz, self.zi(t), self.zi(t)+self.Lz], dB
            else:
                return 0, 0, 0, 0

        if direction=='z':
            zt = self.zi(t)
            zt = [zt-self.Lz, zt, zt+self.Lz]
            dist = min(np.absolute(zt-x))
            if dist<self.d/2:
                dB = 2*np.sqrt((self.d/2)**2 - dist**2)
                return 1, [self.xi(t)-self.Lx, self.xi(t), self.xi(t)+self.Lx], [self.yi(t)-self.Ly, self.yi(t), self.yi(t)+self.Ly], dB
            else:
                return 0, 0, 0, 0

    def vdti(self,t):
        ti = self.t
        dt = ti[1]-ti[0]
        vi = interpolate.interp1d(self.t[0:-1]+dt/2, np.diff(self.v),fill_value="extrapolate")
        return vi(t)


def read_tke_file(file_path):
    encodings_to_try = ['utf-8', 'utf-16', 'ISO-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                data = [list(map(lambda x: float(x.replace('D', 'E')), line.split())) for line in lines]
                tkemat = np.array(data)
                # process the lines as needed
            break
        except UnicodeDecodeError:
            print(f"Failed to open with encoding: {encoding}")

        #print(np.shape(tkemat))
    return tkemat






class TKE:
    def __init__(self,sim,new=True):
        def read_tke(sim,i):
            i = int(i)
            path = sim.tkepath + '/' + format(i,'06d')
            usable=False
            for filename in os.listdir(path):
                if 'dissipation' in filename: usable=True
            if usable:
                tke_props = ['/dissipation.dat','/interface_term.dat','/p_av.dat','/out_of_balance.dat','/press_diffusion.dat','/production.dat','/RST.dat','/tke.dat','/tripel_correlation.dat','/u_av.dat','/visc_diffusion.dat','/void_fraction_distribution.dat']
                A = []
                for i,tke_prop in enumerate(tke_props):
                    B = read_tke_file(path+tke_prop)
                    if np.size(B)==0: return 0
                    if i==0:
                        A.append(B)
                    else:
                        A.append(B[:,1:])
                M = np.concatenate(A,1)
                return M
            return 0

        self.tke_load = sim.resultpath + '/tke_loading'
        if new:
            try:
                os.makedirs(self.tke_load)
            except FileExistsError:
                pass

            ltke = len(sim.tkeI)
            for ii,i in enumerate(sim.tkeI):
                print('Reading TKE at Step' + format(ii,'d'))
                M = read_tke(sim,i)

                if np.size(M)>1:
                    if ii == 0:
                        y = M[:,0]
                        ly = len(y)
                        TKE = np.full((ly,19,ltke),np.nan)
                        time = np.full(ltke,np.nan)
                        mask = np.ones(ltke, dtype=bool)
                    if not M is None and np.size(M)>0 and i<len(sim.tf):
                        TKE[:,:,ii] = M[:,1:]
                        time[ii] = sim.tf[int(i-1)]
                    else:
                        mask[ii] = False
            time = time[mask]
            t_sort = np.argsort(time)
            TKE = TKE[:,:,mask]
            TKE = TKE[:,:,t_sort]
            time = time[t_sort]

            np.save(self.tke_load + '/tke.npy',TKE)
            np.save(self.tke_load + '/tke_t.npy',time)
            np.save(self.tke_load + '/tke_y.npy',y)
            np.save(self.tke_load + '/tkeM.npy',np.nanmean(TKE,0))

        turb = np.load(self.tke_load+'/tkeM.npy')
        self.diss = turb[0,:]
        self.tke = turb[12,:]
        self.prod = turb[5,:]
        self.inter = turb[1,:]
        self.p_av = turb[2,:]
        self.out_of_balance = turb[3,:]
        self.press_diff = turb[4,:]

        self.rst = turb[6:12,:]

        self.tripel_corr = turb[13,:]
        self.u_av = turb[14:17,:]
        self.u_b = turb[17:20,:]
        self.void_f_dist = turb[21,:]

        self.t = np.load(self.tke_load + '/tke_t.npy')
        self.y = np.load(self.tke_load + '/tke_y.npy')

        turb = np.load(self.tke_load+'/tke.npy')
        turb = np.transpose(turb,(0,2,1))
        self.diss_y = turb[:,:,0]
        self.tke_y = turb[:,:,12]
        self.prod_y = turb[:,:,5]
        self.inter_y = turb[:,:,1]
        self.p_av_y = turb[:,:,2]
        self.out_of_balance_y = turb[:,:,3]
        self.press_diff_y = turb[:,:,4]

        self.rst_y = turb[:,:,6:12]

        self.tripel_corr_y = turb[:,:,13]
        self.u_av_y = turb[:,:,14:17]
        self.u_b_y = turb[:,:,17:20]
        self.void_f_dist_y = turb[:,:,21]

        #print(self.t)
        #print(self.diss)

        #self.dissi = interpolate.interp1d(self.t, self.diss,fill_value="extrapolate")
        #self.tkei = interpolate.interp1d(self.t, self.tke,fill_value="extrapolate")
        #self.prodi = interpolate.interp1d(self.t, self.prod,fill_value="extrapolate")
        #self.interi = interpolate.interp1d(self.t, self.inter,fill_value="extrapolate")


class voro:
    def __init__(self, sim, newbinary=False, newrandom=False, read_sim=True, read_random=True, Nt = 2, NsubT = 2, Nr = 2, NsubR = 2, Naw = 150, additive=True,Nbin=20,avg=1, randfromfile=True, random_type='no', cluster_strength=0.5):
        from .voronoi import rand_hist
        from .voronoi import moving_average
        if newbinary:
            from .voronoi import make_Vvoro
            t = np.linspace(sim.t_conv,sim.te,Nt)
            make_Vvoro(sim, t, Nsub=NsubT, additive=additive,add_N=Naw)

        if newrandom:
            from .voronoi import make_rand_voro
            make_rand_voro(sim, Nv=Nr, Nsub=NsubR, additive=additive, onlygk=0,add_N = Naw,random_type=random_type,cluster_strength=cluster_strength)

        try:

            if read_sim:
                print('test')
                if additive:
                    vpath = '/voronoi/AWVolC'
                else:
                    vpath = '/voronoi/VolC'

                VolC = np.load(sim.resultpath+vpath+f"voro_{Nt}.npy")
                tV   = np.load(sim.resultpath+vpath+f"T_{Nt}.npy")
                p   = np.load(sim.resultpath+vpath+f"p_{Nt}.npy")

                self.Adjpath = sim.resultpath+vpath+f"Adja_{Nt}.npy"

                x,Vmean,Vmean1,Vmean2,Vmean3,Vsigma,Vsigma1,Vsigma2,Vsigma3,hov,h1,h2,h3,kdex,kdeVol,kdeV1,kdeV2,kdeV3,xmix,pkde1, pkde2, pkde3, Vschief, Vschief1, Vschief2, Vschief3 = rand_hist(sim,VolC,p,Nbins=Nbin)
                print('Hey')
                hov = moving_average(hov,avg)
                h1 = moving_average(h1,avg)
                h2 = moving_average(h2,avg)
                h3 = moving_average(h3,avg)

                self.VV = VolC
                self.pp = p
                self.tV = tV

                VBi = sim.d**3 / 6 * np.pi
                vlen = np.shape(VolC)[1]
                VBi = np.column_stack([VBi]*vlen)
                VolVB = VolC / VBi

                self.VVB = VolVB
                self.eg = 1/VolVB

                self.x = x
                self.Vm = Vmean
                self.Vmi = [Vmean1,Vmean2,Vmean3]
                self.Vstd = Vsigma
                self.Vstdi = [Vsigma1,Vsigma2,Vsigma3]

                self.Vschief = Vschief
                self.Vschiefi = [Vschief1,Vschief2,Vschief3]

                self.hov = hov
                self.hi = [h1,h2,h3]

                self.kdex = kdex
                self.kdeV = kdeVol
                self.kdeVi = [kdeV1, kdeV2, kdeV3]
                self.p = [pkde1,pkde2,pkde3]
                self.xp = xmix
                pvar = np.var(p, axis=(0,2))
                self.psigma = np.sqrt(pvar)


            if read_random:
                if additive:
                    if random_type=='no':
                        Rvpath = '/voronoi/R_AWVolC'
                    else:
                        Rvpath = f"/voronoi/R_AWVolC_{random_type}_{cluster_strength}"
                else:
                    if random_type=='no':
                        Rvpath = '/voronoi/R_VolC'
                    else:
                        Rvpath = f"/voronoi/R_VolC_{random_type}_{cluster_strength}"

                RVolC = np.load(sim.resultpath+Rvpath+f"voro_{Nr}.npy")
                Rp   = np.load(sim.resultpath+Rvpath+f"p_{Nr}.npy")


                Rx,RVmean,RVmean1,RVmean2,RVmean3,RVsigma,RVsigma1,RVsigma2,RVsigma3,Rhov,Rh1,Rh2,Rh3,Rkdex,RkdeV,RkdeV1,RkdeV2,RkdeV3,Rxp,Rpkde1, Rpkde2, Rpkde3, RVschief, RVschief1, RVschief2, RVschief3 = rand_hist(sim,RVolC,Rp,Nbins=Nbin)

                Rhov = moving_average(Rhov,avg)
                Rh1 = moving_average(Rh1,avg)
                Rh2 = moving_average(Rh2,avg)
                Rh3 = moving_average(Rh3,avg)

                self.RV = RVolC
                self.pR = Rp

                self.RVschief = RVschief
                self.RVschiefi = [RVschief1, RVschief2, RVschief3]

                self.Rx = Rx
                self.RVm = RVmean
                self.RVmi = [RVmean1, RVmean2, RVmean3]
                self.RVstd = RVsigma
                self.RVstdi = [RVsigma1, RVsigma2, RVsigma3]

                self.Rhov = Rhov
                self.Rhi = [Rh1, Rh2, Rh3]

                self.Rkdex = Rkdex
                self.RkdeV = RkdeV
                self.RkdeVi = [RkdeV1, RkdeV2, RkdeV3]
                Rpvar = np.var(Rp, axis=(0,2))
                self.Rpsigma = np.sqrt(Rpvar)
                self.Rxp = Rxp
                self.Rp = [Rpkde1, Rpkde2, Rpkde3]

            if read_sim and read_random:
                gku, un = np.unique(sim.GKi,return_counts=True)
                pGK = un/sum(un)
                print(pGK)

                var0 = pGK*(1-pGK)
                if sim.gkN==1:
                    self.IC = Vsigma**2 / RVsigma**2
                    self.ICs = Vschief**2 / RVschief**2
                    self.IS = 0
                else:
                    if sim.gkN==2:
                        self.IS = 1 - np.array([(var0[0] - pvar[0])/(var0[0] - Rpvar[0]), 0, (var0[1]-pvar[1])/(var0[1]-Rpvar[1])])
                        self.IC = [Vsigma1**2 / RVsigma1**2, 0, Vsigma3**2 / RVsigma3**2]
                        self.ICs = [Vschief1**2 / RVschief1**2, 0, Vschief3**2 / RVschief3**2]
                    else:
                        self.IS = 1 - np.array((var0 - pvar) / (var0 - Rpvar))
                        self.IC = [Vsigma1**2/RVsigma1**2, Vsigma2**2/RVsigma2**2, Vsigma3**2/RVsigma3**2]
                        self.ICs = [Vschief1**2/RVschief1**2, Vschief2**2/RVschief2**2, Vschief3**2/RVschief3**2]
                Vcrit = np.zeros(3)
                IntV = np.zeros(3)
                for k,(kV,RkV) in enumerate(zip(self.kdeVi,self.RkdeVi)):
                    if np.size(kV)>0:
                        vcrit = np.where( np.logical_and(np.isclose(kV, RkV, rtol=1e-3), kV>1E-2))[0]
                        vcritR = np.where( np.logical_and(np.isclose(RkV, kV, rtol=1e-3), kV>1E-2))[0]
                        if np.size(vcrit)>0:
                            mkV = np.gradient(kV, self.kdex)[vcrit]
                            mRkV = np.gradient(RkV, self.Rkdex)[vcrit]
                            if mkV[0]<mRkV[0]:
                                Vcrit[k] = self.kdex[vcrit[0]]

                                xvcrit = self.kdex[0:vcrit[0]]
                                xvcritR = self.Rkdex[0:vcritR[0]]

                                Vcii = kV[0:vcrit[0]]
                                VciR = RkV[0:vcritR[0]]

                                IntV[k] = np.trapz(Vcii,x=xvcrit) - np.trapz(VciR,x=xvcritR)
                IntV2 = np.zeros(3)
                for k,(kV,RkV) in enumerate(zip(self.kdeVi,self.RkdeVi)):
                    if np.size(kV)>0:
                        rvm = self.RVmi[k]
                        pdfI = interpolate.interp1d(self.kdex,kV, fill_value=0, bounds_error=False)
                        kVtoR = pdfI(self.Rkdex)
                        Cint = np.logical_and(kVtoR > RkV, self.Rkdex < rvm)
                        splitn = np.where(np.diff(Cint) > 1)[0] + 1
                        csplit = np.split(Cint, splitn)
                        for cinti in csplit:
                            xcc = self.Rkdex[cinti]
                            IntV2[k] += np.trapz(kVtoR[cinti],x=xcc) - np.trapz(RkV[cinti],x=xcc)

                self.Vcrit = Vcrit
                self.IntV = IntV

                self.IntV2 = IntV2


        except Exception as e:
            print(e)

    def A(self):
        return np.load(self.Adjpath)



def extract_numeric(filename, out=False, type='int'):
    if out:
        print('Ermittle Index aus " '+filename+' "')
    # Funktion, die aus den Benennungen 'uvw_0010' den Index extrahiert.
    parts = filename.split('_')
    numeric_part = ''.join(filter(lambda x: x.isdigit() or x == '.' or 'e' in x or '-' in x, parts[-1]))
    #numeric_part = ''.join(filter(str.isdigit, filename))
    try:
        if type=='int':
            numeric_value = int(numeric_part)
        if type=='float':
            numeric_value = float(numeric_part)
        if out:
            print('-> '+numeric_part)
        return numeric_value
    except ValueError:
        if out:
            print("no content")
        return 10000

def write_bool(file_path, value,fr):
    if not fr:
        with open(file_path, 'w') as file:
            file.write('1' if value else '0')

def read_bool(file_path,fr):
    if fr:
        return False
    else:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                bool =  content.strip() == '1'
                if bool:
                    if check_recent(file_path):
                        print('Einlese-Markierung liegt mehr als 10 min zurück und wird Ignoriert')
                        bool=False
                        write_bool(file_path,False)
                return bool
        except FileNotFoundError:
            return False  # Return False if the file doesn't exist

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
    # Create an empty list to store the values
    values = []

    # Define the regex pattern to match the format 'cfl_con =   <value>'
    pattern = r"cfl_con\s*=\s*([+-]?\d+\.\d+E[+-]?\d+)"

    # Open the file for reading
    with open(filename, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Search for lines matching the pattern
            match = re.search(pattern, line)
            if match:
                # Append the matched value (which is captured in the first group) to the list
                values.append(float(match.group(1)))

    return np.array(values)

def extract_cfl_values(folder_path):
    cfl_values = []
    file_pattern = re.compile(r'out_prime(?:_(\d+)_(\d+))?\.log')

    # Get sorted list of files based on the first numeric index
    files = []
    for filename in os.listdir(folder_path):
        match = file_pattern.match(filename)
        if match:
            num = int(match.group(1)) if match.group(1) is not None else -1
            files.append((num, filename))

    files.sort()  # Sort by the first number in the filename

    cfl_pattern = re.compile(r'cfl_con =\s+([\d\.E\+\-]+)')
    cfl_values = []

    for _, filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            print(f'CFL: searching {file_path}')
            for line in file:
                match = cfl_pattern.search(line)
                if match:
                    cfl_values.append(float(match.group(1)))


    return cfl_values

def extract_cfl_diff_values(folder_path):
    cfl_values = []
    file_pattern = re.compile(r'out_prime(?:_(\d+)_(\d+))?\.log')

    # Get sorted list of files based on the first numeric index
    files = []
    for filename in os.listdir(folder_path):
        match = file_pattern.match(filename)
        if match:
            num = int(match.group(1)) if match.group(1) is not None else -1
            files.append((num, filename))

    files.sort()  # Sort by the first number in the filename

    cfl_pattern = re.compile(r'cfl_dif =\s+([\d\.E\+\-]+)')
    cfl_values = []

    for _, filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            print(f'CFL: searching {file_path}')
            for line in file:
                match = cfl_pattern.search(line)
                if match:
                    cfl_values.append(float(match.group(1)))

    return cfl_values

#
# def extract_cfl_con_values(file_path):
#     # Regular expression to match the pattern "cfl_con = <number>" including scientific notation
#     pattern = re.compile(r'cfl_con\s*=\s*([-+]?\d*\.?\d+[eE][-+]?\d+|\d*\.?\d+)')
#
#     values = []
#
#     with open(file_path, 'r') as file:
#         for line in file:
#             match = pattern.search(line)
#             if match:
#                 # Extract the numerical value as a string and convert it to float
#                 value = float(match.group(1))
#                 values.append(value)
#     values = np.array(values)
#     return values

def extract_value_from_out(file_path,name):
    # Regular expression to match the pattern "cfl_con = <number>" including scientific notation
    #pattern = re.compile(rf'{name}\s*=\s*([-+]?\d*\.?\d+[eE][-+]?\d+|\d*\.?\d+)\s*') #
    #pattern = re.compile(rf'{name}(?<=\s*=\s*)[^ ]+')
    pattern = re.compile(rf'{name}\s*=\s*([^ ]+)')
    #pattern = re.compile(rf'{name}\s*=\s*([-+]?\d*\.?\d+[eEdD][-+]?\d+|\d*\.?\d+)\s*,')
    if name=='radius':
        pattern = re.compile(r'initial bubble radius is\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Extract the numerical value as a string and convert it to float
                print(match.group(1))
                number_str = match.group(1).replace('d', 'e').replace('D', 'e')
                return float(number_str)
    print(f'VALUE NOT FOUND IN OUT FILE: {name}')
    return 0.5

def find_latest_out_prime_file(directory):
    out_prime_files = []
    mons_out = 'out_prime.log'
    for filename in os.listdir(directory):
        if filename.startswith("out_prime") and filename.endswith(".log"):
            if filename == mons_out:
                return os.path.join(directory, mons_out)
            try:
                # Extract the numerical part after "out_prime_" and before ".log"
                numerical_value = int(re.findall(r'out_prime_(\d+)_\d+\.log', filename)[0])
                out_prime_files.append((numerical_value, filename))
            except IndexError:
                continue

    # Sort the files by their numerical value and get the highest one
    if out_prime_files:
        latest_file = min(out_prime_files, key=lambda x: x[0])[1]
        print(latest_file)
        return os.path.join(directory, latest_file)
    return None
