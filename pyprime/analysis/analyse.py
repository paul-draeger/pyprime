#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 21:27:00 2023

@author: paul
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from .sim import sim
from scipy.fft import fft
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
import os
import random


def interpolate_to_uniform_grid(x, y, u, v,p, ymin, ymax, xmax, grid_res=100, method='nearest'):
    from scipy.interpolate import griddata
    """
    Interpolate velocity data from unstructured grid to a uniform meshgrid.

    Parameters:
    - x, y: Arrays of unstructured grid point coordinates.
    - u, v: Arrays of velocity components at each (x, y) point.
    - grid_resolution: Number of points in x and y directions for the uniform grid.
    - method: Interpolation method ('linear', 'nearest', 'cubic').

    Returns:
    - X, Y: 2D meshgrid coordinates.
    - U, V: Interpolated velocity components on the uniform grid.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)
    p = np.asarray(p)

    # Create uniform grid covering the bounding box of the data
    xi = np.linspace(0, xmax, grid_res)
    yi = np.linspace(ymin, ymax, grid_res)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate u and v velocities separately
    U = griddata((x, y), u, (X, Y), method=method)
    V = griddata((x, y), v, (X, Y), method=method)
    P = griddata((x, y), p, (X, Y), method=method)

    return X, Y, U, V, P


def void_I(sim,i,y,t):
    VF = np.zeros((len(y),len(t)))
    vG = np.zeros((len(y),len(t)))
    Ly = sim.Ly
    print('Ellipsoid '+format(i,'d'))
    e = sim.e(i)
    r = e.d / 2
    for j in range(len(y)):
        #print(f"Ell. {i}, y {j}")
        for k in range(len(t)):
            dy = e.yi(t[k]) - y[j]
            dy = np.min(np.abs([dy, dy + Ly, dy - Ly ]))
            inside = dy < r
            vf = inside * (r**2 - dy**2)*np.pi
            VF[j,k] += vf
            vG[j,k] += e.vi(t[k])*vf
    return VF,vG

def voidf(sim,y,t):
    import concurrent.futures

    i = range(1,sim.eNr+1)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(void_I,[sim]*len(i), i, [y]*len(i),[t]*len(i)))
    VF,vG = zip(*results)
    VF = np.array(VF)
    #print(VF.shape)
    VF = np.sum(VF,0)

    vG = np.array(vG)
    vG = np.sum(vG,0)

    vm = vG/VF
    VF = VF / (sim.Lx*sim.Lz)
    vm[vm==0] = np.nan
    return VF, vG, vm


def tke_lines(s1,Ns=30,newbinary=0):
    def x_line(t):
        exyz = s1.exyz(t)
        di2 = s1.d**2
        noell = True
        randX = np.random.permutation(range(nx))
        for i in randX:
            x = Xu[:,i]
            y = Yu[:,i]
            z = Zu[:,i]
            for j in range(ny):
                edl = (exyz[:,0]-x[j])**2 + (exyz[:,1]-y[j])**2 + (exyz[:,2]-z[j])**2
                if np.any(edl<di2):
                    noell=False
                    #print('Found Collsion - forget x Line')
                    break
            if noell:
                return i
        return -1

    def y_line(t):
        exyz = s1.exyz(t)
        randY = np.random.permutation(range(ny))
        di2 = s1.d**2
        noell = True
        for i in randY:
            x = Xu[i,:]
            y = Yu[i,:]
            z = Zu[i,:]
            for j in range(nx):
                edl = (exyz[:,0]-x[j])**2 + (exyz[:,1]-y[j])**2 + (exyz[:,2]-z[j])**2
                if np.any(edl<di2):
                    noell=False

                    break
            if noell:
                return i
        return -1

    sspath = s1.resultpath + '/fl_slices_tkeL'
    try:
        os.mkdir(sspath)
    except:
        pass
    if newbinary:
        slicepath = s1.resultpath + '/temp_spectral'
        slpu = slicepath+'/'
        Xu = s1.Xu[:,:,round(s1.Lz/2)]
        Yu = s1.Yu[:,:,round(s1.Lz/2)]
        Zu = s1.Zu[:,:,round(s1.Lz/2)]
        nx = s1.nx
        ny = s1.ny

        ux = []
        uy = []
        vx = []
        vy = []
        files = os.listdir(slicepath)
        #random.shuffle(files)
        fi = []
        for file in files:
            fi.append(extract_numeric(file))
        fi = np.array(fi)
        inds = np.unique(fi)
        inds = np.random.permutation(inds)
        #inds = inds[inds<s1.tei-1]
        # try:
        #     tfi = self.tf[inds]
        #     tcv = s1.t_conv
        #     inds = inds[tfi>tcv]
        # except:
        #     pass
        p = -1
        while len(ux)<Ns and p<(len(inds)-1):
            p+=1
            ii = inds[p]
            print(f"Fluid-Slice {ii} von {len(inds)}")
            file_inds = np.argwhere(fi==ii)

            if len(file_inds)==3:
                fileii = []
                for fii in file_inds:
                    fileii.append(files[int(fii)])
                for fileN in fileii:
                    print(f"Dateien: {fileN}")
                    if 'u' in fileN:

                        u = np.load(slpu+fileN)
                    if 'v' in fileN:

                        v = np.load(slpu+fileN)
                    # if 'w' in fileN:
                    #     w = np.load(slpu+fileN)
                if (len(s1.tf)-1)>ii:
                    t = s1.tf[ii]
                    ix = x_line(t)
                    iy = y_line(t)
                    if ix>=0:
                        ux.append(u[:,ix])
                        print(f"added x line {ix}")
                    if iy>=0:
                        uy.append(u[iy,:])
                        print(f"added y line {ix}")
                    if ix>=0:
                        vx.append(v[:,ix])
                    if iy>=0:
                        vy.append(v[iy,:])
        ux = np.array(ux)
        uy = np.array(uy)
        vx = np.array(vx)
        vy = np.array(vy)
        np.save(sspath+'/ux.npy',ux)
        np.save(sspath+'/uy.npy',uy)
        np.save(sspath+'/vx.npy',vx)
        np.save(sspath+'/vy.npy',vy)
    else:
        ux = np.load(sspath+'/ux.npy')
        uy = np.load(sspath+'/uy.npy')
        vx = np.load(sspath+'/vx.npy')
        vy = np.load(sspath+'/vy.npy')

    return ux,uy,vx,vy


def voronoi_Tesselation(s1,ti):

    def find_value_index(arr, value):
        for index, element in enumerate(arr):
            if element == value:
                return index
        return -1  # If the value is not found in the array

    Ex = np.ones((s1.eNr,3))
    for i in range(0,s1.eNr):
        e = s1.e(i+1)
        Ex[i,0] = e.xi(ti)
        Ex[i,1] = e.yi(ti)
        Ex[i,2] = e.zi(ti)

    x_per = np.array([0, -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
    y_per = np.array([0, -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
    z_per = np.array([0, -1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
    x_per = x_per * s1.Lx
    y_per = y_per * s1.Ly
    z_per = z_per * s1.Lz

    ExP = Ex
    for l in range(1,28):
        Exl = np.column_stack((Ex[:,0]+x_per[l], Ex[:,1]+y_per[l], Ex[:,2]+z_per[l]))
        ExP = np.row_stack((ExP,Exl))

        # Compute the Voronoi diagram
    vor = Voronoi(ExP)

    o = 0
    regions = vor.regions
    Rind = np.zeros(len(regions),dtype=int)
    VolC = np.zeros(len(regions))
    regI = np.zeros(len(regions))
    inside = vor.point_region[0:50]
    vorE = np.ones((len(regions),3))*-1
    ang = np.ones(len(regions))*-1
    aR = np.ones(len(regions))*-1

    for region_index, region in enumerate(vor.regions):
        vertices = [vor.vertices[i] for i in region]
        regI[o] = find_value_index(inside, region_index)
        if regI[o]!=-1:
            Rind[o] = 1
            VolC[o] = ConvexHull(vertices).volume
            eigvals, eigvecs = np.linalg.eig(np.cov(np.transpose(np.stack(vertices))))
            vorE[o,:] = eigvecs[:,np.argmax(eigvals)]
            ang[o] = np.degrees(np.arccos(np.dot(vorE[o,:], [0, 1, 0])))
            aR[o] = np.max(eigvals)/np.min(eigvals)
        else:
            Rind[o] = 0
            VolC[o] = -1
        o += 1

    VolC = VolC[VolC!=-1]
    regI = regI[regI!=-1]
    ang  = ang[ang!=-1]
    aR   = aR[aR!=-1]
    vorE = vorE[np.any(vorE != -1, axis=1)]

    VolC = VolC[regI.argsort()]
    regI = regI[regI.argsort()]
    vor.regions = [x for x, indI in zip(vor.regions, Rind) if indI == 1]

    return vor, VolC, ang, aR



def analyse_Re(sim, ti, allRe=False, empty=False):
    Ne = sim.eNr
    N = len(ti)
    Remgk = []
    Restdgk = []
    GKs = []
    if allRe: Reigk = []
    dg = np.zeros(4)
    if not sim.continous:
        for gk in range(1,4):
            Regk = []
            for i in range(0,Ne):
                e = sim.e(i+1)
                if e.GK==gk:
                    Regk.append(e.Rei(ti))
            if len(Regk)>0 or empty:
                GKs.append(gk)
                Regk = np.array(Regk)
                Remgk.append(np.mean(Regk,0))
                Restdgk.append(np.std(Regk,0))
                if allRe: Reigk.append(Regk)
    else:
        d = sim.d
        d0 = np.min(d)
        dd = ( np.max(d) - np.min(d) ) / 3
        dg = [d0, d0+dd, d0+dd*2, d0+dd*3]
        for gk in range(1,4):
            Regk = []
            for i in range(0,Ne):
                e = sim.e(i+1)
                gki = 0
                for dgi in dg:
                    if e.d > dgi:
                        gki+=1
                    else:
                        break
                if gki==gk:
                    Regk.append(e.Rei(ti))
            if len(Regk)>0 or empty:
                GKs.append(gk)
                Regk = np.array(Regk)
                Remgk.append(np.mean(Regk,0))
                Restdgk.append(np.std(Regk,0))
        for j in range(sim.eNr):
            if allRe:
                e = sim.e(j+1)
                Reigk.append(e.Rei(ti))
        Reigk = np.array(Reigk)

    if not allRe:
        return GKs, Remgk,Restdgk,dg
    else:
        return GKs, Remgk,Restdgk,dg,Reigk

def violin_data(sims, N = 10000):
    print('1/2 Reading Data')
    for k,sim in enumerate(sims):
        ti = np.linspace(sim.t_conv, sim.te ,N)
        Ne = sim.eNr

        pReE = []
        pdi = []
        pgk= []
        for i in range(0,Ne):
            print('Reading Elliposid'+format(i,'d') + '/' + format(Ne,'d') )
            e = sim.e(i+1)
            pReE = np.append(pReE, e.Rei(ti))
            if any(pReE<0):
                print(pReE)
                print(i)
                print(sim.label)
            pdi = np.append(pdi, e.d*np.ones(N))
            pgk = np.append(pgk, e.GK*np.ones(N))
        disp = np.ones(N*Ne)*np.size(sim.du)
        eg = np.ones(N*Ne)*sim.e_g
        df = pd.DataFrame({
        'Re': pReE,
        'di': pdi,
        'eg': eg,
        'disp': disp,
        'gk': pgk,
        'sim': sim.label
        })
        if k==0:
            dfv = df
        else:
            dfv = pd.concat([dfv, df], axis=0, ignore_index=True)

    return dfv


def collect_data():
    print('Datensammlungs-Programm')
    dirs = ['01_Mo-5', '02_Tri-5', '03_Bi-5', '04_Mo-10', '05_Tri-10', '06_Bi-10', '07_MoSm-10', '08_MoLa-10', '09_Mo-15', '10_Tri-15', '11_Bi-15']
    e_g = np.repeat(['5','10','15'],[3,5,3])
    disp = [1,3,2,1,3,2,1,1,1,3,2]
    ID = range(1,12)
    i = 0
    df = []
    dftke = []
    for si in dirs:
        print('Sammle Daten, Sim-'+format(i,'d'))
        s1 = sim('../'+si+'/results',newbinary=1)
        dfi = s1.dfE()
        dfti = s1.dfTKE()
        li = len(dfi)
        lti = len(dfti)

        dfi['e_g'] = np.repeat(e_g[i],li)
        dfi['sim'] = np.repeat(ID[i],li)
        dfi['poly'] = np.repeat(disp[i],li)

        dfti['e_g'] = np.repeat(e_g[i],lti)
        dfti['sim'] = np.repeat(ID[i],lti)
        dfti['poly'] = np.repeat(disp[i],lti)


        if i==0:
            df = dfi
            dftke = dfti
        else:
            df = pd.concat([df, dfi], axis=0).reset_index(drop=True)
            dftke = pd.concat([dftke, dfti], axis=0).reset_index(drop=True)
        i += 1
        df.to_parquet('E_Data.parquet', index=False)
        dftke.to_parquet('TKE_Data.parquet', index=False)
    df.to_parquet('E_Data.parquet', index=False)
    dftke.to_parquet('TKE_Data.parquet', index=False)


def make_videos(current=True):
    from .visualize import vis_videos
    print('Datensammlungs-Programm')
    dirs = ['01_Mo-5', '02_Tri-5', '03_Bi-5', '04_Mo-10', '05_Tri-10', '06_Bi-10', '07_MoSm-10', '08_MoLa-10', '09_Mo-15', '10_Tri-15', '11_Bi-15']
    Re = (400,400,400,400,400,400,400,400,400,400,400)
    i = 0
    for si in dirs:
        print('Erstelle Videos, Sim-'+format(i,'d'))
        s1 = sim('../'+si+'/results',newbinary=0)
        ui = Re[i]*s1.vis/2.1
        vis_videos(s1,ui,current=current)
        i += 1


def calc_angles(i,N,x,y,z,gk,x_per,y_per,z_per,rq):
    Ne = len(x)
    q = 0
    rp = np.zeros(N*Ne)
    phi = np.zeros(N*Ne)
    gki = np.zeros(N*Ne)
    gkj = np.zeros(N*Ne)

    gk1 = gk[i]
    x1 = x[i]
    y1 = y[i]
    z1 = z[i]
    print('E-'+format(i,'d')+' ')
    for k in range(0,Ne):
        if k!=i:
            x2 = x[k]
            y2 = y[k]
            z2 = z[k]
            gk2 = gk[k]
            for p in range(N):
                dx0 = x1[p]-x2[p]
                dy0 = y1[p]-y2[p]
                dz0 = z1[p]-z2[p]

                for l in range(0,27):
                    dx = dx0 + x_per[l]
                    dy = dy0 + y_per[l]
                    dz = dz0 + z_per[l]

                    r = dx**2 + dy**2 + dz**2
                    if r<=rq:
                        r = np.sqrt(r)
                        rp[q] = r
                        phi[q] = np.arccos(dy/r)
                        gki[q] = gk1
                        gkj[q] = gk2
                        q += 1
                        break
    mask = rp!=0
    return rp[mask], phi[mask], gki[mask], gkj[mask]



def angular_conv(sim, N=10, newbinary=1):
    import concurrent.futures

    path = sim.resultpath + '/ellipsoid/ell_binary/A'+ format(N,"d")
    if newbinary ==1:
        ta = sim.t_conv
        rmax = sim.Lx/2
        rmax_sq = rmax**2
        eN = sim.eNr
        x_per = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
        y_per = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
        z_per = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
        x_per = x_per * sim.Lx
        y_per = y_per * sim.Ly
        z_per = z_per * sim.Lz
        t = np.linspace(ta, sim.te, N)
        dsize = eN**2 * N * 27
        print('1/6 Calculating Angle and Distance for up to '+format(dsize, "d")+' Ellipsoid Pairs')

        phipdf = np.zeros((dsize,))
        gki = np.zeros((dsize,))
        gkj = np.zeros((dsize,))
        eidx = np.zeros((dsize,))
        p = 0

        #Preprocessing Ellipsoid Data
        x_list = []
        y_list = []
        z_list = []
        gkl = []
        i = []
        for j in range(1,sim.eNr+1):
            e = sim.e(j)
            x_list.append(e.xi(t))
            y_list.append(e.yi(t))
            z_list.append(e.zi(t))
            gkl.append(e.GK)
            i.append(j-1)

        x_list = [x_list]*eN
        y_list = [y_list]*eN
        z_list = [z_list]*eN
        gk     = [gkl]*eN
        Nlist = [N]*eN
        rmq_l = [rmax_sq]*eN
        xp = [x_per]*eN
        yp = [y_per]*eN
        zp = [z_per]*eN

        #Processing Pairs (via Multithreading)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(calc_angles, i, Nlist, x_list,y_list,z_list,gk,xp,yp,zp,rmq_l))
        Lr,Lphi,Lgki,Lgkj = zip(*results)
        r = np.concatenate(Lr)
        del Lr
        phi = np.concatenate(Lphi)
        del Lphi
        gki = np.concatenate(Lgki)
        del Lgki
        gkj = np.concatenate(Lgkj)
        del Lgkj

        np.save(path+'ell_ang.npy', phi)
        np.save(path+'ell_r.npy', r)
        np.save(path+'ell_gki.npy', gki)
        np.save(path+'ell_gkj.npy', gkj)
    else:
        phi = np.load(path+'ell_ang.npy')
        r = np.load(path+'ell_r.npy')
        gki = np.load(path+'ell_gki.npy')
        gkj = np.load(path+'ell_gkj.npy')
    return r, phi, gki, gkj


def angular_conv_bub(sim, N=10000, newbinary=1,NI=100000):
    import concurrent.futures

    path = sim.resultpath + '/bubble/bub_binary/A'+ format(N,"d")
    if newbinary ==1:
        ta = sim.b(1).t[NI[0]]
        rmax = sim.Lx/2
        rmax_sq = rmax**2
        eN = sim.bNr
        x_per = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
        y_per = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
        z_per = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
        x_per = x_per * sim.Lx
        y_per = y_per * sim.Ly
        z_per = z_per * sim.Lz
        t = np.linspace(ta, sim.b(1).t[NI[1]], N)
        dsize = eN**2 * N * 27
        print('1/6 Calculating Angle and Distance for up to '+format(dsize, "d")+' Ellipsoid Pairs')

        p = 0

        #Preprocessing Ellipsoid Data
        x_list = []
        y_list = []
        z_list = []
        i = []
        for j in range(1,sim.bNr+1):
            print(f'Read bubble ({j}/{sim.bNr})')
            e = sim.b(j)
            x_list.append(e.xi(t))
            y_list.append(e.yi(t))
            z_list.append(e.zi(t))
            i.append(j-1)

        x_list = [x_list]*eN
        y_list = [y_list]*eN
        z_list = [z_list]*eN
        Nlist = [N]*eN
        rmq_l = [rmax_sq]*eN
        xp = [x_per]*eN
        yp = [y_per]*eN
        zp = [z_per]*eN

        #Processing Pairs (via Multithreading)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(calc_angles_bub, i, Nlist, x_list,y_list,z_list,xp,yp,zp,rmq_l))
        Lr,Lphi = zip(*results)
        r = np.concatenate(Lr)
        del Lr
        phi = np.concatenate(Lphi)
        del Lphi

        np.save(path+'bub_ang.npy', phi)
        np.save(path+'bub_r.npy', r)
    else:
        phi = np.load(path+'bub_ang.npy')
        r = np.load(path+'bub_r.npy')
    return r, phi



def calc_angles_bub(i,N,x,y,z,x_per,y_per,z_per,rq):
    Ne = len(x)
    q = 0
    rp = np.zeros(N*Ne)
    phi = np.zeros(N*Ne)

    x1 = x[i]
    y1 = y[i]
    z1 = z[i]
    print('B-'+format(i,'d')+' ')
    for k in range(0,Ne):
        if k!=i:
            x2 = x[k]
            y2 = y[k]
            z2 = z[k]
            for p in range(N):
                dx0 = x1[p]-x2[p]
                dy0 = y1[p]-y2[p]
                dz0 = z1[p]-z2[p]

                for l in range(0,27):
                    dx = dx0 + x_per[l]
                    dy = dy0 + y_per[l]
                    dz = dz0 + z_per[l]

                    r = dx**2 + dy**2 + dz**2
                    if r<=rq:
                        r = np.sqrt(r)
                        rp[q] = r
                        phi[q] = np.arccos(dy/r)
                        q += 1
                        break
    mask = rp!=0
    return rp[mask], phi[mask]



def cut_data(sim, ta, N=100 , tc = 0.75, ist = 1, fn = 10 ):
    nc = int(np.floor( (sim.te-ta) / tc ))
    print('Cutting each Trajectory in '+format(nc,"d")+' Segments')
    X = []
    Y = []
    T = []
    I = []
    Z = []
    E = []
    D = []
    Df = []
    Ne = sim.eNr
    Pf = np.zeros((Ne*nc,fn))
    Rf = np.zeros((Ne*nc,fn))
    Fx = np.zeros((Ne*nc,fn))
    Fy = np.zeros((Ne*nc,fn))
    Fz = np.zeros((Ne*nc,fn))
    tf = np.zeros(Ne*nc)
    If = np.zeros(Ne*nc)
    ef = np.zeros(Ne*nc)

    idx = ist
    for i in range(0,Ne):
        print('Reading Elliposid'+format(i,'d') + '/' + format(Ne,'d') )
        e = sim.e(i+1)
        t1 = ta
        t2 = ta + tc
        while t2<sim.te:
            ti = np.linspace(t1,t2,N)
            tf[idx-1] = ti[0]

            T = np.append(T, ti)
            I = np.append(I, np.ones(N)*idx)
            xp = e.xi(ti)
            dxp = np.append(0,np.diff(xp))
            dxp[np.abs(dxp) < 0.5] = 0
            dxp = np.cumsum(-dxp)
            xp += dxp
            fx = np.abs(fft(xp))
            fx = fx[0:fn]
            Fx[idx-1,:] = fx[0:fn]


            yp = e.yi(ti)
            dyp = np.append(0,np.diff(yp))
            dyp[np.abs(dyp) < 0.5] = 0
            dyp = np.cumsum(-dyp)
            yp += dyp
            fy = np.abs(fft(xp))
            fy = fy[0:fn]
            Fy[idx-1,:] = fy[0:fn]



            zp = e.zi(ti)
            dzp = np.append(0,np.diff(zp))
            dzp[np.abs(dzp) < 0.5] = 0
            dzp = np.cumsum(-dzp)
            zp += dzp
            fz = np.abs(fft(xp))
            Fz[idx-1,:] = fz[0:fn]

            zm = zp - np.mean(zp)
            xm = xp - np.mean(xp)
            phi = np.arctan(xm/zm)
            r = np.sqrt(zm**2 + xm**2)
            phif = np.abs(fft(phi))
            rf = np.abs(fft(r))
            Pf[idx-1,:] = phif[0:fn]
            Rf[idx-1,:] = rf[0:fn]


            X = np.append(X, xp)
            Y = np.append(Y, yp)
            Z = np.append(Z, zp)

            ef[idx-1] = i
            If[idx-1] = idx

            t1 += tc + tc/N
            t2 += tc
            idx += 1
        E = np.append(E, i*np.ones(N*nc))

        if e.GK==1:
            D = np.append(D, sim.du[0]*np.ones(N*nc))
            Df = np.append(Df, sim.du[0]*np.ones(nc))
        if e.GK==2:
            D = np.append(D, sim.du[1]*np.ones(N*nc))
            Df = np.append(Df, sim.du[1]*np.ones(nc))
        if e.GK==3:
            D = np.append(D, sim.du[2]*np.ones(N*nc))
            Df = np.append(Df, sim.du[2]*np.ones(nc))
    disp = np.ones(N*Ne*nc)*np.size(sim.du)
    dispf = np.ones(nc*Ne)*np.size(sim.du)

    print(np.shape(disp))
    print(np.shape(E))

    eg = np.ones(N*Ne*nc)*sim.e_g
    egf = np.ones(Ne*nc)*sim.e_g

    colFx = [f"xF{i+1}" for i in range(0,fn)]
    colFy = [f"yF{i+1}" for i in range(0,fn)]
    colFz = [f"zF{i+1}" for i in range(0,fn)]
    colP = [f"pF{i+1}" for i in range(0,fn)]
    colR = [f"rF{i+1}" for i in range(0,fn)]


    df = pd.DataFrame({
    't': T,
    'd': D,
    'x': X,
    'y': Y,
    'z': Z,
    'enr': E,
    'idx': I,
    'disp': disp,
    'eg': eg
    })

    dfi = pd.DataFrame({
    't': tf,
    'd': Df,
    'enr': ef,
    'idx': If,
    'disp': dispf,
    'eg': egf
    })

    dfx = pd.DataFrame(Fx, columns=colFx)
    print(dfx)
    dfy = pd.DataFrame(Fy, columns=colFy)
    print(dfy)
    dfz = pd.DataFrame(Fz, columns=colFz)
    print(dfz)
    dfp = pd.DataFrame(Pf, columns=colP)
    dfr = pd.DataFrame(Rf, columns=colR)

    dff = pd.concat([dfx, dfz], axis=1)
    dffp = pd.concat([dfp,dfr], axis = 1)

    return df, dfi, dff, dffp


def svf_step(s1,ti,vfI,rsqm,x_per,y_per,z_per,NP):
    vf = np.zeros((np.size(ti),s1.eNr))
    for o,t in enumerate(ti):
        for i in range(1,s1.eNr+1):
            vfi = 0
            ei = s1.e(i)
            xi = ei.xi(t)
            yi = ei.yi(t)
            zi = ei.zi(t)
            print(i)
            for j in range(1,s1.eNr):
                if i!=j:
                    ej = s1.e(j)
                    II = vfI[ej.GK-1]
                    rmax = rsqm[ej.GK-1]
                    for k in range(NP):
                        rijsq = (xi-ej.xi(t)+x_per[k])**2 + (yi-ej.yi(t)+y_per[k])**2 + (zi-ej.zi(t)+z_per[k])**2
                        if rijsq<rmax:
                            vfi += II(rijsq)
            vf[o,i-1] = vfi
    return vf


def svf_ell(s1,C_R=2,m=1,i0=1,N=100,Nsub=20, newbinary=0):
    from concurrent.futures import ProcessPoolExecutor

    def periodic_boundaries(s1,n):
        from itertools import product
        period = np.arange(-n,n+1)
        xp = list(product(period, repeat=3))
        xp = np.array(xp)
        return xp[:,0]*s1.Lx, xp[:,1]*s1.Ly, xp[:,2]*s1.Lz

    def svf(rs,r):
        if r<2*rs:
            return 1 + np.cos(np.pi/2*r/rs)
        else:
            return 0

    def svf_InterP1Dsq(d,rs,S=2.25,NI=200,newbinary=0):
        print('Berechne Interpolations-Stempel')
        rI = rs*S
        ri = np.linspace(0,rI,NI)**2
        vf = np.zeros(NI)
        VK = d**3*np.pi/6
        V_SVF = 32*rs**3*(np.pi/3-2/np.pi)
        for i in range(NI):
            vf[i] = svf(rs,np.sqrt(ri[i]))
        vf = vf/V_SVF*VK
        return interpolate.interp1d(ri,vf)

    print('Smoothed Void Fraction Analyse')
    if newbinary==1:
        ti = np.linspace(s1.t_conv,s1.te,N)

        dGK = [1.022,1.573,2.125]

        d0 = dGK[i0]
        S = 2.25
        vfI = []
        rsqm = []
        for d in dGK:
            rs = C_R*(d/d0)**m
            vfI.append(svf_InterP1Dsq(d,rs,S=S))
            rsqm.append( (rs*S)**2 )

        nP = np.ceil( np.sqrt(np.max(rsqm))/s1.Lx )
        print(f"Anzahl der Perioden für SVF: {nP}")
        x_per,y_per,z_per = periodic_boundaries(s1, nP)
        NP = np.size(x_per)

        tii = np.array_split(ti,Nsub)

        with ProcessPoolExecutor() as executor:
            vfi = list(executor.map(svf_step, [s1]*Nsub, tii, [vfI]*Nsub, [rsqm]*Nsub,[x_per]*Nsub,[y_per]*Nsub,[z_per]*Nsub,[NP]*Nsub))
        vf = np.concatenate(vfi,axis=0)
        vf = vf.T
        svf_path = s1.resultpath+'/svf'
        if not os.path.exists(svf_path): os.mkdir(svf_path)
        np.save(svf_path+f"/svf_{N}_CR_{C_R}_m_{m}.npy",vf)
        np.save(svf_path+f"/t_{N}_CR_{C_R}_m_{m}.npy",ti)

    svf_path = s1.resultpath+'/svf'
    vf = np.load(svf_path+f"/svf_{N}_CR_{C_R}_m_{m}.npy")
    t = np.load(svf_path+f"/t_{N}_CR_{C_R}_m_{m}.npy")
    return vf,t


#Kollisionsanalyse
def extract_coll(e,ne):

    cy = e.cx
    ind0 = np.arange(0,len(cy))

    cy[np.isnan(cy)] = 0


    c_ind = ( cy != 0 )
    cs_ind = np.nonzero( np.diff(c_ind) )[0] +1

    splitc = np.split(cy,cs_ind)
    ind0S  = np.split(ind0,cs_ind)
    coll = []
    Icoll = []

    for spi,spii in zip(splitc,ind0S):
        if spi[0]!=0:
            coll.append(spi)
            Icoll.append(spii)

    return coll,Icoll


def shared_coll(s1):
    ne = 0
    print(ne)
    EV = []
    IEV = []
    Ecoll = []
    ncombi=0
    for i in range(s1.eNr):
        ev,icoll = extract_coll( s1.e(i+1),ne )
        idev = []
        for j,evi in enumerate(ev):
            idev.append( np.abs(np.sum(evi)) )
            ncombi+=1
        EV.append(idev)
        Ecoll.append(ev)
        IEV.append(icoll)

    #print(Ecoll)
    combi = []
    coll = []
    gki = s1.GKi
    combigk= []
    fcg = []
    fcalpha = []
    fcga = []
    found = 0
    tc = []
    dtc = []
    for i in range(s1.eNr):
        found = []
        for j in range(i,s1.eNr):
            if i!=j:
                _,cii,cjj = np.intersect1d(EV[i],EV[j],return_indices=True,assume_unique=True)
                if np.size(cii)>0:
                    ei = s1.e(i+1)
                    for ciji, bjj in zip(cii,cjj):
                        found.append(ciji)
                        cinds = IEV[i][ciji]
                        combi.append([i,j])
                        combigk.append([ gki[i],gki[j] ])
                        cse = round(np.mean(cinds))
                        coll.append(cse)
                        fcxi = ei.cx[ cinds ]
                        fcyi = ei.cy[ cinds ]
                        fczi = ei.cz[ cinds ]
                        te = ei.t
                        tc.append( ( te[cinds[-1]] + te[cinds[0]] )/2 )
                        #Fou[j][cij[1]] = 1
                        dt = te[cinds[-1]] - te[cinds[0]]
                        dtc.append(dt)

                    EV[i] = [EV[i][p] for p in range(len(EV[i])) if p not in cii]
                    EV[j] = [EV[j][p] for p in range(len(EV[j])) if p not in cjj]

                    IEV[i] = [IEV[i][p] for p in range(len(IEV[i])) if p not in cii]
                    IEV[j] = [IEV[j][p] for p in range(len(IEV[j])) if p not in cjj]

                    Ecoll[i] = [Ecoll[i][p] for p in range(len(Ecoll[i])) if p not in cii]
                    Ecoll[j] = [Ecoll[j][p] for p in range(len(Ecoll[j])) if p not in cjj]

    # print('Mehrfachkollisionen:')
    # IEV3 = []
    # Ecoll3 = []
    # gk3 = []
    # gki3 = []
    # for i in range(s1.eNr):
    #     print(f"bei {i} bleiben {len(IEV[i])} Kollisionen zurück")
    #     if len(IEV[i])>0:
    #         for iev in IEV[i]:
    #             IEV3.append(iev)
    #         for iev in Ecoll[i]:
    #             Ecoll3.append(iev)
    #             gk3.append(gki[i])
    #             gki3.append(i)
    # ts = s1.e(2).t
    # print(len(ts))
    # print(len(s1.e(2).cx))
    # T3 = np.zeros((len(IEV3),len(ts)+1))
    # for q,(iev,ecoll) in enumerate(zip(IEV3,Ecoll3)):
    #         #print(iev)
    #         #print(ecoll)
    #         ecoll = ecoll[iev<len(ts)]
    #         iev = iev[iev<len(ts)]
    #
    #         T3[q,iev] = ecoll
    # T3i = T3!=0
    # _,t3u = np.unique(T3i,axis=1,return_index=True)
    # T3 = T3[:,t3u[1:]]
    # ctest = np.sum(T3,axis=0)
    # for k,cti in enumerate(ctest):
    #     if cti==0:
    #         ins = np.nonzero(T3[:,k])[0]
    #         for iic in range(len(ins)-1):
    #             #print(ins)
    #             combi.append([gki3[ins[iic]],gki3[ins[iic+1]]])
    #             combigk.append([ gk3[ins[iic]],gk3[ins[iic+1]] ])
    #             coll.append(t3u[k])
    #             tc.append( ts[ t3u[k] ] )
    #             #dt = ts[ t3u[k] ] - ts[ t3u[k] ]
    #             dtc.append(0)



    #print(ctest)
    #print(T3[:,0])



    combi = np.array(combi)
    combigk = np.array(combigk)
    coll = np.array(coll)
    fcg = np.array(fcg)
    fcalpha = np.array(fcalpha)
    fcga = np.array(fcga)
    tc = np.array(tc)
    dtc = np.array(dtc)

    return combi,combigk,coll,fcg,fcalpha,fcga,tc,dtc



def x_collision(s1):
    dfc = 0.1*np.max(s1.du)
    #coll = []
    collgk = []
    collt = []
    icoll = []
    collC = []
    for i in range(s1.eNr):
        print(f"Ellipsoid - {i}")
        ei = s1.e(i+1)
        te = ei.t
        gki = ei.GK
        for j in range(s1.eNr):
            ej = s1.e(j+1)
            gkj = ej.GK
            if len(ei.t)!=len(ej.t):
                xe = min([len(ei.t), len(ej.t)])
                ind_ij = np.arange(0,xe)
                dx = ei.x[ind_ij] - ej.x[ind_ij]
                dy = ei.y[ind_ij] - ej.y[ind_ij]
                dz = ei.z[ind_ij] - ej.z[ind_ij]
            else:
                dx = ei.x - ej.x
                dy = ei.y - ej.y
                dz = ei.z - ej.z


            r2 = dx**2 + dy**2 + dz**2
            k = 0
            el = len(r2)
            r2S = (ei.d/2 + ej.d/2 + dfc)**2
            r2E = ((ei.d/2 + ej.d/2 + dfc)*1.05)**2
            while k<el:
                colli=[]
                while k<el and r2[k]>r2S:
                    k+=1
                cS = k
                while k<el and r2[k]<r2E:
                    k +=1
                #coll.append(colli)
                if k!=el:
                    collgk.append( np.array([gki,gkj]) )
                    collC.append( np.array([i,j]) )
                    indt = round((cS + k)/2)
                    collt.append( te[indt] )
                    icoll.append(indt)

    collgk = np.array(collgk)
    collC = np.array(collC)
    collt = np.array(collt)
    icoll = np.array(icoll)
    return collgk,collt,icoll,collC



def GKcoll(s1,cgk,tmin,tmax):
    indE11 = np.logical_and( cgk[:,0] ==1 , cgk[:,1] ==1 )
    indE22 = np.logical_and( cgk[:,0] ==2 , cgk[:,1] ==2 )
    indE33 = np.logical_and( cgk[:,0] ==3 , cgk[:,1] ==3 )
    indE12 = np.logical_or( np.logical_and( cgk[:,0] ==2 , cgk[:,1] ==1 ), np.logical_and( cgk[:,0] ==1 , cgk[:,1] ==2 ))
    indE13 = np.logical_or( np.logical_and( cgk[:,0] ==3 , cgk[:,1] ==1 ), np.logical_and( cgk[:,0] ==1 , cgk[:,1] ==3 ))
    indE23 = np.logical_or( np.logical_and( cgk[:,0] ==2 , cgk[:,1] ==3 ), np.logical_and( cgk[:,0] ==3 , cgk[:,1] ==2 ))

    hd11 = np.sum(indE11) / ( tmax - tmin)
    hd22 = np.sum(indE22) / ( tmax - tmin)
    hd33 = np.sum(indE33) / ( tmax - tmin)

    hd12 = np.sum(indE12) / ( tmax - tmin)
    hd13 = np.sum(indE13) / ( tmax - tmin)
    hd23 = np.sum(indE23) / ( tmax - tmin)

    hc = 0
    Nc = 0
    if s1.continous:
        hc = len(cgk) / ( tmax - tmin)
        Nc = len(cgk)

    return hd11,hd22,hd33,hd12,hd13,hd23,hc,Nc

def make_coll_mat(Cgk):
    if np.max(Cgk)<4:
        gkl = 3
    else:
        gkl = 50
        print(Cgk)
    print(gkl)
    M = np.zeros((gkl,gkl))
    for ci in Cgk:
        M[ci[0]-1,ci[1]-1]+=1
    return M


def analyse_coll(sim,newbinary=0,tmin=-1,tmax=-1,getMat=False,getcombi=True):
    cpath = sim.resultpath + '/collision_stats'
    if newbinary==1 and 1==2:
        collgk,collt,icoll,collC = x_collision(sim)
        #if sim.continous: print(combi)
        try:
            os.mkdir(cpath)
        except:
            pass
        np.save(cpath + "/combi.npy",icoll)
        np.save(cpath + "/combigk.npy",collgk)
        #np.save(cpath + "/coll.npy",coll)
        np.save(cpath + "/fcg.npy",collC)
        #np.save(cpath + "/fcalpha.npy",fcalpha)
        #np.save(cpath + "/fcga.npy",fcga)
        np.save(cpath + "/tc.npy",collt)
        #np.save(cpath + "/dtc.npy",dtc)
        tc = collt
        combigk = collgk
    else:
        combi = np.load(cpath + "/combi.npy")
        combigk = np.load(cpath + "/combigk.npy")
        try:
            coll = np.load(cpath + "/coll.npy")
        except:
            pass
        fcg = np.load(cpath + "/fcg.npy")
        #fcalpha = np.load(cpath + "/fcalpha.npy")
        #fcga = np.load(cpath + "/fcga.npy")
        tc = np.load(cpath + "/tc.npy")
        #dtc = np.load(cpath + "/dtc.npy")
    if tmin==-1:
        tmin = sim.t_conv
    if tmax==-1:
        tmax = sim.te
    indT = (tc>tmin) & (tc<tmax)
    cgk = combigk[indT]


    tcdt = tc[indT]
    h11,h22,h33,h12,h13,h23,hc,Nc = GKcoll(sim,cgk,tmin,tmax)

    ind1 = np.logical_or( combigk[:,0] ==1 , combigk[:,1] ==1 )
    ind2 = np.logical_or( combigk[:,0] ==2 , combigk[:,1] ==2 )
    ind3 = np.logical_or( combigk[:,0] ==3 , combigk[:,1] ==3 )

    tc1 = tc[ind1]
    tc2 = tc[ind2]
    tc3 = tc[ind3]

    Nc = len(tcdt)

    if not getMat:
        if getcombi:
            return fcg, combigk, tc
        else:
            return h11,h22,h33,h12,h13,h23,hc,Nc,tcdt, tc1, tc2, tc3
    else:
        return make_coll_mat(cgk)


def read_coll(s1):
    cpath = s1.resultpath + '/collision_stats'
    combi = np.load(cpath + "/combi.npy")
    combigk = np.load(cpath + "/combigk.npy")
    coll = np.load(cpath + "/coll.npy")
    fcg = np.load(cpath + "/fcg.npy")
    #fcalpha = np.load(cpath + "/fcalpha.npy")
    #fcga = np.load(cpath + "/fcga.npy")
    tc = np.load(cpath + "/tc.npy")

    return combigk,tc

def get_3dphase(s1,ti):

    dx = s1.dx
    R = []
    iE = []
    xe = []
    ye = []
    ze = []
    re = []
    x_per = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
    y_per = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
    z_per = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
    x_per = x_per * s1.Lx
    y_per = y_per * s1.Ly
    z_per = z_per * s1.Lz
    Lx = s1.Lx
    Ly = s1.Ly
    Lz = s1.Lz
    nx = s1.nx
    ny = s1.ny
    nz = s1.nz


    for i in range(1,s1.eNr+1):
        e = s1.e(i)
        xi = e.xi(ti)
        yi = e.yi(ti)
        zi = e.zi(ti)
        r = e.d/2
        for l in range(27):
            xl = xi + x_per[l]
            yl = yi + y_per[l]
            zl = zi + z_per[l]
            if xl>-r and xl<Lx+r and xl>-r and yl<Ly+r and zl>-r and zl<Lz+r:
                xe.append(xl)
                ye.append(yl)
                ze.append(zl)
                re.append(r)

    ixR = []
    jyR = []
    kzR = []
    S = np.ones_like(s1.Xp)
    X = np.transpose(s1.Xp,(1,0,2))
    Y = np.transpose(s1.Yp,(1,0,2))
    Z = np.transpose(s1.Zp,(1,0,2))
    #print(np.shape(X))


    for ixe,iye,ize,ir in zip(xe,ye,ze,re):
        nxi = round(ixe/dx)
        nyi = round(iye/dx)
        nzi = round(ize/dx)
        nri = round(np.floor(ir/dx)+2)
        rs = ir**2
        for i in range(nxi-nri, nxi+nri+1):
            if i>=0 and i<nx:
                for j in range(nyi-nri, nyi+nri+1):
                    if j>=0 and j<ny:
                        for k in range(nzi-nri, nzi+nri+1):
                            if k>=0 and k<nz:
                                if (Z[i,j,k] - ize)**2 + (Y[i,j,k] - iye)**2 + (X[i,j,k] - ixe)**2 <= rs:
                                    S[j,i,k] = 0

    return S

def read_sims(newbinary=0):
    paths = Spath()
    si = []
    for k,path in enumerate(paths):
        rpath = path + '/results'
        si.append(sim(rpath,newbinary=0))
    return si


def tke_dissipation(s1,n):
    f = s1.f(n)

    du_dx, du_dy, du_dz = np.gradient(f.u, axis=(1, 0, 2))
    dv_dx, dv_dy, dv_dz = np.gradient(f.v, axis=(1, 0, 2))
    dw_dx, dw_dy, dw_dz = np.gradient(f.w, axis=(1, 0, 2))

    tau_12 = du_dy + dv_dx
    tau_13 = du_dz + dw_dx
    tau_23 = dv_dz + dw_dy
    e = -3.16955e-02/s1.dx**2 * (2*du_dx * du_dx + tau_12*du_dx + tau_13*du_dz + tau_12 * dv_dx +  2*dv_dy * dv_dy + tau_23*dv_dz + tau_13*dw_dx + tau_23*dw_dy + 2*dw_dz*dw_dz)
    del du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz, tau_12, tau_13, tau_23

    ti = s1.tf[n-1]
    S = get_3dphase(s1,ti)

    e = e*S
    dissipation = np.mean(e)

    del e

    uf = f.u - np.mean(f.u*S)
    wf = f.w - np.mean(f.w*S)
    vf = f.v - np.mean(f.v*S)

    tke = 0.5 * (uf*uf + vf*vf + wf*wf)
    tkem = np.mean(tke*S)
    return tkem,dissipation





def get_spec_uvw(s1,newbinary=0):
    stpath = s1.resultpath + '/temp_spectral'
    stspath = s1.resultpath + '/temp_spectral_conv'
    if newbinary:
        print("### Spectral Analysis started###")
        if not 'Q' in s1.label:
            ix = [round(s1.nx/8),round(s1.nx*3/8),round(s1.nx*5/8),round(s1.nx*7/8)]
            iy = [round(s1.ny/16),round(s1.ny*3/16),round(s1.ny*5/16),round(s1.ny*7/16),round(s1.ny*9/16),round(s1.ny*11/16),round(s1.ny*13/16),round(s1.ny*15/16)]
        else:
            print('Meshgrid')
            ix = [round(s1.nx/16),round(s1.nx*3/16),round(s1.nx*5/16),round(s1.nx*7/16),round(s1.nx*9/16),round(s1.nx*11/16),round(s1.nx*13/16),round(s1.nx*15/16)]
            iy = [round(s1.ny/16),round(s1.ny*3/16),round(s1.ny*5/16),round(s1.ny*7/16),round(s1.ny*9/16),round(s1.ny*11/16),round(s1.ny*13/16),round(s1.ny*15/16)]

        Ix,Iy = np.meshgrid(ix,iy)

        xg = Ix.ravel()
        yg = Iy.ravel()

        xl = s1.xce[xg]
        yl = s1.yce[yg]

        tf = s1.tf
        indspec = np.argwhere(tf>s1.te)
        lind = len(indspec)
        uvw = np.zeros((len(indspec),3,np.size(xg)))
        eongrid = np.zeros((len(indspec),np.size(xg)))

        print(lind)
        ti = tf[indspec]
        for k,i in enumerate(indspec):
            print(f"Reading {k}/{lind}")
            inn = (i+1)
            ii = inn.item()
            ufile = stpath + f"/u_{ii}.npy"
            vfile = stpath + f"/v_{ii}.npy"
            wfile = stpath + f"/w_{ii}.npy"
            print(ufile)
            if os.path.exists(ufile) and os.path.exists(vfile) and os.path.exists(wfile):
                ui = np.load(ufile)
                uvw[k,0,:] = ui[yg,xg]

                vi = np.load(vfile)
                uvw[k,1,:] = vi[yg,xg]

                wi = np.load(wfile)
                uvw[k,2,:] = wi[yg,xg]

                for i in range(s1.eNr):
                    ei = s1.e(i+1)
                    xe = ei.xi(ti[k])
                    ye = ei.yi(ti[k])
                    ze = ei.zi(ti[k])
                    rs = (ei.d/2)**2
                    for p,(xil,yil) in enumerate(zip(xl,yl)):
                        if (xe-xil)**2 + (ye-yil)**2 < rs:
                            eongrid[k,p] = 1
                print('Read')
        try:
            os.mkdir(stspath)
        except:
            pass
        np.save(stspath+"/collected_binary.npy",uvw)
        np.save(stspath+"/collected_binary_ti.npy",ti)
        np.save(stspath+"/collected_binary_eongrid.npy",eongrid)
    else:
        uvw = np.load(stspath+"/collected_binary.npy")
        ti = np.load(stspath+"/collected_binary_ti.npy")
        eongrid = np.load(stspath+"/collected_binary_eongrid.npy")
    return uvw,eongrid,ti


from scipy.special import sph_harm
from scipy.spatial import SphericalVoronoi, geometric_slerp
import pickle

def fibonacci_sphere(NL,new=False):
    #theta, phi, FAC, regions
    dir = '/home/s5941119/Documents/sphere_dist'
    os.makedirs(dir,exist_ok=True)
    file_theta   = dir + f'/theta_{NL}.npy'
    file_phi     = dir + f'/phi_{NL}.npy'
    file_FAC     = dir + f'/FAC_{NL}.npy'
    file_regions = dir + f'/regions_{NL}.pkl'

    if new:
        points = []
        phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
        R=1

        x = np.zeros(NL)
        y = np.zeros(NL)
        z = np.zeros(NL)
        for i in range(NL):
            y1 = 1 - (i / float(NL - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y1 * y1)  # radius at y
            y[i] = y1

            theta = phi * i  # golden angle increment

            x[i] = np.cos(theta) * radius
            z[i] = np.sin(theta) * radius

        theta = np.arctan2(x, z) + np.pi
        phi = np.arccos(y / R)
        #SC = np.column_stack([theta,phi])
        X = np.column_stack([x,y,z])
        sv = SphericalVoronoi(X, R, (0,0,0))
        faces = sv._simplices
        regions = []
        if 1==2:
            for i in range(NL):
                if i%10==0: print(i)
                regi = []
                for fac in faces:
                    if i in fac:
                        for faci in fac:
                            if faci != i: regi.append(faci)
                regions.append(regi)

            #regions = sv.regions
            nf = len(faces)
            FAC = []
            for k in range(0,nf):
                FAC.append([3, faces[k,0], faces[k,1], faces[k,2] ])
            FAC = np.hstack(FAC)

        np.save(file_theta,theta)
        np.save(file_phi,phi)
        np.save(file_FAC,faces)
        with open(file_regions, 'wb') as file:
            pickle.dump(regions, file)

    else:
        theta = np.load(file_theta)
        phi = np.load(file_phi)
        faces = np.load(file_FAC)


    return theta, phi, faces


def fibonacci_sphere_Simple(NL):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    R=1

    x = np.zeros(NL)
    y = np.zeros(NL)
    z = np.zeros(NL)
    for i in range(NL):
        y1 = 1 - (i / float(NL - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y1 * y1)  # radius at y
        y[i] = y1

        theta = phi * i  # golden angle increment

        x[i] = np.cos(theta) * radius
        z[i] = np.sin(theta) * radius

    phi = np.arctan2(x, y) #+ np.pi
    theta = np.arccos(z / 1)
    #SC = np.column_stack([theta,phi])
    X = np.column_stack([x,y,z])
    sv = SphericalVoronoi(X, R, (0,0,0))
    faces = sv._simplices

    return theta, phi, faces

def fibonacci_points(NL):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    R=1

    x = np.zeros(NL)
    y = np.zeros(NL)
    z = np.zeros(NL)
    for i in range(NL):
        y1 = 1 - (i / float(NL - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y1 * y1)  # radius at y
        y[i] = y1

        theta = phi * i  # golden angle increment

        x[i] = np.cos(theta) * radius
        z[i] = np.sin(theta) * radius

    phi = np.arctan2(x, y) #+ np.pi
    theta = np.arccos(z / 1)
    #SC = np.column_stack([theta,phi])
    X = np.column_stack([x,y,z])

    return theta, phi



def first_principal_axis(points):
    """
    Calculate the normalized first principal axis (eigenvector) of a set of 3D points.

    Parameters:
    points (np.ndarray): A Nx3 numpy array where each row is a point (x, y, z).

    Returns:
    np.ndarray: A 1x3 numpy array representing the normalized first principal axis.
    """
    # Center the data
    mean = np.mean(points, axis=0)
    centered_data = points - mean

    # Calculate the covariance matrix
    cov_matrix = np.cov(centered_data.T)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvectors by eigenvalues (in descending order)
    idx = np.argsort(eigenvalues)[::-1]
    first_principal_axis = eigenvectors[:, idx[-1]]

    # Normalize the first principal axis
    normed_first_axis = first_principal_axis / np.linalg.norm(first_principal_axis)

    return normed_first_axis


def calculate_spherical_harmonics_value(coefficients, theta, phi):
    """
    Calculate the value of the function represented by spherical harmonics coefficients
    at given angles θ (theta) and φ (phi).

    :param coefficients: List of tuples, where each tuple is (l, m, coefficient).
    :param theta: Angle θ in radians (polar angle, 0 <= θ <= π).
    :param phi: Angle φ in radians (azimuthal angle, 0 <= φ < 2π).
    :return: The computed value at the specified angles.
    """
    value = 0.0

    for l, m, coefficient in coefficients:
        # sph_harm(m, l, phi, theta) computes the spherical harmonic Y_lm(theta, phi)
        value += coefficient * sph_harm(m, l, phi, theta)

    return value.real  # The function value is usually real


def sphX(s1,I,NLI=2000,NF=8,newmesh=False,bubi=1):
    cnm_path = s1.resultpath + f'/bubble/bub_{bubi:03d}_cnm.bin'

    # A = []
    # with open(cnm_path, 'r') as file:
    #     # Iterate over each line in the file
    #     for line in file:
    #         values = line.split()
    #         for value in values:
    #             A.append( float(value) )
    with open(cnm_path, 'rb') as fin:
        A = np.fromfile(fin, dtype=np.float64,offset=4)
    #A = np.array(A)
    #print(np.shape(A))

    #NF = 12
    NL1 = (NF+1)*(NF+2)/2
    NL = NL1*2
    #print(NL)

    NL = round(NL) + 2
    Ncof = round ( NL*3 )
    #print(f'Number of Coefficients: {Ncof}')
    lA = len(A)
    maxcoef = lA%Ncof
    A = A[0:-maxcoef]
    cnm = np.reshape(A,[-1,Ncof])

#    tsh = b[:,0]
    #cnm = b[:,:]

    cnmx = cnm[:,0:NL]
    cnmy = cnm[:,NL:(2*NL)]
    cnmz = cnm[:,(2*NL):(3*NL)]
    NLh = round(NL/2)
    # anmxR = cnmx[:,::2]
    # anmxI = cnmx[:,1::2]
    #
    # anmyR = cnmy[:,::2]
    # anmyI = cnmy[:,1::2]
    #
    # anmzR = cnmz[:,::2]
    # anmzI = cnmz[:,1::2]
    anmxR = cnmx[ : , 0:NLh           ]
    anmxI = cnmx[ : , NLh:(2*NLh)     ]

    anmyR = cnmy[ : , 0:NLh           ]
    anmyI = cnmy[ : , NLh:(2*NLh)     ]

    anmzR = cnmz[ : , 0:NLh           ]
    anmzI = cnmz[ : , NLh:(2*NLh)     ]

    Naa = 0

    I_end = np.shape(anmxR)[0]

    TT,PP, faces = fibonacci_sphere(NLI,new=newmesh)


    valx = np.zeros(NLI)
    valy = np.zeros(NLI)
    valz = np.zeros(NLI)

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)
        anmxR[I,Zh] = 0.5 * anmxR[I,Zh]
        anmyR[I,Zh] = 0.5 * anmyR[I,Zh]
        anmzR[I,Zh] = 0.5 * anmzR[I,Zh]

        for m in range(0,n+1):
           Naa += 1
           zz = round( n*(n+1)/2 + m )

           Ysh = sph_harm(m, n, TT, PP)
           valxi =  complex( anmxR[I,zz], anmxI[I,zz] ) * Ysh
           valyi =  complex( anmyR[I,zz], anmyI[I,zz] ) * Ysh
           valzi =  complex( anmzR[I,zz], anmzI[I,zz] ) * Ysh
           # valxi =  anmxR[it,zz] * sph_harm(m, n, TT, PP)
           # valyi =  anmyR[it,zz] * sph_harm(m, n, TT, PP)
           # valzi =  anmzR[it,zz] * sph_harm(m, n, TT, PP)
           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )
    valx = valx*2
    valy = valy*2
    valz = valz*2
    X = np.column_stack([valx,valy,valz])
    return X, I_end


def sphX_fac(s1,I,NLI=2000,NF=8,newmesh=False):
    cnm_path = s1.resultpath + '/bubble/bub_001_cnm.bin'

    # A = []
    # with open(cnm_path, 'r') as file:
    #     # Iterate over each line in the file
    #     for line in file:
    #         values = line.split()
    #         for value in values:
    #             A.append( float(value) )
    with open(cnm_path, 'rb') as fin:
        A = np.fromfile(fin, dtype=np.float64,offset=4)
    #A = np.array(A)
    #print(np.shape(A))

    #NF = 12
    NL1 = (NF+1)*(NF+2)/2
    NL = NL1*2
    #print(NL)

    NL = round(NL) + 2
    Ncof = round ( NL*3 )
    #print(f'Number of Coefficients: {Ncof}')
    lA = len(A)
    maxcoef = lA%Ncof
    A = A[0:-maxcoef]
    cnm = np.reshape(A,[-1,Ncof])

#    tsh = b[:,0]
    #cnm = b[:,:]

    cnmx = cnm[:,0:NL]
    cnmy = cnm[:,NL:(2*NL)]
    cnmz = cnm[:,(2*NL):(3*NL)]
    NLh = round(NL/2)
    # anmxR = cnmx[:,::2]
    # anmxI = cnmx[:,1::2]
    #
    # anmyR = cnmy[:,::2]
    # anmyI = cnmy[:,1::2]
    #
    # anmzR = cnmz[:,::2]
    # anmzI = cnmz[:,1::2]
    anmxR = cnmx[ : , 0:NLh           ]
    anmxI = cnmx[ : , NLh:(2*NLh)     ]

    anmyR = cnmy[ : , 0:NLh           ]
    anmyI = cnmy[ : , NLh:(2*NLh)     ]

    anmzR = cnmz[ : , 0:NLh           ]
    anmzI = cnmz[ : , NLh:(2*NLh)     ]

    Naa = 0

    I_end = np.shape(anmxR)[0]

    TT,PP, faces, regions = fibonacci_sphere(NLI,new=newmesh)


    valx = np.zeros(NLI)
    valy = np.zeros(NLI)
    valz = np.zeros(NLI)

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)
        anmxR[I,Zh] = 0.5 * anmxR[I,Zh]
        anmyR[I,Zh] = 0.5 * anmyR[I,Zh]
        anmzR[I,Zh] = 0.5 * anmzR[I,Zh]

        for m in range(0,n+1):
           Naa += 1
           zz = round( n*(n+1)/2 + m )

           Ysh = sph_harm(m, n, TT, PP)
           valxi =  complex( anmxR[I,zz], anmxI[I,zz] ) * Ysh
           valyi =  complex( anmyR[I,zz], anmyI[I,zz] ) * Ysh
           valzi =  complex( anmzR[I,zz], anmzI[I,zz] ) * Ysh
           # valxi =  anmxR[it,zz] * sph_harm(m, n, TT, PP)
           # valyi =  anmyR[it,zz] * sph_harm(m, n, TT, PP)
           # valzi =  anmzR[it,zz] * sph_harm(m, n, TT, PP)
           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )
    valx = valx*2
    valy = valy*2
    valz = valz*2
    X = np.column_stack([valx,valy,valz])
    return X, faces


#def fft_cut(df):


    #
    #     i = 0
    #     x1 = np.zeros((dsize,3))
    #     x2 = np.zeros((dsize,3))
    #     gkij = np.zeros((dsize,2))
    #     for j in range(1,sim.eNr+1):
    #         ej = sim.e(j)
    #         print('Ellipsoid ' + format(j, "d") + '/' + format(sim.eNr, "d"))
    #         for k in range(j,sim.eNr+1):
    #             ek = sim.e(j)
    #             x1[i:(i+N),0] = ej.xi(t)
    #             x1[i:(i+N),1] = ej.yi(t)
    #             x1[i:(i+N),2] = ej.zi(t)
    #
    #             x2[i:(i+N),0] = ek.xi(t)
    #             x2[i:(i+N),1] = ek.yi(t)
    #             x2[i:(i+N),2] = ek.zi(t)
    #
    #             gkij[i:(i+N),0] = ej.GK
    #             gkij[i:(i+N),1] = ek.GK
    #             i += N
    #
    #     print('Diffrence')
    #     print('dx')
    #     dx = np.subtract(x1[:,0],x2[:,0])
    #     print('dy')
    #     dy = np.subtract(x1[:,1],x2[:,1])
    #     print('dz')
    #     dz = np.subtract(x1[:,2],x2[:,2])
    #     del x1
    #     del x2
    #     print('Repeat')
    #     lx = len(dx)
    #     dx = da.repeat(dx,27)
    #     x_per_r = da.from_array(np.repeat(x_per,lx), chunks = dx.chunks)
    #     dx = dx + x_per_r
    #     del x_per_r
    #
    #     dy = da.repeat(dy,27)
    #     y_per_r = da.from_array(np.repeat(y_per,lx), chunks = dy.chunks)
    #     dy = dy + y_per_r
    #     del y_per_r
    #
    #     dz = da.repeat(dz,27)
    #     y_per_r = da.from_array(np.repeat(z_per,lx), chunks = dz.chunks)
    #     dz = dz + z_per_r
    #     del z_per_r
    #
    #
    #
    #     dy = da.repeat(dy,27) + np.repeat(y_per,lx)
    #     dz = da.repeat(dz,27) + np.repeat(z_per,lx)
    #     print('Radius Squared')
    #     r = dx**2 + dy**2 + dz**2
    #     del dz
    #     del dx
    #     print('Mask')
    #     ind = r<rmax_sq
    #     dy = dy[ind]
    #     r = r[ind]
    #     gkij = gkij[ind,:]
    #     del ind
    #     print('Radius (Root)')
    #     r = da.sqrt(r)
    #     print('Angle')
    #     phi = da.arccos(dy/r)
    #     print('Done - Saving')
    #     np.save(path+'ell_ang.npy', phi)
    #     np.save(path+'ell_r.npy', r)
    #     np.save(path+'ell_gkij.npy', gkij)
    # else:
    #     phi = np.load(path+'ell_ang.npy')
    #     r = np.load(path+'ell_r.npy')
    #     gkij = np.load(path+'ell_gkij.npy')
    #
    #
    # return r, phi, gkij

from scipy.stats import gaussian_kde


def shear_pdf(s1,Iarray,ax1,ax2,NF=12,NLag=10000,bubi = 1,clr='r',Ntat=10014):
    import pyvista as pv

    cnm_path = s1.resultpath + f'/bubble/bub_{bubi:03d}_cnm.bin'

    with open(cnm_path, 'rb') as fin:
        A = np.fromfile(fin, dtype=np.float64,offset=4)

    NL1 = (NF+1)*(NF+2)/2
    NL = NL1*2

    NL = round(NL) + 2
    Ncof = round ( NL*3 )

    lA = len(A)
    maxcoef = lA%Ncof
    A = A[0:-maxcoef]
    cnm = np.reshape(A,[-1,Ncof])

    cnmx = cnm[:,0:NL]
    cnmy = cnm[:,NL:(2*NL)]
    cnmz = cnm[:,(2*NL):(3*NL)]
    NLh = round(NL/2)

    anmxR = cnmx[ : , 0:NLh           ]
    anmxI = cnmx[ : , NLh:(2*NLh)     ]

    anmyR = cnmy[ : , 0:NLh           ]
    anmyI = cnmy[ : , NLh:(2*NLh)     ]

    anmzR = cnmz[ : , 0:NLh           ]
    anmzI = cnmz[ : , NLh:(2*NLh)     ]

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)
        anmxR[:,Zh] = 0.5 * anmxR[:,Zh]
        anmyR[:,Zh] = 0.5 * anmyR[:,Zh]
        anmzR[:,Zh] = 0.5 * anmzR[:,Zh]

    if 1==1:
        NL = NLag
        #TT,PP, faces, regions = fibonacci_sphere(NL,new=False)
        Nlx = round(np.sqrt(NL))
        NL = Nlx*round(Nlx/2)
        pc = np.linspace(0,np.pi*2,Nlx)
        tc = np.linspace(0,np.pi,round(Nlx/2))
        TT,PP = np.meshgrid(tc,pc)
        dP = 2*np.pi / Nlx
        dT = np.pi / round(Nlx/2)
        wi = dT*dP*np.sin(TT)

        # fT = TT[TT<=0]
        # fP = PP[PP<=0]
        # print(fT)
        # print(fP)
        Naa = 0



        Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, 1 ,1 )

        # anm = anmxR[it,:] + 1i * anmxI[it,:]
        # print(np.shape(anm))
        # valxi =  np.sum( * Ysh,1)
        # valyi =  np.sum(complex( anmyR[it,:], anmyI[it,:] ) * Ysh,1)
        # valzi =  np.sum(complex( anmzR[it,:], anmzI[it,:] ) * Ysh,1)
        HH = []
        AA = []
        SS = []
        for it in Iarray:
            valx = np.zeros_like(TT)
            valy = np.zeros_like(TT)
            valz = np.zeros_like(TT)

            valxT = np.zeros_like(TT)
            valyT = np.zeros_like(TT)
            valzT = np.zeros_like(TT)

            valxP = np.zeros_like(TT)
            valyP = np.zeros_like(TT)
            valzP = np.zeros_like(TT)
            print(f'times step {it}')
            for n in range(0,NF+1):
                Zh = round(n*(n+1)/2)

                for m in range(0,n+1):
                   zz = round( n*(n+1)/2 + m )

        #               Ysh1 = sph_harm(m, n, TT, PP)
                   #Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, n ,m)
                   valxi =  complex( anmxR[it,zz], anmxI[it,zz] ) * Ysh[zz,:,:]
                   valyi =  complex( anmyR[it,zz], anmyI[it,zz] ) * Ysh[zz,:,:]
                   valzi =  complex( anmzR[it,zz], anmzI[it,zz] ) * Ysh[zz,:,:]

                   valx += np.real( valxi )
                   valy += np.real( valyi )
                   valz += np.real( valzi )

                   valxiT =  complex( anmxR[it,zz], anmxI[it,zz] ) *Ydt[zz,:,:]#* Ysh
                   valyiT =  complex( anmyR[it,zz], anmyI[it,zz] ) *Ydt[zz,:,:]#* Ysh
                   valziT =  complex( anmzR[it,zz], anmzI[it,zz] ) *Ydt[zz,:,:]#* Ysh

                   valxiP =  complex( anmxR[it,zz], anmxI[it,zz] ) *Ydp[zz,:,:]#* Ysh
                   valyiP =  complex( anmyR[it,zz], anmyI[it,zz] ) *Ydp[zz,:,:]#* Ysh
                   valziP =  complex( anmzR[it,zz], anmzI[it,zz] ) *Ydp[zz,:,:]#* Ysh

                   valxT += np.real( valxiT )
                   valyT += np.real( valyiT )
                   valzT += np.real( valziT )


                   valxP += np.real( valxiP )
                   valyP += np.real( valyiP )
                   valzP += np.real( valziP )

            valx = valx*2
            valy = valy*2
            valz = valz*2

            valxT = valxT*2
            valyT = valyT*2
            valzT = valzT*2

            valxP = valxP*2
            valyP = valyP*2
            valzP = valzP*2

            g_kov = np.zeros([2,2,np.shape(TT)[0],np.shape(TT)[1] ])

            g_kov[0,0,:,:] = valxT**2 + valyT**2 + valzT**2
            g_kov[0,1,:,:] = valxT*valxP + valyT*valyP + valzT*valzP
            g_kov[1,1,:,:] = valxP**2 + valyP**2 + valzP**2
            g_kov[1,0,:,:] = g_kov[0,1,:,:]

            detg = g_kov[0,0,:,:]*g_kov[1,1,:,:] - g_kov[0,1,:,:]*g_kov[1,0,:,:]
            S = np.sum(np.sqrt(detg)*wi)


            R0 = np.sqrt( S / (4*np.pi) )
            #SK = 0.5**2 * 4*np.pi
            #R0 = 0.5

            GR_kon = np.zeros([2,2,np.shape(TT)[0],np.shape(TT)[1] ])
            GR_kon[0,0,:,:] = 1/R0**2
            GR_kon[1,1,:,:] = 1/(R0**2 * np.sin(TT)**2)
            detgR = 1/(R0**4 * np.sin(TT)**2)

            trC = GR_kon[0,0,:,:]*g_kov[0,0,:,:] + GR_kon[1,1,:,:]*g_kov[1,1,:,:]

            #SK = 0.5**2 * 4*np.pi
            R0 = 0.5
            GR_kon0 = np.zeros([2,2,np.shape(TT)[0],np.shape(TT)[1] ])
            GR_kon0[0,0,:,:] = 1/R0**2
            GR_kon0[1,1,:,:] = 1/(R0**2 * np.sin(TT)**2)
            detgR0 = 1/(R0**4 * np.sin(TT)**2)


            dA= np.sqrt(detg * detgR0) * R0**2 * 4*np.pi / Ntat
            detC = detg*detgR
            #print(detC)

            A = (np.sqrt(detg)*wi).T.flatten()

            gamma = max_shear(trC.T.flatten(),detC.T.flatten())

            dhydro = np.sqrt(dA)
            dhydro = dhydro.T.flatten()

            #HH.append(dhydro)
            AA.append(A)
            SS.append(gamma)
            HH.append(dhydro)


        A = np.array(AA).flatten()
        gamma = np.array(SS).flatten()
        dhydro = np.array(HH).flatten() / s1.dx
        gamma = np.arctan(gamma) * 180/np.pi

        valid_hydro = ~np.isnan(dhydro) & ~np.isnan(A)
        valid_shear = ~np.isnan(gamma) & ~np.isnan(A)

        kde_hydro = gaussian_kde(dhydro[valid_hydro], weights=A[valid_hydro])
        kde_shear = gaussian_kde(gamma[valid_shear], weights=A[valid_shear])

        Nx = 1000
        xhydro = np.linspace(np.nanmin(dhydro), np.nanmax(dhydro), Nx)
        xshear = np.linspace(np.nanmin(gamma), np.nanmax(gamma), Nx)

        pdf_hydro = kde_hydro(xhydro)
        pdf_shear = kde_shear(xshear)
        ax1.plot(xhydro,pdf_hydro,label='$\sqrt{\Delta A} \, / \, \Delta x$',color=clr)
        ax2.plot(xshear,pdf_shear,label='$gamma$',color=clr)


def compute_spherical_harmonics(NF, TT, PP, ni, mi):
    """
    Compute spherical harmonics and derivatives.
    """

    sinT = np.sin(TT)
    cosT = np.cos(TT)

    NZ  = round((NF+1)*(NF+2)/2)

    Pnm = np.zeros([NZ,np.shape(TT)[0],np.shape(TT)[1]])
    PnmdT = np.zeros([NZ,np.shape(TT)[0],np.shape(TT)[1]])

    Pnm[0,:,:] = 1.0               # = Pnm(zz(0,0),:) = P_00
    Pnm[1,:,:] = cosT               # = Pnm(zz(1,0),:) = P_10
    Pnm[2,:,:] = sinT               # = Pnm(zz(1,2),:) = P_11
    Pnm[3,:,:] = 0.5*(3*cosT**2-1)# = Pnm(zz(2,0),:) = P_20
    Pnm[4,:,:] = 3*cosT*sinT        # = Pnm(zz(2,1),:) = P_21
    Pnm[5,:,:] = 3*sinT**2          # = Pnm(zz(2,2),:) = P_22

    # Recursive computation
    for n in range(3, NF + 1):
        for m in range(0, n - 1):
            Pnm[zz(n, m), :, :] = (1.0 / (n - m)) * ((2 * n - 1) * cosT * Pnm[zz(n - 1, m), :, :] - (n + m - 1) * Pnm[zz(n - 2, m), :, :])

        m = n - 1
        Pnm[zz(n, m), :, :] = (2 * n - 1) * cosT * Pnm[zz(n, -1), :, :]

        m = n
        Pnm[zz(n, m), :, :] = (2 * n - 1) * sinT * Pnm[zz(n, -1), :, :]

    # Derivatives of P_nm
    for n in range(1, NF + 1):
        m = 0
        PnmdT[zz(n, m), :, :] = -Pnm[zz(n, 1), :, :]

        for m in range(1, n):
            PnmdT[zz(n, m), :, :] = 0.5 * ((n + m) * (n - m + 1) * Pnm[zz(n, m - 1), :, :] - Pnm[zz(n, m + 1), :, :])

        m = n
        PnmdT[zz(n, m), :, :] = n * Pnm[zz(n, n - 1), :, :]

    #eimP(0:NF,sh%nl)
    # Build spherical harmonics and derivatives
    eimP = np.zeros([NF + 1, np.shape(sinT)[0], np.shape(sinT)[1]], dtype=complex)
    for m in range(0, NF + 1):
        eimP[m, :, :] = np.exp(1j * m * PP )

    n = ni
    m = mi
    Y = np.zeros_like(Pnm, dtype=complex)
    Ydp = np.zeros_like(Pnm, dtype=complex)
    Ydt = np.zeros_like(Pnm, dtype=complex)
    for n in range(0,NF+1):
        for m in range(0,n+1):
            Z = zz(n, m)
            Y[Z,:,:]   = fnm(n,m) * eimP[m, :, :] * Pnm[Z, :, :]
            Ydp[Z,:,:] = 1j * m * Y[Z,:,:]
            Ydt[Z,:,:] = fnm(n,m) * eimP[m, :, :] * PnmdT[Z, :, :]

    return Y, Ydp, Ydt


def compute_spherical_harmonics_array(NF, TT, PP, ni, mi):
    """
    Compute spherical harmonics and derivatives.
    """

    sinT = np.sin(TT)
    cosT = np.cos(TT)

    NZ  = round((NF+1)*(NF+2)/2)

    Pnm = np.zeros([NZ,np.size(TT)])
    PnmdT = np.zeros([NZ,np.size(TT)])

    Pnm[0,:] = 1.0               # = Pnm(zz(0,0),:) = P_00
    Pnm[1,:] = cosT               # = Pnm(zz(1,0),:) = P_10
    Pnm[2,:] = sinT               # = Pnm(zz(1,2),:) = P_11
    Pnm[3,:] = 0.5*(3*cosT**2-1)# = Pnm(zz(2,0),:) = P_20
    Pnm[4,:] = 3*cosT*sinT        # = Pnm(zz(2,1),:) = P_21
    Pnm[5,:] = 3*sinT**2          # = Pnm(zz(2,2),:) = P_22

    # Recursive computation
    for n in range(3, NF + 1):
        for m in range(0, n - 1):
            Pnm[zz(n, m), :] = (1.0 / (n - m)) * ((2 * n - 1) * cosT * Pnm[zz(n - 1, m), :] - (n + m - 1) * Pnm[zz(n - 2, m), :])

        m = n - 1
        Pnm[zz(n, m), :] = (2 * n - 1) * cosT * Pnm[zz(n, -1), :]

        m = n
        Pnm[zz(n, m), :] = (2 * n - 1) * sinT * Pnm[zz(n, -1), :]

    # Derivatives of P_nm
    for n in range(1, NF + 1):
        m = 0
        PnmdT[zz(n, m), :] = -Pnm[zz(n, 1), :]

        for m in range(1, n):
            PnmdT[zz(n, m), :] = 0.5 * ((n + m) * (n - m + 1) * Pnm[zz(n, m - 1), :] - Pnm[zz(n, m + 1), :])

        m = n
        PnmdT[zz(n, m), :] = n * Pnm[zz(n, n - 1), :]

    #eimP(0:NF,sh%nl)
    # Build spherical harmonics and derivatives
    eimP = np.zeros([NF + 1, np.size(TT)], dtype=complex)
    for m in range(0, NF + 1):
        eimP[m, :] = np.exp(1j * m * PP )

    n = ni
    m = mi
    Y = np.zeros_like(Pnm, dtype=complex)
    Ydp = np.zeros_like(Pnm, dtype=complex)
    Ydt = np.zeros_like(Pnm, dtype=complex)
    for n in range(0,NF+1):
        for m in range(0,n+1):
            Z = zz(n, m)
            Y[Z,:]   = fnm(n,m) * eimP[m, :] * Pnm[Z, :]
            Ydp[Z,:] = 1j * m * Y[Z,:]
            Ydt[Z,:] = fnm(n,m) * eimP[m, :] * PnmdT[Z, :]

    return Y, Ydp, Ydt




def max_shear(trace, determinant):
    # Compute the eigenvalues using the quadratic formula
    delta = np.sqrt(trace**2 - 4 * determinant)
    lambda_max = (trace + delta) / 2
    lambda_min = (trace - delta) / 2

    # Maximum shear strain
    gamma_max = 0.5 * (lambda_max - lambda_min)
    return gamma_max


def zz(n, m):
    """
    Vectorized z-index for triangle-shaped n-m-space.
    """
    zz = round(n*(n+1)/2 + m )
    #print(zz)
    return zz

from math import sqrt, factorial, pi

def fnm(n,m):
    """
    Create and compute the fnm array based on the given parameters.

    Parameters:
    - NZ: Size of the fnm array (must be pre-defined)
    - NF: The maximum degree of the factorial calculations

    Returns:
    - fnm: A NumPy array of calculated values
    """

    fnm = np.sqrt(( (2*n+1)*factorial(n-m) )/( 4*np.pi*factorial(n+m) ))

    return fnm


def fibonacci_sphere_pole_clustered(n, beta=2.0):
    """
    Deterministic, regular point set on unit sphere, with controllable clustering at poles.

    beta = 1   -> standard Fibonacci sphere (near-uniform)
    beta > 1   -> more points near poles
    beta < 1   -> more points near equator

    Returns: (x, y, z) arrays of shape (n,)
    """
    i = np.arange(n, dtype=float)

    # golden angle
    ga = np.pi * (3.0 - np.sqrt(5.0))

    # base "uniform-ish" z in [-1,1]
    z0 = 1.0 - 2.0*(i + 0.5)/n

    # warp z to bias toward poles while keeping symmetry
    z = np.sign(z0) * (np.abs(z0) ** (1.0 / beta))

    # radius at latitude
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))

    # azimuth
    phi = ga * i

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    #phi = np.arctan2(x, y) #+ np.pi
    theta = np.arccos(z / 1)

    return theta, phi
