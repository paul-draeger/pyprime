#Sammlung an Funktionen für die 3D-Visualisierung

from .sim import sim
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


def vis_sc(s1, N, u_max, ne_res = 70/2, opa=0.65, show_progress=True,taurus=True,sliceEll=False):
    print('Visualisierung (Slice + Kontur)')
    dx = s1.dx
    # v-Komponente Einlesen + 'ImageData' Erstellen
    V = s1.f(N).v
    grid = pv.wrap(V)
    grid.spacing = (dx, dx, dx)

    if taurus: pv.start_xvfb() #virtual-frame-buffer (bei Taurus notwendig)

    plc1 = pv.Plotter(off_screen=True)
    plc2 = pv.Plotter(off_screen=True)
    plc1.background_color = "white"
    plc2.background_color = "white"

    #Ellipsoide hinzuügen
    pl = (plc1,plc2)
    #pl = slice_ell(s1, 0, s1.Lx, 0, s1.Ly, 0, s1.Lz, N, pl,ne_res,nocut=False)
    if not sliceEll:
        pl = fade_ell(s1, 0, s1.Lx, 0, s1.Ly, 0, s1.Lz, N, pl,ne_res)
        slice_label = ""
    else:
        pl = slice_ell(s1, 0, s1.Lx, 0, s1.Ly, 0, s1.Lz, N, pl,ne_res)
        slice_label = "_sliced"

    plc1 = pl[0]
    plc2 = pl[1]

    Lx = s1.Ly
    Ly = s1.Lx
    Lz = s1.Lz

    ang = 30/180*np.pi
    r = 38 / 8.461861568222 * s1.Lx
    cy = Ly/2 - np.sin(ang)*r
    cz = Lz/2 + np.cos(ang)*r

    plc1.camera.azimuth = 0
    plc1.camera.position = [Lx/2,cy,cz]
    plc1.camera.focal_point = [Lx/2,Ly/2,Lz/2] #Fokus auf Ellipsoid
    plc1.camera.up = [0,0,0]
    plc1.camera.roll = 90

    plc2.camera.azimuth = 0
    plc2.camera.position = [Lx/2,cy,cz]
    plc2.camera.focal_point = [Lx/2,Ly/2,Lz/2]
    plc2.camera.up = [0,0,0]
    plc2.camera.roll = 90

    # #Kameras Positionieren
    # plc1.camera_position = 'xy'
    # plc1.camera.roll = 90
    # plc1.camera.azimuth = 30

    # plc2.camera_position = 'xy'
    # plc2.camera.roll = 90
    # plc2.camera.azimuth = 30

    # Colormap-Limits
    clim = [-u_max, u_max]
    clim2 = [-u_max, u_max]

    #Slices erstellen + Hinzufügen
    slices = grid.slice_orthogonal(x=0.01,y = 100, z=0.01)
    plc2.add_mesh(slices,clim=clim2,cmap="RdBu_r",ambient=0.5,diffuse=0.5,show_scalar_bar=False)

    try:
        os.mkdir(s1.resultpath+'/vis_S')
    except FileExistsError:
        print('Directory already exists')


    print('Rendering - Slice')
    plc2.show(screenshot=s1.resultpath+'/vis_S/S_frame_'+format(N, "4d")+slice_label+'.png',window_size=[3000,4000],auto_close=False)
    print('Done')

    #Isooberflächen erstellen und hinzufügen
    contours = grid.contour((-u_max/2.25,-u_max/2.5))
    if contours.n_points > 0:
        plc2.add_mesh(contours, clim=clim,cmap="RdBu_r",opacity=opa,ambient=0.5,diffuse=0.5,specular=0.5,show_scalar_bar=False)

    try:
        os.mkdir(s1.resultpath+'/vis_C2')
    except FileExistsError:
        print('Directory already exists')

    #Optional: plc1.enable_depth_peeling()
    print('Rendering - Kontur')
    plc2.show(screenshot=s1.resultpath+'/vis_C2/C_frame_'+format(N, "4d")+slice_label+'.png',window_size=[3000,4000],auto_close=True)
    #if taurus: pv.start_xvfb() #virtual-frame-buffer (bei Taurus notwendig)
    print('C2-done')

    contours = grid.contour((u_max/2,u_max/2))
    plc1.add_mesh(slices,clim=clim,cmap="RdBu_r",lighting=True,ambient=0.5,diffuse=0.5,show_scalar_bar=False)
    if contours.n_points > 0:
        plc1.add_mesh(contours, clim=clim2,cmap="RdBu_r",ambient=0.5,diffuse=0.5,opacity=opa,specular=0.5,show_scalar_bar=False)

    try:
        os.mkdir(s1.resultpath+'/vis_C1')
    except FileExistsError:
        print('Directory already exists')

    plc1.show(screenshot=s1.resultpath+'/vis_C1/C_frame_'+format(N, "4d")+slice_label+'.png',window_size=[3000,4000],auto_close=True)
    print('C1-done')
    plc1.disable()
    plc1.deep_clean()
    plc1.close()
    plc2.disable()
    plc2.deep_clean()
    plc2.close()


def vis_v(s1,N ,u_max,ne_res = 70/2,taurus=True):
    print('Visualisierung (Volumen)')
    dx = s1.dx
    V = s1.f(N).v
    grid = pv.wrap(V)
    grid.spacing = (dx, dx, dx)

    #if taurus: pv.start_xvfb() #virtual-frame-buffer (bei Taurus notwendig)

    plv = pv.Plotter(off_screen=True)
    plv = slice_ell(s1, 0, s1.Lx, 0, s1.Ly, 0, s1.Lz, N, [plv],ne_res)[0]

    climV = (-u_max*1.2,u_max*1.2)

    plv.add_volume(
        grid,
        cmap="RdBu_r",
        clim=climV,
        opacity=[0.25, 0.1, 0.025, 0, 0.025, 0.1, 0.25],
        show_scalar_bar=False,
        ambient=0.51,
        diffuse=0.5
    )

    plv.camera_position = 'xy'
    plv.camera.roll = 90
    plv.camera.azimuth = 30
    try:
        os.mkdir(s1.resultpath+'/vis_V')
    except FileExistsError:
        print('Directory already exists')
    plv.show(screenshot=s1.resultpath+'/vis_V/V_frame_'+format(N, "4d")+'.png',window_size=[3000,4000])

    print('Done')


def vis_E(s1, ti, ne_res = 70/2,taurus=True):
    print('Visualisierung (Ellipsoid)')

    if taurus: pv.start_xvfb() #virtual-frame-buffer (bei Taurus notwendig)

    ple = pv.Plotter(off_screen=True)
    ple.background_color = "white"

    #Ellipsoide hinzuügen
    pl = fade_ell(s1, 0, s1.Lx, 0, s1.Ly, 0, s1.Lz, 0, [ple],ne_res,ti=ti)
    ple = pl[0]

    Lx = s1.Ly
    Ly = s1.Lx
    Lz = s1.Lz

    ang = 30/180*np.pi
    r = 38 / 8.461861568222 * s1.Lx
    cy = Ly/2 - np.sin(ang)*r
    cz = Lz/2 + np.cos(ang)*r

    ple.camera.azimuth = 0
    ple.camera.position = [Lx/2,cy,cz]
    ple.camera.focal_point = [Lx/2,Ly/2,Lz/2] #Fokus auf Ellipsoid
    ple.camera.up = [0,0,0]
    ple.camera.roll = 90

    try:
        os.mkdir(s1.resultpath+'/vis_E')
    except FileExistsError:
        print('dir exists')


    ple.show(screenshot=s1.resultpath+'/vis_E/E_frame_'+format(ti, ".2f")+'.png',window_size=[3000,4000],auto_close=True)
    ple.disable()
    ple.deep_clean()
    ple.close()




    print('Done')


def vis_arrows(s1,N,u_max,NE=50,every=120,arrs=12,ne_res = 80, lagrange_arrows=0, pressure=False,pre_max = 20,taurus=True):
    print('Erstelle Visualisierung (Pfeile)')
    try:
        os.mkdir(s1.resultpath+'/vis_A')
    except FileExistsError:
        print('Directory already exists')

    # Strömungsfeld
    U = s1.f(N).u
    V = s1.f(N).v
    W = s1.f(N).w
    if pressure:
        P = s1.f(N).p

    # Ellipsoid Durchmesser
    de50 = s1.e(NE).d/2
    dem = 2.2
    every = round(every*de50/dem/3)*3

    U = set_random(U, every)

    #Fluid-Gitter einlesen
    X = s1.Xp
    Y = s1.Yp
    Z = s1.Zp

    dx = s1.dx
    tE = s1.tf[N-1]

    # Ellipsoid Koordinaten (x und y vertauscht)
    E = s1.e(NE)
    xc = E.yi(tE)
    yc = E.xi(tE)
    zc = E.zi(tE)

    # Verschiebe U,V,W,X,Y,Z sodass Ellipsoid möglichst in der Mitte ist
    U,xv = mat_mid(U,dx,yc,2)
    V,_ = mat_mid(V,dx,yc,2)
    W,_ = mat_mid(W,dx,yc,2)
    if pressure: P,_ = mat_mid(P,dx,yc,2)

    X,_ = mat_mid(X,dx,yc,2,per=s1.Lx)
    Y,_ = mat_mid(Y,dx,yc,2)
    Z,_ = mat_mid(Z,dx,yc,2)

    U,yv = mat_mid(U,dx,xc,1)
    V,_ = mat_mid(V,dx,xc,1)
    W,_ = mat_mid(W,dx,xc,1)
    if pressure: P,_ = mat_mid(P,dx,xc,1)

    X,_ = mat_mid(X,dx,xc,1)
    Y,_ = mat_mid(Y,dx,xc,1,per=s1.Ly)
    Z,_ = mat_mid(Z,dx,xc,1)

    U,zv = mat_mid(U,dx,zc,3)
    V,_ = mat_mid(V,dx,zc,3)
    W,_ = mat_mid(W,dx,zc,3)
    if pressure: P,_ = mat_mid(P,dx,zc,3)

    X,_ = mat_mid(X,dx,zc,3)
    Y,_ = mat_mid(Y,dx,zc,3)
    Z,_ = mat_mid(Z,dx,zc,3,per=s1.Lz)

    #Pfeile
    v_points = np.column_stack((Y.ravel(), X.ravel(), Z.ravel()))
    vec = np.column_stack((V.ravel(), U.ravel(), W.ravel()))  # Assign your actual vector data

    #Pfeile ignorieren wenn U Komponente ausgeschaltet wurde
    ind_U = (vec[:,1]!=1000)
    v_points = v_points[ind_U]
    vec = vec[ind_U]

    #nur Pfeile in der Ebene des Ellipsoids + Mindestgeschwindigkeit
    ell_ind = (v_points[:,2]<zc+de50/2) & (v_points[:,2]>zc-de50/2)
    v_points = v_points[ell_ind]
    vec = vec[ell_ind]

    #keine Pfeile, die in Ellipsoiden liegen
    vec = check_in_ell(s1,N,v_points,vec)
    #Betrag
    vel_m = np.sqrt(vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)

    #if taurus: pv.start_xvfb() #virtual-frame-buffer (bei Taurus notwendig)
    plg = pv.Plotter(off_screen=True)

    #Kamera
    plg.camera.azimuth = 0
    plg.camera.position = [xc,yc,zc+20]
    plg.camera.focal_point = [xc,yc,zc] #Fokus auf Ellipsoid
    plg.camera.up = [0,0,0]
    plg.camera.roll = 90

    #Ellipsoide hinzufügen
    plg = slice_ell(s1, xv, s1.Lx+xv, yv, s1.Ly+yv, zc-dem, zc+dem, N, [plg],ne_res, nocut=True)[0]

    #ImageData
    grid = pv.wrap(V)
    grid.spacing = (dx, dx, dx)
    grid.origin = (yv,xv,zv) # Modifizeirter Ursprung

    climV = (-u_max,u_max)
    climA = (0,u_max*1.1)
    slices = grid.slice_orthogonal(x=100,y = 100, z=zc)
    plg.add_mesh(slices,cmap="RdBu_r",clim=climV,lighting=True,ambient=0.5,diffuse=0.5,show_scalar_bar=False)

    plg.show(screenshot=s1.resultpath+'/vis_A/no_arrow_'+format(NE, "4d")+'_frame_'+format(N, "4d")+'.png',window_size=[2300,3000],auto_close=False)
    #Pfeile hinzufügen
    plg.add_arrows(v_points, vec,scalars=np.repeat(vel_m,15),cmap="Greys", mag=1/arrs*0.5,clim=climA,show_scalar_bar=False)

    print('Rendering Pfeile')
    plg.show(screenshot=s1.resultpath+'/vis_A/arrow_'+format(NE, "4d")+'_frame_'+format(N, "4d")+'.png',window_size=[2300,3000],auto_close=True)
    print('...done')
    plg.disable()
    plg.deep_clean()
    plg.close()

    if lagrange_arrows:
        print('Arrows - Lagrange')
        #Relative Geschwindigkeitsvektoren
        vec[:,0] = vec[:,0] - E.vi(tE)
        vec[:,1] = vec[:,1] - E.ui(tE)
        vec[:,2] = vec[:,2] - E.wi(tE)
        #Betrag
        vel_m = np.sqrt(vec[:,0]**2 + vec[:,1]**2 + vec[:,2]**2)
        climA = (0,u_max*1.75)

        #if taurus: pv.start_xvfb() #virtual-frame-buffer (bei Taurus notwendig)
        pl_lag = pv.Plotter(off_screen=True)
        #Ellipsoide hinzufügen
        pl_lag = slice_ell(s1, xv, s1.Lx+xv, yv, s1.Ly+yv, zc-dem, zc+dem, N, [pl_lag],ne_res, nocut=True)[0]

        pl_lag.add_mesh(slices,cmap="RdBu_r",clim=climV,lighting=True,ambient=0.5,diffuse=0.5,show_scalar_bar=False)
        #Pfeile hinzufügen
        pl_lag.add_arrows(v_points, vec,scalars=np.repeat(vel_m,15),cmap="Greys",clim=climA, mag=1/arrs*0.3,show_scalar_bar=False)
        #Kamera
        pl_lag.camera.azimuth = 0
        pl_lag.camera.position = [xc,yc,zc+20]
        pl_lag.camera.focal_point = [xc,yc,zc] #Fokus auf Ellipsoid
        pl_lag.camera.up = [0,0,0]
        pl_lag.camera.roll = 90
        print('Rendering Pfeile (Lagrange)')
        pl_lag.show(screenshot=s1.resultpath+'/vis_A/arrow_Lag_'+format(NE, "4d")+'_frame_'+format(N, "4d")+'.png',window_size=[2300,3000],auto_close=True)
        print('...done')
        pl_lag.disable()
        pl_lag.deep_clean()
        pl_lag.close()

    if pressure:
        print('Arrows - Druck')
        #if taurus: pv.start_xvfb() #virtual-frame-buffer (bei Taurus notwendig)
        p_pre = pv.Plotter(off_screen=True)
        p_preM = pv.Plotter(off_screen=True)
        #Ellipsoide hinzufügen
        p_p = slice_ell(s1, xv, s1.Lx+xv, yv, s1.Ly+yv, zc-dem, zc+dem, N, [p_pre, p_preM],ne_res, nocut=True)

        p_pre = p_p[0]
        p_preM = p_p[1]

        #Kamera
        p_pre.camera.azimuth = 0
        p_pre.camera.position = [xc,yc,zc+20]
        p_pre.camera.focal_point = [xc,yc,zc] #Fokus auf Ellipsoid
        p_pre.camera.up = [0,0,0]
        p_pre.camera.roll = 90
        #Kamera
        p_preM.camera.azimuth = 0
        p_preM.camera.position = [xc,yc,zc+20]
        p_preM.camera.focal_point = [xc,yc,zc] #Fokus auf Ellipsoid
        p_preM.camera.up = [0,0,0]
        p_preM.camera.roll = 90

        grid = pv.wrap(P)
        grid.spacing = (dx, dx, dx)
        grid.origin = (yv,xv,zv) # Modifizeirter Ursprung

        P = P-np.mean(P)
        gridp = pv.wrap(P)
        gridp.spacing = (dx, dx, dx)
        gridp.origin = (yv,xv,zv) # Modifizeirter Ursprung

        climP = (-pre_max,pre_max)
        cmap = plt.cm.get_cmap("coolwarm", 19)

        slice_pre = grid.slice_orthogonal(x=100,y = 100, z=zc)
        p_pre.add_mesh(slice_pre,cmap=cmap,clim=climP,lighting=True,ambient=0.5,diffuse=0.5,show_scalar_bar=False)
        print('Rendering Druck')
        p_pre.show(screenshot=s1.resultpath+'/vis_A/arrow_pre_'+format(NE, "4d")+'_frame_'+format(N, "4d")+'.png',window_size=[2300,3000],auto_close=True)
        print(' ...done')
        slice_pre = gridp.slice_orthogonal(x=100,y = 100, z=zc)
        p_preM.add_mesh(slice_pre,cmap=cmap,clim=climP,lighting=True,ambient=0.5,diffuse=0.5,show_scalar_bar=False)
        print('Rendering Druck (gemittelt)')
        p_preM.show(screenshot=s1.resultpath+'/vis_A/arrow_preM_'+format(NE, "4d")+'_frame_'+format(N, "4d")+'.png',window_size=[2300,3000],auto_close=True)
        print(' ...done')
        p_pre.disable()
        p_pre.deep_clean()
        p_pre.close()
        p_preM.disable()
        p_preM.deep_clean()
        p_preM.close()



    print('Visualisierung (Pfeile) - Done')

def simple_add_ell(s1,tE,pl,inds,tags,random=False,Rxyz=0,box=True):
    Gbox0 = pv.Cube(center=(s1.Ly/2, s1.Lx/2, s1.Lz/2), x_length=s1.Ly, y_length=s1.Lx, z_length=s1.Lz )
    Gbox = Gbox0.triangulate()
    if box:
        for pli in pl:
            pli.add_mesh(Gbox.outline(), color="black",line_width=5)
    Ne = s1.eNr
    ne_res = 200
    x_per = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
    y_per = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
    z_per = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
    x_per = x_per * s1.Lx
    y_per = y_per * s1.Ly
    z_per = z_per * s1.Lz
    if inds==0:
        inds = range(0,Ne)
    for i in inds:
        print(f"Add Ellipsoid {i}")
        e = s1.e(i+1)
        #Koordinaten
        if not random:
            xe = e.xi(tE)
            ye = e.yi(tE)
            ze = e.zi(tE)
        else:
            print('Random xyz choosen!')
            xe = Rxyz[i,0]
            ye = Rxyz[i,1]
            ze = Rxyz[i,2]

        if tags:

            poly = pv.PolyData(np.column_stack([ xe+x_per , ye+y_per , ze+z_per]))

            poly["My Labels"] = [f"{i},{j}" for j in range(poly.n_points)]
            for pli in pl:
                pli.add_point_labels(poly, "My Labels", point_size=20, font_size=20)


        spheres = pv.Sphere(center=(ye,xe,ze), radius=e.d/2, theta_resolution=round(ne_res*e.d), phi_resolution=round(ne_res*e.d/2))

        if e.GK==1:
            for pli in pl:
                pli.add_mesh(spheres, color="white", opacity=1)
        if e.GK==2:
            for pli in pl:
                pli.add_mesh(spheres, color="lightgrey", opacity=1)
        if e.GK==3:
            for pli in pl:
                pli.add_mesh(spheres, color="dimgrey", opacity=1)


    return pl


def X_add_ell(pl,Rxyz=0,box=True,Lx=0,Ly=0,Lz=0,dB=1):
    Gbox0 = pv.Cube(center=(Ly/2, Lx/2, Lz/2), x_length=Ly, y_length=Lx, z_length=Lz )
    Gbox = Gbox0.triangulate()
    if box:
        for pli in pl:
            pli.add_mesh(Gbox.outline(), color="black",line_width=5)
    Ne = np.shape(Rxyz)[0]
    print(f'X_add_ell() -> Anzahl Blasen: {Ne:d}')
    ne_res = 200
    x_per = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
    y_per = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
    z_per = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
    x_per = x_per * Lx
    y_per = y_per * Ly
    z_per = z_per * Lz
    for i in range(0,Ne):
        xe = Rxyz[i,0]
        ye = Rxyz[i,1]
        ze = Rxyz[i,2]

        spheres = pv.Sphere(center=(ye,xe,ze), radius=dB/2, theta_resolution=round(ne_res), phi_resolution=round(ne_res/2))
        pli.add_mesh(spheres, color="white", opacity=1)

    return pl



def fade_ell(s1,x_min,x_max,y_min,y_max,z_min,z_max,N,pl,ne_res,ti=0):
    #Fügt Ellipsoide hinzu. Ellipsoide außerhalb der Grenzen werden geschnitten (nocut==0) oder transparent gemacht
    Gbox0 = pv.Cube(center=(y_min+(y_max-y_min)/2, x_min+(x_max-x_min)/2, z_min+(z_max-z_min)/2),x_length=(y_max-y_min),y_length=(x_max-x_min),z_length=(z_max-z_min))
    Gbox = Gbox0.triangulate()
    for pli in pl:
        pli.add_mesh(Gbox.outline(), color="black")
    Ne = s1.eNr
    if ti==0:
        tE = s1.tf[N-1]
    else:
        tE = ti
    #Koordinaten für Kopien
    x_per = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
    y_per = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
    z_per = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
    x_per = x_per * s1.Lx
    y_per = y_per * s1.Ly
    z_per = z_per * s1.Lz

    #Umrandung
    for pli in pl:
        pli.add_mesh(Gbox.outline(), color="black")

    for i in range(0,Ne):
        #print('Ellipsoid hinzufuegen '+format(i,'d') + '/' + format(Ne,'d') )
        e = s1.e(i+1)
        #Koordinaten
        xe = e.xi(tE)
        ye = e.yi(tE)
        ze = e.zi(tE)

        for l in range(0,27):
            xl = xe + x_per[l]
            yl = ye + y_per[l]
            zl = ze + z_per[l]
            #Check, ob Ellipsoid vollständig außerhalb der Grenzen liegt
            if xl<x_max+e.d and xl>x_min-e.d and yl<y_max+e.d and yl>y_min-e.d and zl<z_max+e.d and zl>z_min-e.d:
                b1 = xl>x_max+e.d/2
                b2 = xl<x_min-e.d/2
                b3 = yl>y_max+e.d/2
                b4 = yl<y_min-e.d/2
                b5 = zl>z_max+e.d/2
                b6 = zl<z_min-e.d/2
                ball = b1 or b2 or b3 or b4 or b5 or b6
                fade = 1
                de = e.d
                spheres = pv.Sphere(center=(yl,xl,zl), radius=e.d/2, theta_resolution=round(ne_res*e.d), phi_resolution=round(ne_res*e.d/2))
                if ball: #Transparenz
                    f1 = int(b1)*((x_max+de-xl)/(de/2))
                    f2 = int(b2)*((xl-x_min+de)/(de/2))
                    f3 = int(b3)*((y_max+de-yl)/(de/2))
                    f4 = int(b4)*((yl-y_min+de)/(de/2))
                    f5 = int(b5)*((z_max+de-zl)/(de/2))
                    f6 = int(b6)*((zl-z_min+de)/(de/2))
                    fade = np.max(np.array([f1,f2,f3,f4,f5,f6]))
                    #print(fade)
                if fade>=1 or fade==0: fade=1
                if s1.continous:
                    cmap = plt.get_cmap('binary')
                    Cindex = e.GK/s1.eNr/1.5
                    for pli in pl:
                        pli.add_mesh(spheres, color =cmap(Cindex), opacity=fade)
                else:
                    if e.GK==1:
                        for pli in pl:
                            pli.add_mesh(spheres, color="white", opacity=fade)
                    if e.GK==2:
                        for pli in pl:
                            pli.add_mesh(spheres, color="lightgrey", opacity=fade)
                    if e.GK==3:
                        for pli in pl:
                            pli.add_mesh(spheres, color="dimgrey", opacity=fade)
    return pl



def slice_ell(s1,x_min,x_max,y_min,y_max,z_min,z_max,N,pl,ne_res,nocut=False,t=-1):
    #Fügt Ellipsoide hinzu. Ellipsoide außerhalb der Grenzen werden geschnitten (nocut==0) oder transparent gemacht
    Gbox0 = pv.Cube(center=(y_min+(y_max-y_min)/2, x_min+(x_max-x_min)/2, z_min+(z_max-z_min)/2),x_length=(y_max-y_min),y_length=(x_max-x_min),z_length=(z_max-z_min))
    Gbox = Gbox0.triangulate()
    Ne = s1.eNr
    if t==-1:
        tE = s1.tf[N-1]
    else:
        tE = t
    #Koordinaten für Kopien
    x_per = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
    y_per = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
    z_per = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
    x_per = x_per * s1.Lx
    y_per = y_per * s1.Ly
    z_per = z_per * s1.Lz

    #Umrandung
    for pli in pl:
        pli.add_mesh(Gbox.outline(), color="black")

    for i in range(0,Ne):
        #print('Ellipsoid hinzufuegen '+format(i,'d') + '/' + format(Ne,'d') )
        e = s1.e(i+1)
        #Koordinaten
        xe = e.xi(tE)
        ye = e.yi(tE)
        ze = e.zi(tE)

        for l in range(0,27):
            xl = xe + x_per[l]
            yl = ye + y_per[l]
            zl = ze + z_per[l]
            #Check, ob Ellipsoid vollständig außerhalb der Grenzen liegt
            if xl<x_max+e.d/2 and xl>x_min-e.d/2 and yl<y_max+e.d/2 and yl>y_min-e.d/2 and zl<z_max+e.d/2 and zl>z_min-e.d/2:
                b1 = xl>x_max-e.d/2
                b2 = xl<x_min+e.d/2
                b3 = yl>y_max-e.d/2
                b4 = yl<y_min+e.d/2
                b5 = zl>z_max-e.d/2
                b6 = zl<z_min+e.d/2
                ball = b1+b2+b3+b4+b5+b6
                fade = 1
                de = e.d
                spheres = pv.Sphere(center=(yl,xl,zl), radius=e.d/2, theta_resolution=round(ne_res*e.d), phi_resolution=round(ne_res*e.d))
                if nocut: #Transparenz
                    fade = b5*((z_max+de/2-zl)/de) + b6*((zl-(z_min-de/2))/de)
                    fade = np.abs(fade)
                    if fade>1 or fade==0: fade=1
                if ball<2: #Schneiden
                    if ball==1 and not nocut:
                        try:
                            spheres = spheres.boolean_intersection(Gbox)
                        except Exception as ex:
                            print(f"An error occurred: {ex}")

                if fade>=1 or fade==0: fade=1
                if s1.continous:
                    if spheres.n_points > 0 and spheres.n_cells > 0: #Hinzufügen
                        cmap = plt.get_cmap('binary')
                        Cindex = e.GK/s1.eNr/1.5
                        for pli in pl:
                            pli.add_mesh(spheres, color =cmap(Cindex), opacity=fade)
                else:
                    if spheres.n_points > 0 and spheres.n_cells > 0: #Hinzufügen
                        if e.GK==1:
                            for pli in pl:
                                pli.add_mesh(spheres, color="white", opacity=fade)
                        if e.GK==2:
                            for pli in pl:
                                pli.add_mesh(spheres, color="lightgrey", opacity=fade)
                        if e.GK==3:
                            for pli in pl:
                                pli.add_mesh(spheres, color="dimgrey", opacity=fade)
    return pl




def mat_mid(A,dx,x,di,per=0):
    # Neu-Anordnen von Matrizen in einer Richtung
    nx = A.shape[di-1]
    ix = round((x-dx/2)/dx - nx/2)
    p1 = 0
    p2 = 0
    if ix>0:
        p2 = per
        p1 = 0
    if ix<0:
        p2 = 0
        p1 = -per

    if di==1:
        A1 = A[ix:,:,:] + p1
        A2 = A[0:ix,:,:] + p2
    if di==2:
        A1 = A[:,ix:,:] + p1
        A2 = A[:,0:ix,:] + p2
    if di==3:
        A1 = A[:,:,ix:] + p1
        A2 = A[:,:,0:ix] + p2

    R = np.concatenate((A1,A2),axis=di-1)

    return R, ix*dx

def check_in_ell(s1,N,v_points,vec):
    #Prüfen, ob Pfeile innerhalb eines Ellipsoids liegen und ggf. Geschwindigkeit auf Ellipsoidgeschwindigkeit setzen
    indE = []
    tf = s1.tf[N-1]
    x_per = np.array([0, -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
    y_per = np.array([0, -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
    z_per = np.array([0, -1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
    x_per = x_per * s1.Lx
    y_per = y_per * s1.Ly
    z_per = z_per * s1.Lz

    for j in range(1,s1.eNr+1):
        e = s1.e(j)
        xe = e.xi(tf)
        ye = e.yi(tf)
        ze = e.zi(tf)

        ue = e.vi(tf)
        ve = e.ui(tf)
        we = e.wi(tf)

        re = (e.d/2)**2
        for l in range(0,28):
            x = xe + x_per[l] - v_points[:,1]
            y = ye + y_per[l] - v_points[:,0]
            z = ze + z_per[l] - v_points[:,2]
            r = x**2+y**2+z**2
            if j!=1 or l!=0:
                indE = r < re
                vec[indE,0] = ue
                vec[indE,1] = ve
                vec[indE,2] = we
    return vec


def set_random(A,every):
    #Motivation: Pfeile f\9Fr das ges. Fluidgitter ist unrealistisch
    #Funktion: Random Matrix-Elemente eliminieren (werden sp\8Ater herausgefiltert mit U=1000)
    shape = A.shape
    np.random.seed(42)
    N = round(np.prod(shape)*(1-1/every))

    unique_indices = np.random.choice(np.prod(shape), size=N, replace=False)
    indices_3d = np.unravel_index(unique_indices, shape)
    A[indices_3d] = 1000
    return A


def extract_numeric(filename, out=False):
    if out:
        print('Ermittle Index aus " '+filename+' "')
    # Funktion, die aus den Benennungen 'uvw_0010' den Index extrahiert.
    numeric_part = ''.join(filter(str.isdigit, filename))
    try:
        numeric_value = int(numeric_part)
        if out:
            print('-> '+numeric_part)
        return numeric_value
    except ValueError:
        if out:
            print("kein Index vorhanden")
        return 10000 #Platzhalter


def get_frames(directory,identifier,Ns,Ne,current=False):
    framelist = []
    for frame in os.listdir(directory):
        if identifier in frame:
            framelist = np.append(framelist,frame)

    ind = []
    for file in framelist:
        ind = np.append(ind,extract_numeric(file,out=False))

    combined_data = list(zip(framelist, ind))
    sorted_data = sorted(combined_data, key=lambda x: x[1])
    sorted_files = [item[0] for item in sorted_data]
    sorted_indices = np.sort(ind)

    ind_min = sorted_indices>Ns
    ind_max = sorted_indices<Ne
    ind_ov = np.logical_and(ind_min,ind_max)
    sorted_files = np.array(sorted_files)

    sorted_files = sorted_files[ind_ov]
    sorted_indices = sorted_indices[ind_ov]

    longest_section = []
    current_section = [0]
    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i-1] + 1:
            current_section.append(i)
        else:
            if len(current_section) > len(longest_section) or (current and len(current_section)>10):
                longest_section = current_section
            current_section = [i]

    if len(current_section) > len(longest_section):
        longest_section = current_section
    print(longest_section)
    return sorted_files[longest_section]


def vis_videos(s1,u_max,Nmin=0,Nmax=10000,current=False):
    print('--Videoerstellung--')
    path_r = s1.resultpath
    dirs = ('/vis_C1','/vis_C2','/vis_S','/vis_A','/vis_A','/vis_A','/vis_A','/vis_A')
    videodir = path_r + '/videos'
    idents = ('C_frame_ ','C_frame_ ','S_frame_ ','arrow_   1_frame_','arrow_Lag_   1_frame_ ','no_arrow_   1_frame_ ','arrow_pre_   1_frame_ ','arrow_preM_   1_frame_ ')
    framelist = []
    names = ('Kontur_1','Kontur_2','Slice','Arrow','Arrow_Lag','No_Arrow','Arrow_Pre','Arrow_PreM')

    def update_frame(i, u_max, framelist, directory):
        print('Lese Frame ' + framelist[i])
        image_path = directory+'/'+framelist[i]  # Adjust the filename pattern as needed
        img = plt.imread(image_path)
        ax.clear()
        im = ax.imshow(img,cmap="RdBu_r",  vmin=-u_max, vmax=u_max)

        if i == 1:
            cbar = fig.colorbar(im, ax=ax, shrink = 0.5)
            cbar.set_label('$v \, / \, v_{ref}$', rotation=270, labelpad=15)
            cbar.ax.yaxis.set_ticks_position('right')


        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')  # Turn off axis lines and labels

    try:
        os.makedirs(videodir)
    except OSError as e:
        print(f"An error occurred: {e}")


    for o in range(len(dirs)):
        directory = path_r + dirs[o]
        identifier = idents[o]
        print(format(o,'d') + '/' + format(len(dirs),'d') + ' Erstelle Video ('+identifier+')')
        framelist = get_frames(directory, identifier, Nmin, Nmax, current=current)
        print('Zu bearbeitende Frames:')
        print(framelist)

        dpi = 500
        fig, ax = plt.subplots(figsize=(4000 / dpi, 4000 / dpi))
        num_frames = len(framelist)-1
        interval = 75

        ani = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=interval,fargs=(u_max, framelist, directory))

        output_file = videodir + '/' + names[o] + '.mp4'
        ani.save(output_file, writer='ffmpeg',dpi = dpi)  # Replace 'ffmpeg' with 'pillow' for GIF
