import pandas as pd
import numpy as np
#if not hasattr(np, 'bool'):
#    np.bool = np.bool_
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from .analyse import analyse_Re
from matplotlib import rc
from . import correlations as corr
import os
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from .plot import colors

from scipy.special import sph_harm
from scipy.spatial import SphericalVoronoi, geometric_slerp
import pickle

from .spherical_harmonics_operations import compute_spherical_harmonics
from .spherical_harmonics_operations import compute_spherical_harmonics_2d
from .spherical_harmonics_operations import compute_spherical_harmonics_fibo
from .spherical_harmonics_operations import zz
from .spherical_harmonics_operations import fnm


def apcdf_bub(ax,sim,nr_bin,a_bin,r,phi,symetric,combi=[0,0]):
    rmax = sim.Lx/2
    rbins = np.linspace(0, rmax , nr_bin)
    abins = np.linspace(0, np.pi, a_bin)
    A, R = np.meshgrid(abins, rbins)
    dr = rmax/nr_bin
    da = np.pi/a_bin
    abin = abins[0:-1]+da/2
    rbin = rbins[0:-1]+dr/2
    Ab, Rb = np.meshgrid(abin, rbin)
    dv = da * Rb**2 * dr * np.sin(Ab) * 2 * np.pi
    CN = 4 * np.pi * rmax**3 / ( 3 * np.size(phi) )
    hist, _, _ = np.histogram2d(phi, r, bins=(abins, rbins))
    h = hist.T/dv*CN

    Lref = 1.0
    #h[h==0] = 1E-2
    # plot
    abinp = abin
    abinp[0] = 0
    abinp[-1] = np.pi
    Abp, Rbp = np.meshgrid(abinp, rbin)
    #R = R / Lref
    Rbp = Rbp / Lref
    h[h<1E-2] = 1E-2
    cmap = plt.cm.RdBu_r  # You can choose any colormap you prefer
    levels = np.logspace(-2,2,200,base=10)
    norm = LogNorm(vmin=1/2,vmax=2)
    pc1 = ax.contourf(Abp, Rbp, h,levels=levels, cmap=cmap,norm=norm)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylabel(f"$ r \: / \: d $")
    ax.set_xticks([])
    ax.set_thetamin(0)
    ax.grid(linestyle='-',color='grey',linewidth = 0.25)
    norm = LogNorm()

    if symetric and False:
        ax.set_thetamax(90)
    else:
        ax.set_thetamax(180)

    return ax,pc1




def bcont(sim, i, x, value='v', direction='x', Lref = 1,uref=1,legend=False,Fsize=4,ustream=np.zeros(3),streams=False,figformat='svg',colorbar=False,with_bub=True, forVista=False, dpi=100):
    print('Creating Contour Plot of a Slice of the Fluid Field with the bubble')
    print('1/2 Read Fluid Data')
    # Make Fluid object and extract components
    print(i)


    f = sim.f(i)
    if value == 'u':
        u = f.u - ustream[0]
    if value == 'v':
        u = f.v - ustream[1]
    if value == 'w':
        u = f.w - ustream[2]
    if value == 'p':
        u = f.p

    # Make Grid for plotting (dependet on direction argument)
    if direction == 'x':
       xtat = sim.xp[x]
       X  = np.transpose(sim.Yu , (0, 2, 1))
       Y  = np.transpose(sim.Zu , (0, 2, 1))
       ui = np.transpose(u, (0, 2, 1))
       if streams:
           us = np.transpose(f.w - ustream[2], (0, 2, 1))
           vs = np.transpose(f.v - ustream[1], (0, 2, 1))

       xlim = max(sim.yp)
       ylim = max(sim.zp)
       xlabel = "$ y \: / \: d_1 $"
       ylabel = "$ z \: / \: d_1 $"

    if direction == 'y':
       xtat = sim.yp[x]
       X  = np.transpose(sim.Xv , (1, 2, 0))
       Y  = np.transpose(sim.Zv , (1, 2, 0))
       ui = np.transpose(u, (1, 2, 0))
       if streams:
           us = np.transpose(f.u - ustream[0], (1, 2, 0))
           vs = np.transpose(f.w - ustream[2], (1, 2, 0))

       xlim = max(sim.xp)
       ylim = max(sim.zp)
       xlabel = "$ x  \: / \: d_1 $"
       ylabel = "$ z  \: / \: d_1 $"

    if direction == 'z':
       xtat = sim.zp[x]
       X = sim.Xw
       Y = sim.Yw
       ui = u
       if streams:
           us = f.u - ustream[0]
           vs = f.v - ustream[1]

       xlim = max(sim.xp)
       ylim = max(sim.yp)
       xlabel = "$ x \: / \: d_1 $"
       ylabel = "$ y \: / \: d_1 $"

    X = X / Lref
    Y = Y / Lref
    xlim = xlim / Lref
    ylim = ylim / Lref
    ui = ui / uref
    ui = ui-np.mean(ui)

    figR = ylim / xlim
    fgsx = figR * Fsize
    fgsy = 1* Fsize
    figs = [ fgsy , fgsx ]

    fmean = np.mean(ui)
    fvar = np.var(ui-fmean)
    fmax = fmean + fvar
    fmin = fmean - fvar

    print('2/4 Create Contour-Plot')
    #Contour-Plot
    fig, ax = plt.subplots(figsize=figs)

    if with_bub and not forVista:
        from matplotlib.patches import Polygon
        for j in range(0,sim.bNr):
            print(j)
            b1 = sim.b(j+1)
            zb = b1.zi(sim.tf[i])
            if np.abs(zb-xtat) < 2:
                iB = i#*sim.bnt_plot
                print(iB)
                #bx = b1.fp(iB,sim)
                bx = sph_points(sim,i*1000,NSH=8,NLag=20000,bubi=j+1)
                print(np.shape(bx))
                dx = sim.dx/2
                ti = sim.tf[i-1]
                if direction == 'x':
                    bh = bx[:,0]
                    print(np.shape(bh))
                    indb = np.abs(bh-xtat) < dx
                    print(np.sum(indb))
                    bx = bx[indb,:]
                    bx = bx[:,[1,2]]
                    #ax.scatter(b1.yi(ti),b1.zi(ti),1000,marker='x',alpha=0.5)

                if direction == 'y':
                    bh = bx[:,1]
                    indb = np.abs(bh-xtat) < dx
                    bx = bx[indb,:]
                    bx = bx[:,[0,2]]
                    #ax.scatter(b1.xi(ti),b1.zi(ti),1000,marker='x',alpha=0.5)

                if direction == 'z':
                    bh = bx[:,2]
                    indb = np.abs(bh-xtat) < dx
                    bx = bx[indb,:]
                    bx = bx[:,[0,1]]
                    #ax.scatter(b1.xi(ti),b1.yi(ti),1000,marker='x',alpha=0.5)
                if np.size(bx)>0:
                    hull = ConvexHull(bx)

                    for simplex in hull.simplices:
                        ax.plot(np.mod(bx[simplex, 0],xlim), np.mod(bx[simplex, 1],ylim), '-',color='k')




    #ax.axis('equal')
    ax.pcolormesh(X[:,:,x], Y[:,:,x], ui[:,:,x],cmap='RdBu_r',vmin=-0.5,vmax=0.5)
    if streams:
        xii =  X[:,:,x]
        yii =  Y[:,:,x]
        Ns = 100
#        starting_points = np.column_stack([np.linspace(sim.Lx*1/5,sim.Lx*5/6,Ns), np.ones(Ns)*sim.Lz*3/4])
        ax.streamplot(xii,yii, us[:,:,x],vs[:,:,x], color='black', linewidth=0.2,density=11,broken_streamlines=True)#,start_points=starting_points)

    print('4/4 Save Figure')
    ax.set_xlim(0,xlim)
    ax.set_ylim(0,ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar and not forVista:
        figc, axc = plt.subplots(figsize=(1,3))
        flp = axc.pcolormesh(X[:,:,x], Y[:,:,x], ui[:,:,x],cmap='RdBu_r',vmax=fmax,vmin=fmin)
        figc.colorbar(flp, shrink=0.6, label= '$' + value + '\: / \:'+value+'_{ref} $')
        figc.savefig('figures_pyprime/contour_' + format(i, "d") + '_' + direction +'_' + format(xtat, ".2f") + '_CLRBAR.' + figformat, format=figformat)

    figpath = sim.resultpath + f'/fluid_fig/fluid_bub_{i}_{x}_{direction}_{value}.{figformat}'
    os.makedirs(sim.resultpath + '/fluid_fig',exist_ok=True)
    if forVista:
        ax.axis('off')
        fig.savefig(figpath,format=figformat,dpi=dpi,bbox_inches='tight', pad_inches=0)
        return figs*dpi, figpath
    else:
        fig.savefig(figpath,format=figformat,dpi=dpi)



def bcont_ax(ax,sim, i, x, value='v', direction='x', Lref = 1,uref=1,legend=False,streams=False,colorbar=False,with_bub=True,ustream=np.zeros(3)):
    print('Creating Contour Plot of a Slice of the Fluid Field with the bubble')
    print('1/2 Read Fluid Data')
    # Make Fluid object and extract components
    print(i)
    f = sim.f(i)
    if value == 'u':
        u = f.u - ustream[0]
    if value == 'v':
        u = f.v - ustream[1]
    if value == 'w':
        u = f.w - ustream[2]
    if value == 'p':
        u = f.p

    # Make Grid for plotting (dependet on direction argument)
    if direction == 'x':
       xtat = sim.xp[x]
       X  = np.transpose(sim.Yu , (0, 2, 1))
       Y  = np.transpose(sim.Zu , (0, 2, 1))
       ui = np.transpose(u, (0, 2, 1))
       if streams:
           us = np.transpose(f.w - ustream[2], (0, 2, 1))
           vs = np.transpose(f.v - ustream[1], (0, 2, 1))

       xlim = max(sim.yp)
       ylim = max(sim.zp)
       xlabel = "$ y \: / \: d_1 $"
       ylabel = "$ z \: / \: d_1 $"

    if direction == 'y':
       xtat = sim.yp[x]
       X  = np.transpose(sim.Xv , (1, 2, 0))
       Y  = np.transpose(sim.Zv , (1, 2, 0))
       ui = np.transpose(u, (1, 2, 0))
       if streams:
           us = np.transpose(f.u - ustream[0], (1, 2, 0))
           vs = np.transpose(f.w - ustream[2], (1, 2, 0))

       xlim = max(sim.xp)
       ylim = max(sim.zp)
       xlabel = "$ x  \: / \: d_1 $"
       ylabel = "$ z  \: / \: d_1 $"

    if direction == 'z':
       xtat = sim.zp[x]
       X = sim.Xw
       Y = sim.Yw
       ui = u
       if streams:
           us = f.u - ustream[0]
           vs = f.v - ustream[1]

       xlim = max(sim.xp)
       ylim = max(sim.yp)
       xlabel = "$ x \: / \: d_1 $"
       ylabel = "$ y \: / \: d_1 $"

    X = X / Lref
    Y = Y / Lref
    xlim = xlim / Lref
    ylim = ylim / Lref
    ui = ui / uref
    ui = ui-np.mean(ui)

    fmean = 0
    fvar = np.var(ui-fmean)
    fmax = fmean + fvar
    fmin = fmean - fvar
    fmax = np.max(ui[:,:,x])

    print('2/4 Create Contour-Plot')
    #Contour-Plot
    #ax.axis('equal')
    ax.pcolormesh(X[:,:,x], Y[:,:,x], ui[:,:,x],cmap='RdBu_r',vmin=-uref*2,vmax=uref*2)
    if streams:
        Ns = 75
        starting_points = np.column_stack([np.linspace(sim.Lx*0.01,sim.Lx*0.99,Ns), np.ones(Ns)*sim.Lz*7/8])

        xii =  X[:,:,x]
        yii =  Y[:,:,x]
        ax.streamplot(xii.T,yii.T, vs[:,:,x].T,us[:,:,x].T, color='black', linewidth=0.4,density=3,broken_streamlines=False,start_points=starting_points)

    print('4/4 Save Figure')
    ax.set_xlim(0,xlim)
    ax.set_ylim(0,ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    #fig.savefig(sim.resultpath+'/Contour_' + format(i, "d") + '_' + direction +'_' + format(xtat, ".2f") + '.svg', format='svg')
    #fig.savefig('figures_pyprime/contour_' + format(i, "d") + '_' + direction +'_' + format(xtat, ".2f") + '.' + figformat, format=figformat)
    if colorbar:
        figc, axc = plt.subplots(figsize=(1,3))
        flp = axc.pcolormesh(X[:,:,x], Y[:,:,x], ui[:,:,x],cmap='RdBu_r',vmax=fmax,vmin=fmin)
        figc.colorbar(flp, shrink=0.6, label= '$' + value + '\: / \:'+value+'_{ref} $')
        figc.savefig('figures_pyprime/contour_' + format(i, "d") + '_' + direction +'_' + format(xtat, ".2f") + '_CLRBAR.' + figformat, format=figformat)
    if with_bub:
        i_fluid = sim.nt[i-1]
        bq, brest = divmod(i_fluid,sim.bnt_plot)
        bq = bq
        if brest!=0: print(f'WARNING: No match in bubble fp and fluid data time step. Shifted by: {brest} timesteps')
        for j in range(0,sim.bNr):
            b1 = sim.b(j+1)
            iB = bq
            print(iB)
            bx = b1.fp(iB,sim)
            print(np.shape(bx))
            dx = sim.dx/2
            ti = sim.tf[i-1]
            if direction == 'x':
                bh = bx[:,0]
                print(np.shape(bh))
                indb = np.abs(bh-xtat) < dx
                print(np.sum(indb))
                bx = bx[indb,:]
                bx = bx[:,[1,2]]
        #        ax.scatter(b1.yi(ti),b1.zi(ti),1000,marker='x')

            if direction == 'y':
                bh = bx[:,1]
                indb = np.abs(bh-xtat) < dx
                bx = bx[indb,:]
                bx = bx[:,[0,2]]
        #        ax.scatter(b1.xi(ti),b1.zi(ti),1000,marker='x')

            if direction == 'z':
                bh = bx[:,2]
                indb = np.abs(bh-xtat) < dx
                bx = bx[indb,:]
                bx = bx[:,[0,1]]
        #        ax.scatter(b1.xi(ti),b1.yi(ti),1000,marker='x')
        if np.size(bx)>0:
            hull = ConvexHull(bx)
            for simplex in hull.simplices:
                ax.plot(np.mod(bx[simplex, 0],xlim), np.mod(bx[simplex, 1],ylim), '.',color='k',markersize=0.1)


import pyvista as pv

def bub3d(s1,I,extra='coll',off_screen=True,path='',TeleyP=0, box=True, add_bcont=False,value='v',x=40,uref=1,sd=0):

    if add_bcont:
        resolution, figpath = bcont(s1, I, x, value=value, direction='x',uref=uref,streams=True,figformat='png', forVista=True, dpi=500)

    plg = pv.Plotter(off_screen=False)

    if box:
        Gbox0 = pv.Cube(center=(s1.Lx/2, s1.Ly/2, s1.Lz/2), x_length=s1.Lx, y_length=s1.Ly, z_length=s1.Lz )
        Gbox = Gbox0.triangulate()
        plg.add_mesh(Gbox.outline(), color="black",line_width=5)

    if add_bcont:
        texture = pv.Texture(figpath)
        normal = [-1, 0, 0]   # Normal vector pointing along the Z-axis
        plane = pv.Plane(center=(s1.Lx*x/s1.nx, s1.Ly/2, s1.Lz/2), direction=normal, i_resolution=int(resolution[0]), j_resolution=int(resolution[1]), i_size=s1.Ly,j_size=s1.Lz)
        plg.add_mesh(plane.rotate_x(90, point=(s1.Lx*x/s1.nx, s1.Ly/2, s1.Lz/2), inplace=False), texture=texture, interpolation='flat')

    for ibub in range(1,s1.bNr+1):
        print(f'- Plotting {ibub:d}/{s1.bNr:d}')
        bclr = colors().plclr[ibub-1]
        b1 = s1.b(ibub)
        if I!=0 and extra!="cnm" and extra!='cnmB':
            points = b1.fp(I,s1)

        #plg.add_points(points,render_points_as_spheres=True,point_size=12,color='grey')
        if extra=='coll':
            dc = b1.fpdc(I,s1)
            #dc = b1.fpdotc(I,s1)/1E3/10
        #    sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=8,NLag=40000,color='white',bubi = ibub,opa=1,curve=True,Nc=40,name=f"b_{ibub:d}",pLz=False)
            n  = b1.n(I,s1)
            dcx = np.column_stack([n[:,0]*dc,n[:,1]*dc,n[:,2]*dc])
            plg.add_arrows(points,dcx,mag=1,cmap='Greens')
            print(np.sum(dc))
            if sd!=0:
                #Isd = 11#11
                vis_subdomains(s1,plg,sd,Isd=-1)
                colormap = plt.cm.get_cmap("prism")
                for sdi in range(0,16):
                    print(f"subdomain = {sdi:d}")
                    for p in range(0,3):
                        try:
                            psd = b1.fpsd(I,s1,sd=sdi, period=p)
                            psd[:,2] += 0.001*sdi
                            plg.add_points(psd,render_points_as_spheres=True,point_size=12,color=colormap(sdi/sd))
                        except: pass

        if extra=='coll0':
            dc = b1.fpdc(I,s1)
            #dc = b1.fpdotc(I,s1)/1E3/10
        #    sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=8,NLag=40000,color='white',bubi = ibub,opa=1,curve=True,Nc=40,name=f"b_{ibub:d}",pLz=False)
            n  = b1.n(I,s1)
            dcx = np.column_stack([n[:,0]*dc,n[:,1]*dc,n[:,2]*dc])
            sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=16,NLag=80000,color='white',bubi = ibub,opa=1,curve=True,Nc=40,name=f"b{ibub:d}",pLz=False)
            plg.add_arrows(points,dcx,mag=1,cmap='Greens')

        if extra=='free':
            dcx = b1.fpMui(I,s1)/20
            xf = b1.fpM(I,s1)
            plg.add_arrows(points,dcx,mag=1)
        if extra=='fp':
            xf = b1.fp(I,s1)
            print(xf)
            plg.add_points(xf,render_points_as_spheres=True)

        if extra=='un':
            dc = b1.un(I,s1)*100
            n  = b1.n(I,s1)
            dcx = np.column_stack([n[:,0]*dc,n[:,1]*dc,n[:,2]*dc])
            plg.add_arrows(points,dcx,mag=1,cmap='Greens')

        if extra=='sk':
            dc = b1.fpsk(I,s1) / 10
            n  = b1.n(I,s1)
            dcx = np.column_stack([n[:,0]*dc,n[:,1]*dc,n[:,2]*dc])
            plg.add_arrows(points,dcx,mag=1,cmap='Greens')

        if extra=='cnm':
            sph_add_mesh(s1,I*s1.bnt_plot,plg,NF=8, NLag=4000,bubi = ibub,opa=1,switch_xy=True,Nc=10,pLz=False, offx=0, offy=0,multi=1,lam_max=1.2)

        if extra=='cnmC':
            dc = b1.fpcoll(I,s1)
            fpx = b1.fp(I,s1)

            dc[dc==1000] = 0


            n  = b1.n(I,s1)
            dcx = np.column_stack([n[:,0]*dc,n[:,1]*dc,n[:,2]*dc])
            sph_add_mesh(s1,I*s1.bnt_plot,plg,NF=12,NLag=4000,bubi = ibub,opa=1,switch_xy=True,Nc=10,pLz=False, offx=0, offy=0,multi=1,lam_max=1.2)
            plg.add_arrows(fpx,dcx,mag=1,cmap='Greens')
#fpcoll

            #I*s1.bnt_plot-1
        if extra=='cnm+TC':
            tc = b1.tc(I,s1)*10
            #ph = b1.ph(I,s1)*1E2
            #no = b1.n(I,s1)
            # ph_n = np.zeros_like(no)
            # ph_n[:,0] = no[:,0]*ph
            # ph_n[:,1] = no[:,1]*ph
            # ph_n[:,2] = no[:,2]*ph

            tc1 = np.abs(tc[:,1])
            tc[tc1>2,:] = 0


            fpx = b1.fp(I,s1)
            sph_add_mesh(s1,I*s1.bnt_plot-1,plg,NF=16,NLag=40000,bubi = 1,opa=1,switch_xy=True,Nc=20,pLz=False, offx=0, offy=0,multi=2,lam_max=1.0)
            #sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=8,NLag=8000,color='white',bubi = ibub,opa=1,curve=True,Nc=20,name=f"b{ibub:d}",pLz=False,withJS=True,multi=4)
            plg.add_arrows(fpx, tc, cmap='RdPu')
            #plg.add_arrows(fpx, ph_n, cmap='Blues')


            plg.add_points(fpx, color='k', render_points_as_spheres=True)


        if extra=='cnmB':
            sph_add_mesh(s1,I,plg,with_fp=False,NF=16,NLag=10000,color='white',bubi = ibub,opa=1,curve=True,Nc=40,name=f"b{ibub:d}",pLz=False)

        if extra=='tau':
            dca = b1.tau_c(I,s1) / s1.dx * 2000
            dc = b1.tau(I,s1)
            dci = dc[:,0]

            dcmag = np.sqrt( dca[:,0]**2 + dca[:,1]**2 + dca[:,2]**2 )
            dca[dcmag>0.5,:] = np.nan
            #dci[dci > 1E-1] = np.nan
            #dci[dci < -1E-1] = np.nan

            fpx = b1.tX(I,s1)
            sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=8,NLag=80000,color='white',bubi = ibub,opa=1,curve=True,Nc=40,name=f"b{ibub:d}",pLz=False)
            #plg.add_points(fpx,scalars=dci,cmap='RdBu',render_points_as_spheres=True,point_size=20)
            plg.add_arrows(fpx, dca, cmap='Oranges')
            plg.add_points(points,render_points_as_spheres=True,point_size=12)

        if extra=='corr':
            dca = b1.tau_c(I,s1) / s1.dx * 10
            dc = b1.tau(I,s1)
            dci = dc[:,0]

            dcmag = np.sqrt( dca[:,0]**2 + dca[:,1]**2 + dca[:,2]**2 )
            dca[dcmag>0.5,:] = np.nan
            #dci[dci > 1E-1] = np.nan
            #dci[dci < -1E-1] = np.nan

            fpx = b1.tX(I,s1)
            sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=12,NLag=40000,color='white',bubi = 1,opa=1,curve=True,Nc=40,name="b1",pLz=False)
            #plg.add_points(fpx,scalars=dci,cmap='RdBu',render_points_as_spheres=True,point_size=20)
            plg.add_arrows(fpx, dca, cmap='Oranges')

        if extra=='ui':
            dcx = b1.fpui(I,s1) / 10
            plg.add_arrows(points,dcx,mag=1,cmap='Greens')

        if extra=='para':
            pointsR = b1.fpR(I,s1)
            c = np.linspace(0,1,len(pointsR))
            #plg.add_points(pointsR,render_points_as_spheres=True,scalars = c, cmap = 'Reds', point_size=12,color='red')
            sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=8,NLag=10000,color='white',bubi = 1,opa=0,curve=True,Nc=50,name="",pLz=False)
            sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=8,NLag=10000,color='white',bubi = 1,opa=0,curve=True,Nc=50,name="",pLz=False,cnm2=True)
            sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=8,NLag=10000,color='white',bubi = 1,opa=0,curve=True,Nc=50,name="",pLz=True)
            sph_add_mesh(s1,I*s1.bnt_plot,plg,with_fp=False,NF=8,NLag=10000,color='white',bubi = 1,opa=0,curve=True,Nc=50,name="",pLz=True,cnm2=True)
            dcx  = b1.fpuT(I,s1)  * 1E2
            dcxR = b1.fpuTR(I,s1) * 1E2
            #plg.add_arrows(points,dcx,mag=1,cmap='Blues')
            #plg.add_arrows(pointsR,dcxR,mag=1,cmap='Reds')

        if extra=='sd':
            if sd!=0:
                #Isd = 11#11
                vis_subdomains(s1,plg,sd,Isd=-1)
                colormap = plt.cm.get_cmap("prism")
                for sdi in range(0,16):
                    print(f"subdomain = {sdi:d}")
                    for p in range(0,3):
                        try:
                            psd = b1.fpsd(I,s1,sd=sdi, period=p)
                            psd[:,2] += 0.001*sdi
                            plg.add_points(psd,render_points_as_spheres=True,point_size=12,color=colormap(sdi/sd))
                        except: pass


    if np.size(TeleyP)>1:
        point_cloudT = pv.PolyData(TeleyP)
        plg.add_mesh(point_cloudT,color='grey',render_points_as_spheres=True, point_size=6)

    if path=='':
        Ppath = s1.resultpath + '/film_fig'
    else:
        Ppath = path
    try:
        os.makedirs(Ppath)
    except FileExistsError:
        pass
    plg.show(screenshot=Ppath+f'/film_{I:04d}.png')


def bubMM(s1,I,nr,off_screen=True,path='',TeleyP=0,box=True):
    import pyvista as pv
    cfl = s1.cfl[I]
    for ibub in range(1,s1.bNr+1):
        b1 = s1.b(ibub)
        #points = b1.fp(I,s1)
        #dMM = b1.dMM(I,j,s1)
        xMM = b1.xMM(I,1,s1)
        rcolor = np.random.rand(len(xMM))
        fig,ax = plt.subplots(3,1,figsize=(3,3))
        figb,axb = plt.subplots(1,1,figsize=(5,2))

        for j in range(2,nr):
            x0M = xMM
            xMM = b1.xMM(I,j,s1)
            xd = xMM-x0M
            xdmag = np.sqrt(xd[:,0]**2 + xd[:,1]**2 + xd[:,2]**2)/ (cfl*s1.dx)
            print(xdmag)
            xdmax = np.max(xdmag)
            xdmin = np.min(xdmag)
            xdmean = np.mean(xdmag)
            ax[0].scatter(j,xdmin)
            ax[1].scatter(j,xdmax)
            ax[2].scatter(j,xdmean)
            axb.violinplot(xdmag,positions=[j+1])



def bub_snapshot(s1,I,off_screen=True,path='',uref=2, value='v',Fsize=4,TeleyP=0,extra='coll'):
    import pyvista as pv

    b1 = s1.b(1)
    ti = s1.tf[I]
    xb = b1.xi(ti)
    ixb = np.argmin(np.abs(s1.xce-xb))
    xf  = s1.xce[ixb]

    plg = pv.Plotter(off_screen=off_screen)
    sph_add_mesh(s1,I*s1.bnt_plot-1,plg,with_fp=False,NF=16,NLag=10000,curve=True,Nc=50)

    resolution, figpath = bcont(s1, I, ixb, value=value, direction='x',uref=uref,streams=True,figformat='png', forVista=True, dpi=700,Fsize=Fsize)

    texture = pv.Texture(figpath)
    normal = [-1, 0, 0]   # Normal vector pointing along the Z-axis
    plane = pv.Plane(center=(xf, s1.Ly/2, s1.Lz/2), direction=normal, i_resolution=int(resolution[0]), j_resolution=int(resolution[1]), i_size=s1.Ly,j_size=s1.Lz)
    pr = plane.rotate_x(90, point=(xf, s1.Ly/2, s1.Lz/2), inplace=False)
    plg.add_mesh(pr, texture=texture, lighting=False)
    plg.add_mesh(pr.translate([0,0, s1.Lz]), texture=texture, lighting=False)

    points = b1.fp(I,s1)
    xm = np.mean(points,0)
    plg.camera.focal_point = xm #Fokus auf Blase
    plg.camera.position =  xm + np.array([-s1.Lx*1.25,0,0])
    plg.add_points(points,render_points_as_spheres=True,point_size=5,color='orange')

    if extra=='coll':
        dc = b1.fpdc(I,s1)
        #dc = b1.fpdotc(I,s1)/1E3
        n  = b1.n(I,s1)
        dcx = np.column_stack([n[:,0]*dc,n[:,1]*dc,n[:,2]*dc])
        plg.add_arrows(points,dcx,mag=1,cmap='Greens')

    if np.size(TeleyP)>1:
        point_cloudT = pv.PolyData(TeleyP)
        plg.add_mesh(point_cloudT,color='black',render_points_as_spheres=True, point_size=6)

    if path=='':
        Ppath = s1.resultpath + '/film_fig'
    else:
        Ppath = path
    try:
        os.makedirs(Ppath)
    except FileExistsError:
        pass
    #plg.view_isometric()
    plg.enable_parallel_projection()
    plg.show(screenshot=Ppath+f'/film_{I:04d}.png')


def sph_add_mesh(s1,it,plg,NF=12,NLag=10000,bubi = 1,opa=1,switch_xy=True,Nc=100,pLz=False, offx=0, offy=0,multi=1,lam_max = 1.5,grid_color='k',cut0=[np.nan,np.nan,np.nan],cutn=[0,0,0],newt=False):
    import pyvista as pv

    if not newt:
        cnm_path = s1.resultpath + f'/bubble/bub_{bubi:03d}_cnm_pnt.bin'
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
    else:
        cnm_path = s1.resultpath + f'/bubble/bub_{bubi:03d}_cnm_pnt.bin'
        with open(cnm_path, 'rb') as fin:
            A = np.fromfile(fin, dtype=np.float64,offset=4)
        NL1 = (NF+1)*(NF+2)/2
        NL = NL1*2
        NL = round(NL) + 2
        Ncof = round ( NL*3 ) + 2
        lA = len(A)
        maxcoef = lA%Ncof
        A = A[0:-maxcoef]
        cnm = np.reshape(A,[-1,Ncof])
        t0 = cnm[:,0]
        cnm0 = cnm[:,2:]
        print(t0)
        cnmx = cnm0[:,0:NL]
        cnmy = cnm0[:,NL:(2*NL)]
        cnmz = cnm0[:,(2*NL):(3*NL)]
        NLh = round(NL/2)



    print(np.shape(cnmx))
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

    NL = NLag
    #TT,PP, faces, regions = fibonacci_sphere(NL,new=False)
    Nlx = round(np.sqrt(NL))
    NL = Nlx**2
    pc = np.linspace(0,np.pi*2,Nlx)
    tc = np.linspace(0,np.pi,Nlx)
    #dpc = np.pi*2/Nlx
    #dtc = np.pi/round(Nlx/2)
    #pc = np.mod(pc+dpc/2,2*np.pi)
    #tc = np.mod(tc+dtc/2,np.pi)
    tc[-1] = np.pi-1E-10
    tc[0] = 1E-10
    TT,PP = np.meshgrid(tc,pc)
    dP = 2*np.pi / Nlx
    dT = np.pi / round(Nlx/2)
    wi = dT*dP*np.sin(TT)
    Naa = 0

    valx = np.zeros_like(TT)
    valy = np.zeros_like(TT)
    valz = np.zeros_like(TT)

    valxT = np.zeros_like(TT)
    valyT = np.zeros_like(TT)
    valzT = np.zeros_like(TT)

    valxP = np.zeros_like(TT)
    valyP = np.zeros_like(TT)
    valzP = np.zeros_like(TT)

    Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, 1 ,1 )

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)

        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m )

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

    b1 = s1.b(bubi)
    R0 = b1.r

    GR_kon = np.zeros([2,2,np.shape(TT)[0],np.shape(TT)[1] ])
    GR_kon[0,0,:,:] = 1/R0**2
    GR_kon[1,1,:,:] = 1/(R0**2 * np.sin(TT)**2)
    detgR = 1/(R0**4 * np.sin(TT)**2)

    trC = GR_kon[0,0,:,:]*g_kov[0,0,:,:] + GR_kon[1,1,:,:]*g_kov[1,1,:,:]
    I2 = detg * detgR
    JS = np.sqrt( I2 )

    I1 = g_kov[0,0,:,:]*GR_kon[0,0,:,:] + g_kov[1,1,:,:]*GR_kon[1,1,:,:]

    lambda_m = np.sqrt( I1/2.0 + np.sqrt(I1**(2.0)/4.0 - I2) )

    beta = s1.dx/(2*R0)*np.sqrt(b1.nl / np.pi)

    print('Max. Ratio:')
    print(np.nanmax(lambda_m/beta))

    xm = np.mean(valx)
    ym = np.mean(valy)
    zm = np.mean(valz)

    valx = (valx-xm)*0.992 + xm  + offx
    valy = (valy-ym)*0.992 + ym  + offy
    valz = (valz-zm)*0.992 + zm

    if pLz: valz += s1.Lz

    if switch_xy:
        mesh = pv.StructuredGrid(valx, valy, valz)
    else:
        mesh = pv.StructuredGrid(valy, valx, valz)

    mesh.point_data['J_S'] = lambda_m.T.flatten() / beta
    upper_lim = lam_max
    lambda_limits = (1/beta,upper_lim)#(upper_lim**(-1), upper_lim)

    bname = f"bubble_{bubi}"
    if not np.isnan(cut0[0]):
        mesh  = mesh.clip(normal=cutn, origin=cut0)

    plg.add_mesh(mesh,scalars='J_S', clim = lambda_limits, cmap = 'YlGnBu',ambient=0.5,diffuse=0.5,specular=0.05,opacity=opa,name=bname,show_edges=False)
    #plg.add_mesh(mesh,color='white',ambient=0.5,diffuse=0.5,specular=0.05,opacity=opa,name=bname,show_edges=False)

    # Add parameters curves
    tc = np.linspace(0,np.pi*2,2*Nc*multi+1)
    pc = np.linspace(0,np.pi,Nc*multi+1)
    #tc = tc[0:-1]
    #pc = pc[0:-1]
    TC,PC = np.meshgrid(tc,pc)
    valx = np.zeros_like(TC)
    valy = np.zeros_like(TC)
    valz = np.zeros_like(TC)

    valx = np.zeros_like(TC)
    valy = np.zeros_like(TC)
    valz = np.zeros_like(TC)

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)
        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m  )

           #Ysh, Ydp, Ydpp, Ydt, Ydtt, Ydtp = compute_spherical_harmonics(NF, TC, PC, n ,m)
           Ysh = sph_harm(m, n, TC, PC)
           valxi =  complex( anmxR[it,zz], anmxI[it,zz] ) * Ysh
           valyi =  complex( anmyR[it,zz], anmyI[it,zz] ) * Ysh
           valzi =  complex( anmzR[it,zz], anmzI[it,zz] ) * Ysh

           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )

    valx = valx*2
    valy = valy*2
    valz = valz*2

    valxT = valxT*2
    valyT = valyT*2
    valzT = valzT*2

    valxP = valxP*2
    valyP = valyP*2
    valzP = valzP*2

    xm = np.mean(valx)
    ym = np.mean(valy)
    zm = np.mean(valz)

    valx = (valx-xm)*0.995 + xm   + offx
    valy = (valy-ym)*0.995 + ym   + offy
    valz = (valz-zm)*0.995 + zm

    if pLz: valz += s1.Lz

    #X = np.column_stack([valx.ravel(),valy.ravel(),valz.ravel()])
    if switch_xy:
        if multi>1:
            line_meshes = []
            #grid = pv.StructuredGrid(valx, valy, valz)
            for j in range(0, Nc*2):
                # take one line along x (fix j,k=0 here for simplicity)
                xline = valx[:, j*multi]
                yline = valy[:, j*multi]
                zline = valz[:, j*multi]

                pts = np.column_stack([xline, yline, zline])
                # build a PolyLine
                line = pv.Spline(pts, n_points=len(pts))
                #tube = line.tube(radius=0.005, n_sides=24, capping=True)
                line_meshes.append(line)

            for j in range(0, Nc):
                # take one line along x (fix j,k=0 here for simplicity)
                xline = valx[j*multi, :]
                yline = valy[j*multi, :]
                zline = valz[j*multi, :]

                pts = np.column_stack([xline, yline, zline])
                # build a PolyLine
                line = pv.Spline(pts, n_points=len(pts))
                #tube = line.tube(radius=0.005, n_sides=24, capping=True)
                line_meshes.append(line)

            grid = pv.merge(line_meshes)
            if not np.isnan(cut0[0]):
                grid  = grid.clip(normal=cutn, origin=cut0)
            plg.add_mesh(grid, color=grid_color, ambient=0.2,line_width=4)
        else:
            grid = pv.StructuredGrid(valx, valy, valz)
    else:
        grid = pv.StructuredGrid(valy, valx, valz)
    if not multi>1:
        plg.add_mesh(grid, show_edges=True,style='wireframe', line_width=5,color='k')  # show_edges=True for gridlines

            #plg.show_grid()  # Add axes grid


def get_cnm(s1,it,NF=12,bubi=1):
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

    return anmxR, anmxI, anmyR, anmyI, anmzR, anmzI


def sph_make_grid(s1,it,NF=12,NLag=10000,bubi = 1,switch_xy=True,Nc=100,name="",pLz=False):
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

    #Nc = 100
    tc = np.linspace(0,np.pi*2,Nc)
    pc = np.linspace(0,np.pi,round(Nc/2))
    TC,PC = np.meshgrid(tc,pc)
    valx = np.zeros_like(TC)
    valy = np.zeros_like(TC)
    valz = np.zeros_like(TC)

    valx = np.zeros_like(TC)
    valy = np.zeros_like(TC)
    valz = np.zeros_like(TC)

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)
        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m  )

           #Ysh, Ydp, Ydpp, Ydt, Ydtt, Ydtp = compute_spherical_harmonics(NF, TC, PC, n ,m)
           Ysh = sph_harm(m, n, TC, PC)
           valxi =  complex( anmxR[it,zz], anmxI[it,zz] ) * Ysh
           valyi =  complex( anmyR[it,zz], anmyI[it,zz] ) * Ysh
           valzi =  complex( anmzR[it,zz], anmzI[it,zz] ) * Ysh

           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )

    valx = valx*2
    valy = valy*2
    valz = valz*2


    xm = np.mean(valx)
    ym = np.mean(valy)
    zm = np.mean(valz)

    valx = (valx-xm) + xm
    valy = (valy-ym) + ym
    valz = (valz-zm) + zm

    if pLz: valz += s1.Lz

    #X = np.column_stack([valx.ravel(),valy.ravel(),valz.ravel()])
    if switch_xy:
        grid = pv.StructuredGrid(valx, valy, valz)
    else:
        grid = pv.StructuredGrid(valy, valx, valz)

    return grid
    #plg.show_grid()  # Add axes grid


def max_shear(trace, determinant):
    # Compute the eigenvalues using the quadratic formula
    delta = np.sqrt(trace**2 - 4 * determinant)
    lambda_max = (trace + delta) / 2
    lambda_min = (trace - delta) / 2

    # Maximum shear strain
    gamma_max = 0.5 * (lambda_max - lambda_min)
    return gamma_max



def vel3d(s1,I,offscreen=True,path='',Reinf=500, barnard=True):
    import pyvista as pv

    b1 = s1.b(1)
    b1 = s1.b(1)
    points = b1.fp(I,s1)
    ym = np.mean(points[:,0])

    qinf = Reinf * s1.vis / s1.d_b
    climQ = [-qinf,qinf]
    #print(climQ)
    V = s1.f(I).v
    dx = s1.dx
    grid = pv.wrap(V)
    grid.spacing = (dx, dx, dx)
    slices = grid.slice_orthogonal(x=250,y =ym,z=250)
    #contours = grid.contour(climQ)
    if barnard: pv.start_xvfb()
    plc = pv.Plotter(off_screen=offscreen)
    plc.add_mesh(slices,clim=climQ,cmap="RdBu_r",ambient=0.5,diffuse=0.5,show_scalar_bar=False,style='points', point_size=9)
    points = points[:,[1,0,2]]
    hull = ConvexHull(points)
    faces = np.column_stack((np.full(len(hull.simplices), 3), hull.simplices)).flatten()
    mesh = pv.PolyData(points, faces)
    plc.add_mesh(mesh, cmap='RdBu', show_scalar_bar=False,opacity=0.5)
    #plc.add_points(points, color='black', point_size=2.5) #colors().plclr[ic]

    #plc.camera.SetParallelProjection(True)
    plc.camera.focal_point = [2.5,ym,2.5]
    plc.camera.position = [2.5,ym-10,2.5]

    if path=='':
        Ppath = s1.resultpath + '/film_vid'
    else:
        Ppath = path
    try:
        os.makedirs(Ppath)
    except FileExistsError:
        pass

    plc.show(screenshot=Ppath+f'/film_{I:04d}.png')


from .analyse import fibonacci_sphere
from .analyse import first_principal_axis

def sph_plot(s1,tb,with_fp=False,plane_nx=5,NSH=8,NLag=10000):
    import pyvista as pv

    cnm_path = s1.resultpath + '/bubble/bub_001_cnm.dat'

    A = []
    with open(cnm_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            values = line.split()
            for value in values:
                A.append( float(value) )
    A = np.array(A)

    NF = NSH
    NL1 = (NF+1)*(NF+2)/2
    NL = NL1*2
    #print(NL)

    NL = round(NL)
    Ncof = round ( NL*3+1 )

    b = np.reshape(A,[-1,Ncof])

    tsh = b[:,0]
    cnm = b[:,1:]

    cnmx = cnm[:,0:NL]
    cnmy = cnm[:,NL:(2*NL)]
    cnmz = cnm[:,(2*NL):(3*NL)]

    anmxR = cnmx[:,::2]
    anmxI = cnmx[:,1::2]

    anmyR = cnmy[:,::2]
    anmyI = cnmy[:,1::2]

    anmzR = cnmz[:,::2]
    anmzI = cnmz[:,1::2]

    Naa = 0
    #tb = 15
    N0 = 100

    NL = NLag
    TT,PP, faces = fibonacci_sphere(NL,new=True)
    GR_kov = np.zeros([2,2])
    GR_kov[1,1] = 1

    fT = TT[TT<=0]
    fP = PP[PP<=0]
    print(fT)
    print(fP)

    for it in range(tb-1,tb+2):
        valx = np.zeros(NL)
        valy = np.zeros(NL)
        valz = np.zeros(NL)

        valxT = np.zeros(NL)
        valyT = np.zeros(NL)
        valzT = np.zeros(NL)

        valxP = np.zeros(NL)
        valyP = np.zeros(NL)
        valzP = np.zeros(NL)

        for n in range(0,NF+1):
            Zh = round(n*(n+1)/2) #zz = round(n*(n+1)/2 + m + 1)
            anmxR[it,Zh] = 0.5 * anmxR[it,Zh]
            anmyR[it,Zh] = 0.5 * anmyR[it,Zh]
            anmzR[it,Zh] = 0.5 * anmzR[it,Zh]

            for m in range(0,n+1):
               Naa += 1
               zz = round( n*(n+1)/2 + m + 1) #zz = round(n*(n+1)/2 + m + 1)

               #Y = sph_harm(m, n, TT, PP)
               Y, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, n ,m)
               valxi =  complex( anmxR[it,zz], anmxI[it,zz] ) *Ysh#* Ysh
               valyi =  complex( anmyR[it,zz], anmyI[it,zz] ) *Ysh#* Ysh
               valzi =  complex( anmzR[it,zz], anmzI[it,zz] ) *Ysh#* Ysh

               valxiT =  complex( anmxR[it,zz], anmxI[it,zz] ) *Ydt#* Ysh
               valyiT =  complex( anmyR[it,zz], anmyI[it,zz] ) *Ydt#* Ysh
               valziT =  complex( anmzR[it,zz], anmzI[it,zz] ) *Ydt#* Ysh

               valxiP =  complex( anmxR[it,zz], anmxI[it,zz] ) *Ydp#* Ysh
               valyiP =  complex( anmyR[it,zz], anmyI[it,zz] ) *Ydp#* Ysh
               valziP =  complex( anmzR[it,zz], anmzI[it,zz] ) *Ydp#* Ysh

               valx += np.real( valxi )
               valy += np.real( valyi )
               valz += np.real( valzi )

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

        g_kov = np.zeros([2,2,NL])

        g_kov[1,1,:] = valxT**2 + valyT**2 + valzT**2
        g_kov[1,2,:] = valxT*valxP + valyT*valyP + valzT*valzP
        g_kov[2,2,:] = valxP**2 + valyP**2 + valzP**2
        g_kov[2,1,:] = g_kov[1,2,:]

        if it == tb:
            X = np.column_stack([valx,valy,valz])
        if it == tb-1:
            un = - np.column_stack([valx,valy,valz])
            t1 = tsh[it]
        if it == tb+1:
            un += np.column_stack([valx,valy,valz])
            t2 = tsh[it]

    detg_kov = g_kov[1,1,:]*g_kov[2,2,:] - g_kov[1,2,:]**2

    pl = pv.Plotter()

    n = np.zeros([len(X),3])
    Xm = np.mean(X,axis=0)
    for k,reg in enumerate(regions):
        points = X[reg,:]
        ni = first_principal_axis(points)
        r0 = X[k,:]-Xm
        ind = np.sign(np.dot(r0,ni))
        n[k,:] = ni * ind
        if k==1 and 1==2:
            pl.add_points(points)
            pl.add_arrows(X[k,:],ni/10)

    import pyvista as pv

    unn = np.zeros(len(X))
    for k, (ui,ni) in enumerate(zip(un,n)):
        unn[k] = np.dot(ui,ni)

    un_m = unn - np.mean(unn)
    umax = np.max(np.abs(unn))
    mesh = pv.PolyData(X, faces)
    mesh.point_data['un'] = un_m
    pl.add_mesh(mesh,scalars='un',ambient=0.5,diffuse=0.4,specular=0.1, cmap='RdBu_r',clim=[-umax,umax])

    pl.show()






def vis_subdomains(s1,plg,sd, Isd=-1):
    # Define the domain lengths
    domain_length_x = s1.Lx
    domain_length_y = s1.Ly
    domain_length_z = s1.Lz

    # Read the file and parse the processor ranges
    file_path = s1.out_prime  # Replace with your file path
    boxes = []

    # Parse the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract the ranges and processor indices
    for i in range(len(lines)):
        if lines[i].startswith("Processor"):
            processor_info = lines[i].strip()
            processor_index = int(processor_info.split("[")[1].split("]")[0])  # Extract the processor index

            # Look for the next line containing the range information
            if i + 1 < len(lines) and "X range of indices" in lines[i + 1]:
                parts = lines[i + 1].split(", ")
                x_range = list(map(int, parts[0].split(":")[1].split()))
                y_range = list(map(int, parts[1].split(":")[1].split()))
                z_range = list(map(int, parts[2].split(":")[1].split()))
                boxes.append((processor_index, x_range, y_range, z_range))

    # Generate and add boxes to the plotter
    for box in boxes:
        processor_index, x_range, y_range, z_range = box
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        z_min, z_max = z_range[0], z_range[1]

        # Scale the indices to the domain lengths
        x_min_scaled = x_min * domain_length_x / 50
        x_max_scaled = x_max * domain_length_x / 50
        y_min_scaled = y_min * domain_length_y / 50
        y_max_scaled = y_max * domain_length_y / 50
        z_min_scaled = z_min * domain_length_z / 100
        z_max_scaled = z_max * domain_length_z / 100

        # Create a box using PyVista
        box_mesh = pv.Box(bounds=(x_min_scaled, x_max_scaled,
                                  y_min_scaled, y_max_scaled,
                                  z_min_scaled, z_max_scaled))
        colormap = plt.cm.get_cmap("prism")
        if Isd==-1 or Isd==processor_index:
            plg.add_mesh(box_mesh, opacity=0.5, show_edges=True, color=colormap(processor_index/sd))
        else:
            plg.add_mesh(box_mesh, opacity=0.25, show_edges=True, color='grey')


def sph_points(s1,it,NSH=8,NLag=2000,bubi=1):

    cnm_path = s1.resultpath + f'/bubble/bub_{bubi:03d}_cnm_pnt.bin'

    with open(cnm_path, 'rb') as fin:
        A = np.fromfile(fin, dtype=np.float64,offset=4)

    NF = NSH
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

    valx = np.zeros_like(TT)
    valy = np.zeros_like(TT)
    valz = np.zeros_like(TT)

    Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, 1 ,1 )

    # anm = anmxR[it,:] + 1i * anmxI[it,:]
    # print(np.shape(anm))
    # valxi =  np.sum( * Ysh,1)
    # valyi =  np.sum(complex( anmyR[it,:], anmyI[it,:] ) * Ysh,1)
    # valzi =  np.sum(complex( anmzR[it,:], anmzI[it,:] ) * Ysh,1)

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

    valx = valx*2
    valy = valy*2
    valz = valz*2
    X = np.column_stack([valx.flatten(),valy.flatten(),valz.flatten()])
    return X



def max_lambda_beta(s1,it,NF=12,NLag=10000,bubi = 1):
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
    Naa = 0

    valx = np.zeros_like(TT)
    valy = np.zeros_like(TT)
    valz = np.zeros_like(TT)

    valxT = np.zeros_like(TT)
    valyT = np.zeros_like(TT)
    valzT = np.zeros_like(TT)

    valxP = np.zeros_like(TT)
    valyP = np.zeros_like(TT)
    valzP = np.zeros_like(TT)

    Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, 1 ,1 )

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)

        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m )

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

    b1 = s1.b(bubi)
    R0 = b1.r

    GR_kon = np.zeros([2,2,np.shape(TT)[0],np.shape(TT)[1] ])
    GR_kon[0,0,:,:] = 1/R0**2
    GR_kon[1,1,:,:] = 1/(R0**2 * np.sin(TT)**2)
    detgR = 1/(R0**4 * np.sin(TT)**2)

    trC = GR_kon[0,0,:,:]*g_kov[0,0,:,:] + GR_kon[1,1,:,:]*g_kov[1,1,:,:]
    I2 = detg * detgR
    JS = np.sqrt( I2 )

    I1 = g_kov[0,0,:,:]*GR_kon[0,0,:,:] + g_kov[1,1,:,:]*GR_kon[1,1,:,:]

    lambda_m = np.sqrt( I1/2.0 + np.sqrt(I1**(2.0)/4.0 - I2) )
    beta = s1.dx/(2*R0)*np.sqrt(b1.nl / np.pi)
    lambda_max = np.nanmax(lambda_m)

    return lambda_max/beta, beta


from scipy.special import sph_harm


def she_mesh_cnm(s1,cnm,it,NF=12,NLag=10000,switch_xy=True):
    import pyvista as pv

    NZ = round( (NF+1)*(NF+2)/2 )+1
    cnmx = cnm[0:NZ,:]
    cnmy = cnm[NZ:(2*NZ),:]
    cnmz = cnm[(2*NZ):(3*NZ),:]

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
    Naa = 0

    valx = np.zeros_like(TT)
    valy = np.zeros_like(TT)
    valz = np.zeros_like(TT)

    valxT = np.zeros_like(TT)
    valyT = np.zeros_like(TT)
    valzT = np.zeros_like(TT)

    valxP = np.zeros_like(TT)
    valyP = np.zeros_like(TT)
    valzP = np.zeros_like(TT)

    Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, 1 ,1 )

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)

        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m )

           valxi =  cnmx[zz,it] * Ysh[zz,:,:]
           valyi =  cnmy[zz,it] * Ysh[zz,:,:]
           valzi =  cnmz[zz,it] * Ysh[zz,:,:]

           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )

    valx = valx*2
    valy = valy*2
    valz = valz*2

    if switch_xy:
        grid = pv.StructuredGrid(valx, valy, valz)
    else:
        grid = pv.StructuredGrid(valy, valx, valz)


    return grid


def she_mesh_cnm_curvature(s1,cnm,it,NF=12,NLag=10000,switch_xy=True,multi=2,Nc=10):
    import pyvista as pv

    NZ = round( (NF+1)*(NF+2)/2 )+1
    cnmx = cnm[0:NZ,:]
    cnmy = cnm[NZ:(2*NZ),:]
    cnmz = cnm[(2*NZ):(3*NZ),:]

    print(np.shape(cnmx))

    NL = NLag
    #TT,PP, faces, regions = fibonacci_sphere(NL,new=False)
    Nlx = round(np.sqrt(NL))
    NL = Nlx*round(Nlx/2)
    pc = np.linspace(0,np.pi*2,Nlx)#+ (2*np.pi/Nlx/2)
    #pc = np.diff(pc0)[0] + pc0
    tc = np.linspace(0,np.pi,round(Nlx/2))
    TT,PP = np.meshgrid(tc,pc)
    dP = 2*np.pi / Nlx
    dT = np.pi / round(Nlx/2)
    wi = dT*dP*np.sin(TT)
    Naa = 0

    valx = np.zeros_like(TT)
    valy = np.zeros_like(TT)
    valz = np.zeros_like(TT)

    valxT = np.zeros_like(TT)
    valyT = np.zeros_like(TT)
    valzT = np.zeros_like(TT)

    valxP = np.zeros_like(TT)
    valyP = np.zeros_like(TT)
    valzP = np.zeros_like(TT)

    valxTT = np.zeros_like(TT)
    valyTT = np.zeros_like(TT)
    valzTT = np.zeros_like(TT)

    valxPP = np.zeros_like(TT)
    valyPP = np.zeros_like(TT)
    valzPP = np.zeros_like(TT)

    valxTP = np.zeros_like(TT)
    valyTP = np.zeros_like(TT)
    valzTP = np.zeros_like(TT)

    Ysh, Ydp, Ydt, Ydtt, Ydpp, Ydtp = compute_spherical_harmonics_2d(NF, TT, PP, 1 ,1 )

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)

        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m )

           valxi =  cnmx[zz,it] * Ysh[zz,:,:]
           valyi =  cnmy[zz,it] * Ysh[zz,:,:]
           valzi =  cnmz[zz,it] * Ysh[zz,:,:]

           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )

           valxiT =  cnmx[zz,it] *Ydt[zz,:,:]#* Ysh
           valyiT =  cnmy[zz,it] *Ydt[zz,:,:]#* Ysh
           valziT =  cnmz[zz,it] *Ydt[zz,:,:]#* Ysh

           valxiP =  cnmx[zz,it] *Ydp[zz,:,:]#* Ysh
           valyiP =  cnmy[zz,it] *Ydp[zz,:,:]#* Ysh
           valziP =  cnmz[zz,it] *Ydp[zz,:,:]#* Ysh

           valxT += np.real( valxiT )
           valyT += np.real( valyiT )
           valzT += np.real( valziT )

           valxP += np.real( valxiP )
           valyP += np.real( valyiP )
           valzP += np.real( valziP )

           valxiTT =  cnmx[zz,it] *Ydtt[zz,:,:]#* Ysh
           valyiTT =  cnmy[zz,it] *Ydtt[zz,:,:]#* Ysh
           valziTT =  cnmz[zz,it] *Ydtt[zz,:,:]#* Ysh

           valxiPP =  cnmx[zz,it] *Ydpp[zz,:,:]#* Ysh
           valyiPP =  cnmy[zz,it] *Ydpp[zz,:,:]#* Ysh
           valziPP =  cnmz[zz,it] *Ydpp[zz,:,:]#* Ysh

           valxiTP =  cnmx[zz,it] *Ydtp[zz,:,:]#* Ysh
           valyiTP =  cnmy[zz,it] *Ydtp[zz,:,:]#* Ysh
           valziTP =  cnmz[zz,it] *Ydtp[zz,:,:]#* Ysh

           valxTT += np.real( valxiTT )
           valyTT += np.real( valyiTT )
           valzTT += np.real( valziTT )

           valxPP += np.real( valxiPP )
           valyPP += np.real( valyiPP )
           valzPP += np.real( valziPP )

           valxTP += np.real( valxiTP )
           valyTP += np.real( valyiTP )
           valzTP += np.real( valziTP )


    valx = valx*2
    valy = valy*2
    valz = valz*2

    valxT = valxT*2
    valyT = valyT*2
    valzT = valzT*2

    valxP = valxP*2
    valyP = valyP*2
    valzP = valzP*2

    valxTT = valxTT*2
    valyTT = valyTT*2
    valzTT = valzTT*2

    valxPP = valxPP*2
    valyPP = valyPP*2
    valzPP = valzPP*2

    valxTP = valxTP*2
    valyTP = valyTP*2
    valzTP = valzTP*2

    # First fundamental form
    E = valxT**2 + valyT**2 + valzT**2
    F = valxT*valxP + valyT*valyP + valzT*valzP
    G = valxP**2 + valyP**2 + valzP**2

    # Normal vector
    nx = valyT*valzP - valzT*valyP
    ny = valzT*valxP - valxT*valzP
    nz = valxT*valyP - valyT*valxP

    n_norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= n_norm
    ny /= n_norm
    nz /= n_norm


    # Second derivatives (replace with your known or computed arrays)
    # e.g., via np.gradient if you don't have them analytically
    # valxTT, valyTT, valzTT, valxTP, valyTP, valzTP, valxPP, valyPP, valzPP

    # Second fundamental form
    L = valxTT*nx + valyTT*ny + valzTT*nz
    M = valxTP*nx + valyTP*ny + valzTP*nz
    N = valxPP*nx + valyPP*ny + valzPP*nz

    #print(np.shape(L))
    # Mean curvature
    H = (E*N - 2*F*M + G*L) / (2*(E*G - F**2))

    if switch_xy:
        grid = pv.StructuredGrid(valx, valy, valz)
    else:
        grid = pv.StructuredGrid(valy, valx, valz)
    grid.point_data['H'] = -H.T.flatten()

    # Add parameters curves
    tc = np.linspace(0,np.pi*2,2*Nc*multi+1)
    pc = np.linspace(0,np.pi,Nc*multi+1)
    #tc = tc[0:-1]
    #pc = pc[0:-1]
    TC,PC = np.meshgrid(tc,pc)
    valx = np.zeros_like(TC)
    valy = np.zeros_like(TC)
    valz = np.zeros_like(TC)

    valx = np.zeros_like(TC)
    valy = np.zeros_like(TC)
    valz = np.zeros_like(TC)

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)
        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m  )

           #Ysh, Ydp, Ydpp, Ydt, Ydtt, Ydtp = compute_spherical_harmonics(NF, TC, PC, n ,m)
           Ysh = sph_harm(m, n, TC, PC)
           valxi =  cnmx[zz,it] * Ysh
           valyi =  cnmy[zz,it] * Ysh
           valzi =  cnmz[zz,it] * Ysh

           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )

    valx = valx*2
    valy = valy*2
    valz = valz*2

    valxT = valxT*2
    valyT = valyT*2
    valzT = valzT*2

    valxP = valxP*2
    valyP = valyP*2
    valzP = valzP*2

    xm = np.mean(valx)
    ym = np.mean(valy)
    zm = np.mean(valz)

    valx = (valx-xm)*1.001 + xm
    valy = (valy-ym)*1.001 + ym
    valz = (valz-zm)*1.001 + zm

    print(np.max(valx))
    print(np.min(valx))


    #X = np.column_stack([valx.ravel(),valy.ravel(),valz.ravel()])
    line_meshes = []
    #grid = pv.StructuredGrid(valx, valy, valz)
    for j in range(0, Nc*2):
        # take one line along x (fix j,k=0 here for simplicity)
        xline = valx[:, j*multi]
        yline = valy[:, j*multi]
        zline = valz[:, j*multi]

        pts = np.column_stack([xline, yline, zline])
        # build a PolyLine
        line = pv.Spline(pts, n_points=len(pts))
        #tube = line.tube(radius=0.005, n_sides=24, capping=True)
        line_meshes.append(line)

    for j in range(0, Nc):
        # take one line along x (fix j,k=0 here for simplicity)
        xline = valx[j*multi, :]
        yline = valy[j*multi, :]
        zline = valz[j*multi, :]

        pts = np.column_stack([xline, yline, zline])
        # build a PolyLine
        line = pv.Spline(pts, n_points=len(pts))
        #tube = line.tube(radius=0.005, n_sides=24, capping=True)
        line_meshes.append(line)

    gridW = pv.merge(line_meshes)



    return grid, gridW

from .analyse import fibonacci_points

def sph_points_fibo(s1,it,NSH=8,NLag=2000,bubi=1):

    cnm_path = s1.resultpath + f'/bubble/bub_{bubi:03d}_cnm_pnt.bin'

    with open(cnm_path, 'rb') as fin:
        A = np.fromfile(fin, dtype=np.float64,offset=4)

    NF = NSH
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

    NL = NLag
    TT,PP = fibonacci_points(NLag)

    #valx = np.sin(TT)*np.cos(PP)
    #valy = np.sin(TT)*np.sin(PP)
    #valz = np.cos(TT)

    if 1==1:
        valx = np.zeros_like(TT)
        valy = np.zeros_like(TT)
        valz = np.zeros_like(TT)

        Ysh, Ydp, Ydt = compute_spherical_harmonics_fibo(NF, TT, PP )

        # anm = anmxR[it,:] + 1i * anmxI[it,:]
        # print(np.shape(anm))
        # valxi =  np.sum( * Ysh,1)
        # valyi =  np.sum(complex( anmyR[it,:], anmyI[it,:] ) * Ysh,1)
        # valzi =  np.sum(complex( anmzR[it,:], anmzI[it,:] ) * Ysh,1)

        for n in range(0,NF+1):
            Zh = round(n*(n+1)/2)

            for m in range(0,n+1):
               zz = round( n*(n+1)/2 + m )

    #               Ysh1 = sph_harm(m, n, TT, PP)
               #Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, n ,m)
               valxi =  complex( anmxR[it,zz], anmxI[it,zz] ) * Ysh[zz,:]
               valyi =  complex( anmyR[it,zz], anmyI[it,zz] ) * Ysh[zz,:]
               valzi =  complex( anmzR[it,zz], anmzI[it,zz] ) * Ysh[zz,:]

               valx += np.real( valxi )
               valy += np.real( valyi )
               valz += np.real( valzi )

        valx = valx*2
        valy = valy*2
        valz = valz*2
    X = np.column_stack([valx.flatten(),valy.flatten(),valz.flatten()])
    return X

from .analyse import fibonacci_sphere_pole_clustered

def sph_points_z(s1,it,NSH=8,bubi=1,NLag=2000,beta=20):

    cnm_path = s1.resultpath + f'/bubble/bub_{bubi:03d}_cnm_pnt.bin'

    with open(cnm_path, 'rb') as fin:
        A = np.fromfile(fin, dtype=np.float64,offset=4)

    NF = NSH
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


    TT,PP = fibonacci_sphere_pole_clustered(NLag,beta=20)

    #valx = np.sin(TT)*np.cos(PP)
    #valy = np.sin(TT)*np.sin(PP)
    #valz = np.cos(TT)

    if 1==1:
        valx = np.zeros_like(TT)
        valy = np.zeros_like(TT)
        valz = np.zeros_like(TT)

        Ysh, Ydp, Ydt = compute_spherical_harmonics_fibo(NF, TT, PP )

        # anm = anmxR[it,:] + 1i * anmxI[it,:]
        # print(np.shape(anm))
        # valxi =  np.sum( * Ysh,1)
        # valyi =  np.sum(complex( anmyR[it,:], anmyI[it,:] ) * Ysh,1)
        # valzi =  np.sum(complex( anmzR[it,:], anmzI[it,:] ) * Ysh,1)

        for n in range(0,NF+1):
            Zh = round(n*(n+1)/2)

            for m in range(0,n+1):
               zz = round( n*(n+1)/2 + m )

    #               Ysh1 = sph_harm(m, n, TT, PP)
               #Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, n ,m)
               valxi =  complex( anmxR[it,zz], anmxI[it,zz] ) * Ysh[zz,:]
               valyi =  complex( anmyR[it,zz], anmyI[it,zz] ) * Ysh[zz,:]
               valzi =  complex( anmzR[it,zz], anmzI[it,zz] ) * Ysh[zz,:]

               valx += np.real( valxi )
               valy += np.real( valyi )
               valz += np.real( valzi )

        valx = valx*2
        valy = valy*2
        valz = valz*2
    X = np.column_stack([valx.flatten(),valy.flatten(),valz.flatten()])
    return X







def sph_test_mesh(A,plg,NF=12,NLag=10000,bubi = 1,opa=1,switch_xy=True,Nc=100,pLz=False, offx=0, offy=0,multi=1,lam_max = 1.5,grid_color='k',cut0=[np.nan,np.nan,np.nan],cutn=[0,0,0]):
    import pyvista as pv
    if 1==1:
        NL1 = (NF+1)*(NF+2)/2
        NL = NL1*2
        NL = round(NL)
        Ncof = round ( NL*3 )
        lA = len(A)
        maxcoef = lA%Ncof
        cnm = A
        cnmx = cnm[0:NL]
        cnmy = cnm[NL:(2*NL)]
        cnmz = cnm[(2*NL):(3*NL)]
        NLh = round(NL/2)

    print(np.shape(cnmx))
    print(np.shape(cnmy))
    print(np.shape(cnmz))
    anmxR = cnmx[ 0:NLh           ]
    anmxI = cnmx[ NLh:(2*NLh)     ]
    anmyR = cnmy[ 0:NLh           ]
    anmyI = cnmy[ NLh:(2*NLh)     ]
    anmzR = cnmz[ 0:NLh           ]
    anmzI = cnmz[ NLh:(2*NLh)     ]

    if 1==1:
        for n in range(0,NF+1):
            Zh = round(n*(n+1)/2)
            print(Zh)
            anmxR[Zh] = 0.5 * anmxR[Zh]
            anmyR[Zh] = 0.5 * anmyR[Zh]
            anmzR[Zh] = 0.5 * anmzR[Zh]

    NL = NLag
    #TT,PP, faces, regions = fibonacci_sphere(NL,new=False)
    Nlx = round(np.sqrt(NL))
    NL = Nlx**2
    pc = np.linspace(0,np.pi*2,Nlx)
    tc = np.linspace(0,np.pi,Nlx)
    #dpc = np.pi*2/Nlx
    #dtc = np.pi/round(Nlx/2)
    #pc = np.mod(pc+dpc/2,2*np.pi)
    #tc = np.mod(tc+dtc/2,np.pi)
    tc[-1] = np.pi-1E-10
    tc[0] = 1E-10
    TT,PP = np.meshgrid(tc,pc)
    dP = 2*np.pi / Nlx
    dT = np.pi / round(Nlx/2)
    wi = dT*dP*np.sin(TT)
    Naa = 0

    valx = np.zeros_like(TT)
    valy = np.zeros_like(TT)
    valz = np.zeros_like(TT)

    valxT = np.zeros_like(TT)
    valyT = np.zeros_like(TT)
    valzT = np.zeros_like(TT)

    valxP = np.zeros_like(TT)
    valyP = np.zeros_like(TT)
    valzP = np.zeros_like(TT)

    Ysh, Ydp, Ydt = compute_spherical_harmonics(NF, TT, PP, 1 ,1 )

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)

        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m )

           valxi =  complex( anmxR[zz], anmxI[zz] ) * Ysh[zz,:,:]
           valyi =  complex( anmyR[zz], anmyI[zz] ) * Ysh[zz,:,:]
           valzi =  complex( anmzR[zz], anmzI[zz] ) * Ysh[zz,:,:]

           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )

           valxiT =  complex( anmxR[zz], anmxI[zz] ) *Ydt[zz,:,:]#* Ysh
           valyiT =  complex( anmyR[zz], anmyI[zz] ) *Ydt[zz,:,:]#* Ysh
           valziT =  complex( anmzR[zz], anmzI[zz] ) *Ydt[zz,:,:]#* Ysh

           valxiP =  complex( anmxR[zz], anmxI[zz] ) *Ydp[zz,:,:]#* Ysh
           valyiP =  complex( anmyR[zz], anmyI[zz] ) *Ydp[zz,:,:]#* Ysh
           valziP =  complex( anmzR[zz], anmzI[zz] ) *Ydp[zz,:,:]#* Ysh

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


    xm = np.mean(valx)
    ym = np.mean(valy)
    zm = np.mean(valz)

    valx = (valx-xm)*0.992 + xm
    valy = (valy-ym)*0.992 + ym
    valz = (valz-zm)*0.992 + zm

    if pLz: valz += s1.Lz

    if switch_xy:
        mesh = pv.StructuredGrid(valx, valy, valz)
    else:
        mesh = pv.StructuredGrid(valy, valx, valz)



    plg.add_mesh(mesh,ambient=0.5,diffuse=0.5,specular=0.05,opacity=opa,show_edges=False,color='white')
    #plg.add_mesh(mesh,color='white',ambient=0.5,diffuse=0.5,specular=0.05,opacity=opa,name=bname,show_edges=False)

    # Add parameters curves
    tc = np.linspace(0,np.pi*2,2*Nc*multi+1)
    pc = np.linspace(0,np.pi,Nc*multi+1)
    #tc = tc[0:-1]
    #pc = pc[0:-1]
    TC,PC = np.meshgrid(tc,pc)
    valx = np.zeros_like(TC)
    valy = np.zeros_like(TC)
    valz = np.zeros_like(TC)

    valx = np.zeros_like(TC)
    valy = np.zeros_like(TC)
    valz = np.zeros_like(TC)

    for n in range(0,NF+1):
        Zh = round(n*(n+1)/2)
        for m in range(0,n+1):
           zz = round( n*(n+1)/2 + m  )

           #Ysh, Ydp, Ydpp, Ydt, Ydtt, Ydtp = compute_spherical_harmonics(NF, TC, PC, n ,m)
           Ysh = sph_harm(m, n, TC, PC)
           valxi =  complex( anmxR[zz], anmxI[zz] ) * Ysh
           valyi =  complex( anmyR[zz], anmyI[zz] ) * Ysh
           valzi =  complex( anmzR[zz], anmzI[zz] ) * Ysh

           valx += np.real( valxi )
           valy += np.real( valyi )
           valz += np.real( valzi )

    valx = valx*2
    valy = valy*2
    valz = valz*2

    valxT = valxT*2
    valyT = valyT*2
    valzT = valzT*2

    valxP = valxP*2
    valyP = valyP*2
    valzP = valzP*2

    xm = np.mean(valx)
    ym = np.mean(valy)
    zm = np.mean(valz)

    #valx = (valx-xm)*0.995 + xm   + offx
    #valy = (valy-ym)*0.995 + ym   + offy
    #valz = (valz-zm)*0.995 + zm

    if pLz: valz += s1.Lz

    #X = np.column_stack([valx.ravel(),valy.ravel(),valz.ravel()])
    if switch_xy:
        if multi>1:
            line_meshes = []
            #grid = pv.StructuredGrid(valx, valy, valz)
            for j in range(0, Nc*2):
                # take one line along x (fix j,k=0 here for simplicity)
                xline = valx[:, j*multi]
                yline = valy[:, j*multi]
                zline = valz[:, j*multi]

                pts = np.column_stack([xline, yline, zline])
                # build a PolyLine
                line = pv.Spline(pts, n_points=len(pts))
                #tube = line.tube(radius=0.005, n_sides=24, capping=True)
                line_meshes.append(line)

            for j in range(0, Nc):
                # take one line along x (fix j,k=0 here for simplicity)
                xline = valx[j*multi, :]
                yline = valy[j*multi, :]
                zline = valz[j*multi, :]

                pts = np.column_stack([xline, yline, zline])
                # build a PolyLine
                line = pv.Spline(pts, n_points=len(pts))
                #tube = line.tube(radius=0.005, n_sides=24, capping=True)
                line_meshes.append(line)

            grid = pv.merge(line_meshes)
            if not np.isnan(cut0[0]):
                grid  = grid.clip(normal=cutn, origin=cut0)
            plg.add_mesh(grid, color=grid_color, ambient=0.2,line_width=8)
        else:
            grid = pv.StructuredGrid(valx, valy, valz)
    else:
        grid = pv.StructuredGrid(valy, valx, valz)
    if not multi>1:
        plg.add_mesh(grid, show_edges=True,style='wireframe', line_width=5,color='k',name='wire')  # show_edges=True for gridlines



        #plg.add_mesh(mesh,ambient=0.5,diffuse=0.5,specular=0.05,name=bname,show_edges=False,color='w')
        #plg.add_mesh(mesh,color='white',ambient=0.5,diffuse=0.5,specular=0.05,opacity=opa,name=bname,show_edges=False)
