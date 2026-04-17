import numpy as np
from .sim import sim
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import os
import pyvoro
from shapely.geometry import Polygon
from scipy.stats import skew


def make_power(s1,ti):
    #from stl import mesh as meshnp
    import concurrent.futures
    import pyvoro

    def compute_voronoi(t):
        print(f"Zeitpunkt: {t:.2f}")
        points = s1.exyz(t)
        lims = [[0.0, s1.Lx], [0.0, s1.Ly], [0.0, s1.Lz]]
        cells = pyvoro.compute_voronoi(points, lims, 1, periodic=(True, True, True))

        VolC = np.zeros(len(cells))

        vert_ind = []
        for i,cell in enumerate(cells):
            vertices = np.array(cell["vertices"])
            VolC[i] = cell['volume']
            if i==0:
                vert_ges = vertices
            else:
                vert_ges = np.concatenate((vert_ges,vertices),0)
            vert_ind = np.append(vert_ind,i*np.ones(len(cell["vertices"])))

        v_pack = np.column_stack((vert_ges,vert_ind))
        np.save(s1.resultpath+'/voronoi/vor_'+format(t,'.5f')+'.npy',v_pack)
        np.save(s1.resultpath+'/voronoi/vorV_'+format(t,'.5f')+'.npy',VolC)

    try:
        os.mkdir(s1.resultpath + '/voronoi')
    except:
        pass

    print('Berechne Power Tesselation')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(compute_voronoi, ti)


def voro_VI(s1,Ns,iN,ti,delaunay=False,onlygk=0,additive=False,add_N=150,random_type='no',cluster_strength=0.5):
    #print(f"Voronoi GK: {onlygk}")
    if type(ti) == int and ti == -1:
        rand_voro=True
    else:
        rand_voro=False
        Ns = np.size(ti)
    #print(rand_voro)

    N = s1.eNr
    rv = s1.d/2
    ds = s1.d

    dij0 = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            dij0[i,j] = (ds[i]/2+ds[j]/2)**2

    if onlygk!=0:
        indgk = s1.GKi==onlygk
        N = np.sum(indgk)
        ds = s1.d[indgk]
        rv = ds/2
    if additive:
        goldenRatio = (1 + 5**0.5)/2
        i = np.arange(0, add_N)
        theta = 2 *np.pi * i / goldenRatio
        phi = np.arccos(1 - 2*(i+0.5)/add_N)
        xAW, yAW, zAW = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
        fbp = np.column_stack((xAW, yAW, zAW))
        del xAW
        del yAW
        del zAW
        del theta
        del phi

    gke = s1.GKi
    gkl = np.size(s1.du)
    pMix = np.zeros((N,gkl,Ns))
    GKI = np.unique(gke)

    state = np.random.get_state()
    cur_seed = state[1][0]
    print(f"current numpy seed: {cur_seed}")
    np.random.seed(seed=(iN+2)*3)
    state = np.random.get_state()
    cur_seed = state[1][0]
    print(f"-> numpy seed set to: {cur_seed}")


    if N>0:
        rv = s1.d/2
        Lx = s1.Lx
        Ly = s1.Ly
        Lz = s1.Lz
        lims = [[0.0, s1.Lx], [0.0, s1.Ly], [0.0, s1.Lz]]

        x_per = np.array([-1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1,  -1,  0,  1, -1,  0,  1, -1,  0,  1])
        y_per = np.array([-1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1,  -1, -1, -1,  0,  0,  0,  1,  1,  1])
        z_per = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1,   0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  1,  1,  1,  1,  1,  1,  1,  1])
        x_per = x_per * s1.Lx
        y_per = y_per * s1.Ly
        z_per = z_per * s1.Lz
        VolC = np.zeros((N,Ns))
        Adja = np.zeros((N,N,Ns))
        mask = np.zeros(Ns, dtype=bool)

        for q in range(Ns):
            if rand_voro:
                print(f"Nr - {q}")
                per_ind = np.random.permutation(len(ds))

                dR = ds[per_ind]
                gkR = s1.GKi[per_ind]
                idgk = np.argsort(dR)
                dij = dij0[per_ind,:]
                dij = dij[:,per_ind]
                X = np.zeros([N,3])
                n=0
                while n<N:
                    x = np.random.rand()*Lx
                    if random_type=='no':
                        y = np.random.rand()*Ly

                    elif random_type=='single_cluster':
                        y = np.mod(np.random.normal(0.5,cluster_strength),1)*Ly

                    elif random_type=='GK1_cluster':
                        if gkR[n]==1:
                            y = np.mod(np.random.normal(0.5,cluster_strength),1)*Ly
                        else:
                            y = np.random.rand()*Ly
                    elif random_type=='GK2_cluster':
                        if gkR[n]==2:
                            y = np.mod(np.random.normal(0.5,cluster_strength),1)*Ly
                        else:
                            y = np.random.rand()*Ly
                    elif random_type=='GK3_cluster':
                        if gkR[n]==3:
                            y = np.mod(np.random.normal(0.5,cluster_strength),1)*Ly
                        else:
                            y = np.random.rand()*Ly

                    elif random_type=='GK13_cluster':
                        if gkR[n]==1:
                            y = np.mod(np.random.normal(0.25,cluster_strength),1)*Ly
                        elif gkR[n]==3:
                            y = np.mod(np.random.normal(0.75,cluster_strength),1)*Ly
                        else:
                            y = np.random.rand()*Ly

                    elif random_type=='seperation2':
                        if gkR[n]==1:
                            y = np.random.rand()*Ly/2
                        else :
                            y = (np.random.rand()/2 + 1/2)*Ly

                    elif random_type=='seperation3':
                        if gkR[n]==1:
                            y = np.random.rand()*Ly/3
                        if gkR[n]==2:
                            y = (np.random.rand()/3 + 1/3)*Ly
                        if gkR[n]==3:
                            y = (np.random.rand()/3 + 2/3)*Ly

                    z = np.random.rand()*Lz
                    inter = False
                    for i in range(n):
                        dijI = dij[n,i]
                        for k in range(27):
                            d = (X[i,0] - x + x_per[k])**2 + (X[i,1]-y + y_per[k])**2 + (X[i,2]-z + z_per[k])**2
                            if d<dijI:
                                inter=True
                                break
                    if not inter:
                        X[n,0] = x
                        X[n,1] = y
                        X[n,2] = z
                        n+=1
                X = X[idgk,:]
                #print(f"generated random of type {random_type}")

            else:
                print(f"{ti[q]}")
                X = s1.exyz(ti[q])
            try:
                if not delaunay:
                    if not additive:
                        cells = pyvoro.compute_voronoi(X, lims,dispersion=0.1,radii=rv, periodic=(True, True, True))
                        for p,cell in enumerate(cells):
                            VolC[p,q] = cell["volume"]
                            fac = cell['faces']
                            for face in fac:
                                fadj = face['adjacent_cell']
                                Adja[p,fadj,q] = 1

                    else:
                        ftiled = np.tile(fbp.T,N) * np.repeat(rv,add_N)
                        X = np.repeat(X,add_N,0) + ftiled.T
                        cells = pyvoro.compute_voronoi(X, lims,dispersion=0.1, periodic=(True, True, True))
                        pa = -1
                        eface = []
                        efacei = []
                        nface = []
                        oface = []
                        for p,cell in enumerate(cells):
                            if np.mod(p,add_N)==0:
                                pa+=1
                                if pa>0:
                                    eface.append(nface)
                                    efacei.append(oface)
                                    nface = []
                                    oface = []
                            VolC[pa,q] += cell["volume"]
                            fac = cell['faces']
                            cv = cell['vertices']
                            for face in fac:
                                fadj = face['adjacent_cell']
                                if fadj < add_N*pa or fadj > add_N*(pa+1):
                                    of = int(np.floor(fadj/add_N))
                                    adjv = face['vertices']
                                    poly = []
                                    for adv in adjv:
                                        poly.append(cv[int(adv)])
                                    nface.append(poly)
                                    oface.append(of)
                        eface.append(nface)
                        efacei.append(oface)
                        del cells
                        print(len(eface))
                        print(pa)
                        if not gkl>3:
                            for k,fell in enumerate(eface):
                                find = np.array(efacei[k])
                                fgk = gke[find]
                                ar = np.zeros(len(fell))
                                for i,face in enumerate(fell):
                                    polygon_coords_3d = face
                                    normal_vector = np.cross(polygon_coords_3d[1] - polygon_coords_3d[0], polygon_coords_3d[2] - polygon_coords_3d[0])
                                    normal_vector /= np.linalg.norm(normal_vector)
                                    rotation_matrix = np.eye(3) - np.outer(normal_vector, normal_vector)
                                    rotated_polygon_coords = np.dot(rotation_matrix, (polygon_coords_3d - polygon_coords_3d[0]).T).T
                                    projected_polygon_coords = rotated_polygon_coords[:, :2]
                                    polygon = Polygon(projected_polygon_coords)

                                    ar[i] = polygon.area
                                ages = np.sum(ar)
                                for j,gkj in enumerate(GKI):
                                    agk = np.sum(ar[fgk==gkj])
                                    pMix[k,j,q] = agk/ages
                        #Nachbarschaft steht in efaci

                        for igraph,efci in enumerate(efacei):
                            for fcii in efci:
                                Adja[igraph,fcii,q] = 1

                        del eface
                        del efacei
                        del oface
                else:
                    if q==0:
                        VolC = delaunay_lengths(X)
                    else:
                        VolC = np.concatenate(VolC,delaunay_lengths(X))
                mask[q] = True
            except Exception as e:
                print(f"Fehler bei Voronoi-Tesselation: {e}")
                mask[q] = False
    else:
        VolC = []
        mask = []
    return VolC,mask,pMix,Adja


def make_Vvoro(sim,t,Nsub=1,onlygk=0,additive=False,add_N = 200,delaunay=False):
    from concurrent.futures import ProcessPoolExecutor
    print('Generating Simulation Voronoi Tesselation')
    if Nsub>1:
        ti = np.array_split(t,Nsub)

        with ProcessPoolExecutor() as executor:
            results = executor.map(voro_VI, [sim]*Nsub,[0]*Nsub, [0]*Nsub,ti,[delaunay]*Nsub,[onlygk]*Nsub,[additive]*Nsub,[add_N]*Nsub)

        VolC,mask,p,Adja = zip(*results)
        VolC = np.concatenate(VolC,1)
        mask = np.concatenate(mask)
        p = np.concatenate(p,2)
        Adja = np.concatenate(Adja,2)
    else:
        VolC,mask,p,Adja = voro_VI(sim,0,0,t,delaunay=delaunay,onlygk=onlygk,additive=additive)

    tV = t[mask]
    VolC = VolC[:,mask]
    p = p[:,:,mask]
    Adja = Adja[:,:,mask]
    if additive:
        vpath = '/voronoi/AWVolC'
    else:
        vpath = '/voronoi/VolC'
    try:
        os.mkdir(sim.resultpath+'/voronoi')
    except:
        pass
    if onlygk==0:
        np.save(sim.resultpath+vpath+f"voro_{len(t)}.npy",VolC)
        np.save(sim.resultpath+vpath+f"T_{len(t)}.npy",tV)
        np.save(sim.resultpath+vpath+f"p_{len(t)}.npy",p)
        np.save(sim.resultpath+vpath+f"Adja_{len(t)}.npy",Adja)
    else:
        np.save(sim.resultpath+vpath+f"voro_{len(t)}_GK{onlygk}.npy",VolC)
        np.save(sim.resultpath+vpath+f"T_{len(t)}_GK{onlygk}.npy",tV)
        np.save(sim.resultpath+vpath+f"p_{len(t)}_GK{onlygk}.npy",p)


def V_voro(sim,t,Nsub=1,newbinary=0,Nbin=100,additive=False):
    if newbinary:
        make_Vvoro(sim,t,Nsub=Nsub,additive=additive)
    if additive:
        vpath = '/voronoi/AWVolC'
    else:
        vpath = '/voronoi/VolC'

    VolC = np.load(sim.resultpath+vpath+f"voro_{len(t)}.npy")
    tV   = np.load(sim.resultpath+vpath+f"T_{len(t)}.npy")
    p   = np.load(sim.resultpath+vpath+f"p_{len(t)}.npy")

    x,Vmean,Vmean1,Vmean2,Vmean3,Vsigma,Vsigma1,Vsigma2,Vsigma3,hov,h1,h2,h3,kdex,kdeVol,kdeV1,kdeV2,kdeV3 = rand_hist(sim,VolC,Nbins=Nbin)

    return Vmean, Vmean1, Vmean1, Vmean1, Vsigma, Vsigma1, Vsigma2, Vsigma3, x, hov, h1, h2, h3, kdex, kdeVol,kdeV1,kdeV2,kdeV3, VarP, xp, hP


def eg_voro(sim,Nt=2000,Nsub=1,newbinary=0,Nbin=100):
    if newbinary:
        make_Vvoro(sim,t,Nsub=Nsub)
    VolC = np.load(sim.resultpath+f"/voronoi/VolCvoro_{Nt}.npy")
    tV   = np.load(sim.resultpath+f"/voronoi/VolCT_{Nt}.npy")
    d = sim.d
    eg = np.zeros((sim.eNr,Nt))
    for i in range(sim.eNr):
        eg[i,:] = d[i]**3 * np.pi/(6 * VolC[i,:])
    return eg, tV

def rand_hist(s1,VolC, p, Nbins=300, Nkde=2000):

    #print(s1.GKi)
    VBi = s1.d**3 / 6 * np.pi
    vlen = np.shape(VolC)[1]
    VBi = np.column_stack([VBi]*vlen)
    #VolC = VolC / VBi
    print('Mittel')
    print(np.mean(VolC))

    x = np.linspace(0,np.max(VolC),Nbins)

    Vmean = np.mean(VolC)
    Vsigma = np.std(VolC)
    V1 = VolC[s1.GKi==1,:]
    Vmean1 = np.mean(V1)
    Vsigma1 = np.std(V1)
    V2 = VolC[s1.GKi==2,:]
    Vmean2 = np.mean(V2)
    Vsigma2 = np.std(V2)
    V3 = VolC[s1.GKi==3,:]
    Vmean3 = np.mean(V3)
    Vsigma3 = np.std(V3)
    print('hey')

    Vschief1 = 0
    Vschief2 = 0
    Vschief3 = 0
    Vschief = skew(VolC.ravel())
    if len(V1)>0:
        Vschief1 = skew(V1.ravel())
    if len(V2)>0:
        Vschief2 = skew(V2.ravel())
    if len(V3)>0:
        Vschief3 = skew(V3.ravel())


    hov,_ = np.histogram(VolC,bins=x)
    h1,_ = np.histogram(V1.ravel(),bins=x)
    h2,_ = np.histogram(V2.ravel(),bins=x)
    h3,_ = np.histogram(V3.ravel(),bins=x)
    hs = [np.sum(hov),np.sum(h1),np.sum(h2),np.sum(h3)]
    hov = hov / hs[0]
    if hs[1]!=0: h1 = h1 / hs[1]
    if hs[2]!=0: h2 = h2 / hs[2]
    if hs[3]!=0: h3 = h3 / hs[3]
    x = x[0:-1] + np.diff(x)/2

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(VolC.ravel(),bw_method=0.1)
    kdex = np.linspace(0, np.max(VolC),Nkde)
    kdeVol = kde(kdex)

    print('hey2')

    if hs[1]!=0:
        kde1 = gaussian_kde(V1.ravel(),bw_method=0.1)
        kdeV1 = kde1(kdex)
    else:
        kdeV1 = []

    if hs[2]!=0:
        kde2 = gaussian_kde(V2.ravel(),bw_method=0.1)
        kdeV2 = kde2(kdex)
    else:
        kdeV2 = []

    if hs[3]!=0:
        kde3 = gaussian_kde(V3.ravel(),bw_method=0.1)
        kdeV3 = kde3(kdex)
    else:
        kdeV3 = []

    print('hey3')
    if np.sum(p)>0 and len(np.unique(s1.d))>1:
        ps = np.shape(p)
        if ps[0]>1:
            p1 = p[:,0,:]
            xmix = np.linspace(np.min(p),np.max(p))
        if ps[1]>1:
            p2 = p[:,1,:]
        if ps[1]>2:
            p3 = p[:,2,:]

        pvar = np.var(p)

        if ps[0]>1 and not s1.continous:
            kde1P = gaussian_kde(p1.ravel())
            pkde1 = kde1P(xmix)
        else:
            pkde1 = []

        if ps[1]>1 and not s1.continous:
            kde2P = gaussian_kde(p2.ravel())
            pkde2 = kde2P(xmix)
        else:
            pkde2 = []

        if ps[1]>2 and not s1.continous:
            kde3P = gaussian_kde(p3.ravel())
            pkde3 = kde3P(xmix)
        else:
            pkde3 = []
    else:
        xmix = 0
        pkde1 = 0
        pkde2 = 0
        pkde3 = 0
    print('hey4')

    return x,Vmean,Vmean1,Vmean2,Vmean3,Vsigma,Vsigma1,Vsigma2,Vsigma3,hov,h1,h2,h3,kdex,kdeVol,kdeV1,kdeV2,kdeV3, xmix, pkde1, pkde2, pkde3, Vschief, Vschief1, Vschief2, Vschief3


def make_rand_voro(s1,Nv=500,Nsub=50,additive=False,add_N=200,onlygk=0,random_type='no',cluster_strength=0.5):
    from concurrent.futures import ProcessPoolExecutor
    print('Generating Random Voronoi Tesselation')

    Nsi = round(Nv/Nsub)
    iN = np.arange(0,Nsub)*Nsi

    with ProcessPoolExecutor() as executor:
        results = executor.map(voro_VI, [s1]*Nsub,[Nsi]*Nsub, iN,[-1]*Nsub,[False]*Nsub,[onlygk]*Nsub,[additive]*Nsub,[add_N]*Nsub,[random_type]*Nsub,[cluster_strength]*Nsub)

    VolC,mask,p,Adja = zip(*results)

    VolC = np.concatenate(VolC,1)
    mask = np.concatenate(mask)
    p = np.concatenate(p,2)
    Adja = np.concatenate(Adja,2)


    VolC = VolC[:,mask]
    p = p[:,:,mask]
    Adja = Adja[:,:,mask]

    x,Vmean,Vmean1,Vmean2,Vmean3,Vsigma,Vsigma1,Vsigma2,Vsigma3,hov,h1,h2,h3,kdex,kdeVol,kdeV1,kdeV2,kdeV3,xmix, pkde1, pkde2, pkde3, Vschief, Vschief1, Vschief2, Vschief3 = rand_hist(s1,VolC,p)
    psigma = np.var(p,axis=(0,2))


    respath = f"./{s1.dirname}/results"
    if additive:
        if random_type=='no':
            vpath = '/voronoi/R_AWVolC'
        else:
            vpath = f"/voronoi/R_AWVolC_{random_type}_{cluster_strength}"
    else:
        if random_type=='no':
            vpath = '/voronoi/R_VolC'
        else:
            vpath = f"/voronoi/R_VolC_{random_type}_{cluster_strength}"

    if onlygk==0:
        np.save(s1.resultpath+vpath+f"voro_{Nv}.npy",VolC)
        np.save(s1.resultpath+vpath+f"p_{Nv}.npy",p)
        np.save(s1.resultpath+vpath+f"Adja_{Nv}.npy",Adja)
    else:
        np.save(respath+vpath+f"voro_GK{onlygk}.npy",VolC)
        np.save(respath+vpath+f"voro_GK{onlygk}_p.npy",p)


def moving_average(data, window_size):
    # Pad the data at the beginning and end to maintain the same length
    padded_data = np.pad(data, (window_size//2, window_size//2), mode='edge')
    # Compute the moving average
    smoothed_data = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
    return smoothed_data


def rand_voro(sim,Nv=20,newbinary=0,Nsub=10,avg=1,additive=False):
    if newbinary:
        make_rand_voro(sim,Nv=Nv,Nsub=Nsub,additive=additive)
    if additive:
        normpath = f"./pyprime/misc/reference_data/random_voronoi/{sim.label}/rand_{Nv}_AW_"
    else:
        normpath = f"./pyprime/misc/reference_data/random_voronoi/{sim.label}/rand_{Nv}_"
    try:
        x = np.load(normpath+'x.npy')
        hov = np.load(normpath+'hov.npy')
        h1 = np.load(normpath+'h1.npy')
        h2 = np.load(normpath+'h2.npy')
        h3 = np.load(normpath+'h3.npy')
        kdex = np.load(normpath+'kdex.npy')
        kdeV = np.load(normpath+'kdeVol.npy')
        kdeV1 = np.load(normpath+'kdeV1.npy')
        kdeV2 = np.load(normpath+'kdeV2.npy')
        kdeV3 = np.load(normpath+'kdeV3.npy')
        pm = np.load(normpath+'pm.npy')

        hov = moving_average(hov,avg)
        h1 = moving_average(h1,avg)
        h2 = moving_average(h2,avg)
        h3 = moving_average(h3,avg)
        return x, hov, h1, h2, h3, kdex, kdeV, kdeV1, kdeV2, kdeV3, pm
    except Exception as e:
        print(f"Fehler bei Voronoi-Tesselation: {e}")
        print(f"No Data found for Nv={Nv}")
        return 0,0,0,0,0,0,0


def sim_voro(sim,Nv=20,newbinary=0,Nsub=10,avg=1):
    if newbinary:
        make_rand_voro(sim,Nv=Nv,Nsub=Nsub)
    normpath = f"./pyprime/misc/reference_data/random_voronoi/{sim.label}/rand_{Nv}_"
    try:
        x = np.load(normpath+'x.npy')
        hov = np.load(normpath+'hov.npy')
        h1 = np.load(normpath+'h1.npy')
        h2 = np.load(normpath+'h2.npy')
        h3 = np.load(normpath+'h3.npy')

        hov = moving_average(hov,avg)
        h1 = moving_average(h1,avg)
        h2 = moving_average(h2,avg)
        h3 = moving_average(h3,avg)
        return x, hov, h1, h2, h3
    except:
        print(f"No Data found for Nv={Nv}")
        return 0,0,0,0,0


def get_rsig(sim,Nv=20000,additive=False):
    if additive:
        normpath = f"./pyprime/misc/reference_data/random_voronoi/{sim.label}/rand_{Nv}_AW_"
    else:
        normpath = f"./pyprime/misc/reference_data/random_voronoi/{sim.label}/rand_{Nv}_"
    sig = np.load(normpath+'sigma.npy')
    sig1 = np.load(normpath+'sigma1.npy')
    sig2 = np.load(normpath+'sigma2.npy')
    sig3 = np.load(normpath+'sigma3.npy')
    return sig,sig1,sig2,sig3


def get_power(s1,ti,newbinary=1):
    if newbinary==1: make_power(s1,ti)
    vol = []
    points = []
    for t in ti:
        pathv = s1.resultpath+'/voronoi/vor_'+format(t,'.5f')+'.npy'
        pathvol = s1.resultpath+'/voronoi/vorV_'+format(t,'.5f')+'.npy'
        vol.append(np.load(pathvol))
        points.append(np.load(pathv))
    return vol, points


def read_voro(s1):
    path = s1.resultpath+'/voronoi'
    vol   = []
    verts = []
    t = []
    for file in os.listdir(path):
        if 'vor_' in file:
            vol.append(np.load(path+'/'+file))
        if 'vorV_' in file:
            verts.append(np.load(path+'/'+file))
            t.append(extract_float(file))
    i_t = np.argsort(t)
    ts = []
    vols =[]
    vers = []
    for i in i_t:
        vols.append(vol[i])
        vers.append(verts[i])
        ts.append(t[i])
    return vols, vers, ts


def delaunay_lengths(points):
    triangulation = Delaunay(points, qhull_options="QJ Pp")

    def calculate_distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    unique_edges = set()

    edge_lengths = []

    for simplex in triangulation.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                if edge not in unique_edges:
                    vertex1 = triangulation.points[edge[0]]
                    vertex2 = triangulation.points[edge[1]]
                    distance = calculate_distance(vertex1, vertex2)
                    edge_lengths.append(distance)
                    unique_edges.add(edge)
    eal = np.array(edge_lengths)
    return eal


def make_rand_delaunay(s1,Nv=500,Nsub=20):
    from concurrent.futures import ProcessPoolExecutor
    print('Generating Random Delaunay Triangulation')

    Nsi = round(Nv/Nsub)
    iN = np.arange(0,Nsub)*Nsi

    with ProcessPoolExecutor() as executor:
        results = executor.map(voro_VI, [s1]*Nsub,[Nsi]*Nsub, iN,[0]*Nsub,[True]*Nsub)

    ld,_ = zip(*results)
    ld = np.concatenate(ld)

    x = np.linspace(0,np.max(ld),200)
    hld,_ = np.histogram(ld,bins=x)
    hld = hld / np.sum(hld)
    x = x[0:-1] + np.diff(x)/2
    try:
        os.mkdir('./pyprime/misc/reference_data/random_delaunay')
    except:
        pass
    try:
        os.mkdir(f"./pyprime/misc/reference_data/random_delaunay/{s1.label}")
    except:
        pass
    normpath = f"./pyprime/misc/reference_data/random_delaunay/{s1.label}/rand_{Nv}_"
    np.save(normpath+'x.npy',x)
    np.save(normpath+'hld.npy',hld)


def rand_delaunay(s1,Nv=500,Nsub=50,newbinary=0):
    if newbinary==1:
        make_rand_delaunay(s1,Nv=Nv,Nsub=Nsub)
    normpath = f"./pyprime/misc/reference_data/random_delaunay/{s1.label}/rand_{Nv}_"
    hdl = 0
    x =0
    try:
        x = np.load(normpath+'x.npy')
        hdl = np.load(normpath+'hld.npy')
    except:
        pass
    return x,hdl


def extract_float(name, out=False):
    name = name[0:-4]
    numeric_part = float(''.join(filter(lambda x: x.isdigit() or x in ".-", name)))
    return numeric_part
