import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import os
import shutil
import time
import re
import sympy as sp



class prop:
    def __init__(self,d=1, vis = 1E-6, g=9.81, rhof=1E3, rhog=1, sig=0.07 ):
        self.d    = d
        self.vis  = vis
        self.g    = g
        self.rhof = rhof
        self.rhog = rhog
        self.sig  = sig

    def sim_ref(self,s1):
        Lref = self.d / s1.d_out
        tref = np.sqrt( s1.g / self.g * Lref )
        tref2 = s1.vis / self.vis * Lref**2
        mref = self.sig / s1.sig * tref**2
        mref2 = Lref**3 * self.rhof / s1.rhof
        mref3 = Lref**3 * self.rhog / s1.rhog

        tol = 1E-2
        dt = np.abs(tref - tref2) / tref
        dm = np.max([np.abs(mref-mref2),np.abs(mref-mref3),np.abs(mref2-mref3)]) / mref
        if dt<tol and dm<tol:
            print("Refernzgrößen eingehalten")
            print(f"L_ref = {Lref}")
            print(f"t_ref = {tref}")
            print(f"m_ref = {mref}")
            print(f"vis_s = {s1.vis}")
            print(f"g_s   = {s1.g}")
            print(f"d_s   = {s1.d_out}")
            print(f"sig_s = {s1.sig}")
        else:
            print("ACHTUNG: Referenzgrößen passen nicht zusammen!!")
            print(f"Abeichung >1% - dt_ref={dt:.3f}")
        return Lref,tref,mref

    def calculate_sim_values(self,prop2,alpha=0):
        val1 = np.array([prop2.d, prop2.g, prop2.vis, prop2.rhof, prop2.sig])
        val_ind = val1!=0

        if np.sum(val_ind)>=3:
            print('Solving...')
            # Define the symbols
            Lref,tref,mref = sp.symbols('Lref tref mref')

            eq = []
            # Define the equations
            if (val1[0] != 0):
                eq.append( sp.Eq(self.d/prop2.d, Lref ))
            if (val1[1] != 0):
                eq.append( sp.Eq(self.g/prop2.g, Lref / tref**2 ))
            if (val1[2] != 0):
                eq.append( sp.Eq(self.vis/prop2.vis, Lref**2 / tref ))
            if (val1[3] != 0):
                eq.append( sp.Eq(self.rhof/prop2.rhof, mref / Lref**3 ))
            if (val1[4] != 0):
                eq.append( sp.Eq(self.sig/prop2.sig, mref / tref**2 ))

            I = 0

            # Solve the system of equations
            Ref   = sp.solve(eq, (Lref,tref,mref))
            LR    = float(Ref[I][0].evalf())
            tR    = float(Ref[I][1].evalf())
            mR    = float(Ref[I][2].evalf())
            gR    =        LR / tR**2
            visR  =     LR**2 / tR
            rhoR  =        mR / LR**3
            sigR  =        mR / tR**2

            L2    =    self.d / LR
            g2    =    self.g / gR
            vis2  =  self.vis / visR
            rho2  = self.rhof / rhoR
            rhog2 = self.rhog / rhoR
            sig2  =  self.sig / sigR
            uref  =        LR / tR

            output = f'L = {L2} \ng = {g2} \nvis = {vis2} \nrho = {rho2} \nsig = {sig2} \nrhog = {rhog2}'

            outR = f'\n \nLR = {LR} \ntR = {tR} \nmR = {mR}'
            print(output)
            print(outR)
            print(f"uR = {uref}")

            if alpha!=0:
                print("für die Schräge Kollision:")
                theta = alpha/180 * np.pi
                gy = np.sin(theta)*g2
                gz = np.cos(theta)*g2
                print(f'\n \ngy = {gy} \ngz = {gz}')

            prop3 = prop(d=L2,g=g2,vis=vis2,rhof=rho2,rhog=rhog2,sig=sig2)
            return LR,tR,mR,prop3
        else:
            print('Input stimmt nicht (mind. 3 variabelen =0)')
            return None
