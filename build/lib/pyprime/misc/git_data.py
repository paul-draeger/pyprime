import os
import shutil


def configurations_to_git():
    print(os.getcwd())
    print('Copy Configuration Data to Git Repository...')
    files = ['/gridbinary/gridinfo.npz', '/rst_info.dat', '/input_ellipsoid_data.dat']
    git_dest = 'pyprime/misc/configuration_data'
    try:
        os.mkdir(git_dest)
    except:
        pass
    for i,path in enumerate(Spath()):
        print(f"Configuration - {i+1}")
        gitdir = git_dest+'/'+path
        try:
            os.mkdir(gitdir)
            os.mkdir(gitdir + '/gridbinary')
        except:
            pass
        for file in files:
            shutil.copy('./'+path+'/results'+file, gitdir+file)
    print('Done')

def sim_from_git():
    from ..analysis.sim import sim
    sims = []
    for path in Spath():
        gitpath = './pyprime/misc/configuration_data/'+path
        sims.append(sim(gitpath,newbinary=0,onlyInfo=True))
    return sims
