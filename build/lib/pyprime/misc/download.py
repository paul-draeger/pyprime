import paramiko
from ..paths import Spath
import os
from tqdm import tqdm  #
import stat
from ..AMP.amp_functions import sort_files
import concurrent.futures


allnew = False
# Define the SSH parameters
hostname = 'dataport1.hpc.tu-dresden.de'
port = 22
username = 's5941119'
password = 'b0:UF]53qZc'

remote_file_path = '/data/horse/ws/s5941119-pfo-draeger/s5941119-pfo_draeger/'

local_file_path = ''
paths = Spath()

ssh_client = paramiko.SSHClient()

ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())


def download_fluid(rpath,lpath):
    try:
        # Connect to the remote server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, port, username, password)

        # Open an SFTP session
        sftp = ssh_client.open_sftp()

        if not os.path.exists(lpath+'/results'): os.mkdir(lpath+'/results')
        if not os.path.exists(lpath+'/results/fluid'): os.mkdir(lpath+'/results/fluid')
        rpath = rpath + '/results/fluid'
        lpath = lpath + '/results/fluid'
        fluidfiles = sort_files(sftp.listdir(rpath))
        if len(fluidfiles)>11:
            fltoget = fluidfiles[-11:]
        else:
            fltoget = fluidfiles
        for fl in fltoget:
            print(fl)
            flRpath = os.path.join(rpath,fl)
            flLpath = os.path.join(lpath,fl)
            if not os.path.exists(flLpath):
                sftp.get(flRpath, flLpath)

        sftp.close()


    except Exception as e:
        print(f"An error occurred: {e}")


def download_directory(src_dir, dst_dir,pics=False):
    exclude_names = ['ellipsoid_trn','bubble_001_MM','ellipsoid_rst','fl_slices','fluid', 'build', 'postproc','postproc_bubble', 'preproc', 'job.sh', 'Makefile', 'Report.txt', 'vis_check','vis_checkT0','temp_spectral', 'rst_info.dat.bak','ellipsoid','rst_info.dat']
    update_list   = ['ux.npy','uy.npy','vx.npy','vy.npy','collected_binary.npy','collected_binary_eongrid.npy','collected_binary_ti.npy']#['monitor.dat','.eps','rst_info']
    if not pics:
        exclude_names.append('vis_C1')
        exclude_names.append('vis_C2')
        exclude_names.append('vis_A')
        exclude_names.append('vis_S')
    try:
        # Connect to the remote server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname, port, username, password)

        # Open an SFTP session
        sftp = ssh_client.open_sftp()

        def download_recursive(remote_path, local_path):
            if not os.path.exists(local_path): os.mkdir(local_path)
            for item in sftp.listdir(remote_path):
                print(item)
                remote_item_path = os.path.join(remote_path, item)
                local_item_path = os.path.join(local_path, item)

                # Check if the item should be excluded
                if item in exclude_names:
                    continue

                # Check if the item is a directory
                if stat.S_ISDIR(sftp.lstat(remote_item_path).st_mode):
                    os.makedirs(local_item_path, exist_ok=True)
                    download_recursive(remote_item_path, local_item_path)
                else:
                    if not os.path.exists(local_item_path) and not allnew:
                        print('...download')
                        sftp.get(remote_item_path, local_item_path)
                        print('done')
                    else:
                        if any(substring in item for substring in update_list):
                            print('...download')
                            sftp.get(remote_item_path, local_item_path)
                            print('done')
        # Start downloading from the remote directory
        download_recursive(src_dir, dst_dir)

        # Close the SFTP session and transport
        sftp.close()

    except Exception as e:
        print(f"An error occurred: {e}")



def download_file(path):
    print('HEEEEEEEEEEEEEEEEEEY')
    print(path)
    rpath = remote_file_path + path
    lpath = local_file_path + path
    if not os.path.exists(lpath):
        os.mkdir(lpath)
    # Replace the following lines with your actual download functions
    #download_fluid(rpath, lpath)
    download_directory(rpath, lpath)
    print(f"Downloaded: {path}")

# Use ThreadPoolExecutor for parallel execution
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     executor.map(download_file, paths)
for path in paths:
    download_file(path)

# for path in paths:
#     rpath = remote_file_path + path
#     lpath = local_file_path + path
#     if not os.path.exists(lpath): os.mkdir(lpath)
#     download_fluid(rpath, lpath)
#     download_directory(rpath, lpath)
