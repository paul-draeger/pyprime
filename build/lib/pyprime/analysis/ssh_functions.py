from ..misc.online_pass import get_online_pass
import paramiko
from ..paths import Spath
import os
from tqdm import tqdm  #
import stat
from ..AMP.amp_functions import sort_files
import concurrent.futures
import json
from scp import SCPClient

def save_dict_to_file(dictionary, file_path):
    with open(file_path, 'w') as file:
        json.dump(dictionary, file, indent=4)


def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def ssh_download_sim(path,ssh_pass,ssh_download_fluid=False, ssh_update=True, bub_fp=False):
    _, _, _, _, remote_file_path, local_filesystem = get_online_pass(ssh_pass)
    path_id_dict = local_filesystem + 'id_dict.json'
    new_mirror = False
    if os.path.exists(path_id_dict):
        id_dict = load_dict_from_file(path_id_dict)
        if path in id_dict:
            print('Simulation data was already downloaded. Checking for updates...')
            lpath = local_filesystem + id_dict[path]
        else:
            last_entry = list(id_dict.items())[-1]
            _, last_value = last_entry
            new_ID = int(extract_numeric(last_value)) + 1
            llpath = f'sim_results_{new_ID}'
            lpath = local_filesystem + llpath
            new_entry = {path: llpath}
            id_dict.update(new_entry)
            print(f'Created a new directory in the local filesystem, ID = {new_ID}')
            save_dict_to_file(id_dict,path_id_dict)
    else:
        print(f'Created the first directory in this local filesystem, ID = 1')
        new_dict = {path: 'sim_results_1'}
        save_dict_to_file(new_dict,path_id_dict)
        lpath = local_filesystem + '/sim_results_1'
        new_mirror = True
    rpath = remote_file_path + path

    if not os.path.exists(lpath):
        os.mkdir(lpath)
    # Replace the following lines with your actual download functions
    #download_fluid(rpath, lpath)
    if ssh_update or new_mirror:
        print('Updating local simulation Data...')
        download_directory(rpath, lpath, ssh_pass, bub_fp = bub_fp)
        print(f"SSH data checked. Local directory is {lpath}")

        if ssh_download_fluid:
            download_fluid(rpath,lpath,ssh_pass,I=0)
        else:
            download_fluid(rpath,lpath,ssh_pass,I=-1)

    return lpath, rpath



def ssh_get_fluid(i, ssh_pass, lpath, rpath):
    download_fluid(rpath,lpath,ssh_pass,I=i)





def download_directory(remote_path, local_path,ssh_pass,pics=False, bub_fp = False):
    exclude_names = ['ellipsoid_trn','ellipsoid_rst','fluid','fl_slices', 'build', 'postproc','postproc_bubble', 'preproc', 'job.sh', 'Makefile', 'Report.txt', 'vis_check','vis_checkT0','fp_00']
    update_list   = ['cnm.bin','cnmR.bin','cnm.dat','cnm_pnt.bin','ux.npy','uy.npy','vx.npy','vy.npy','collected_binary.npy','collected_binary_eongrid.npy','collected_binary_ti.npy','rst_info.dat','monitor']#['monitor.dat','.eps','rst_info']
    bubfpnames = ['_fp','_MM','_fpf','_tc_','_pri_','_sk_','tau','_fpMu','_fpMu1','_fpMui','_fp_Ct','_fp_unc','_fp_dotc','_fp_un','_fp_ph','_fp_sk','_umove','_fpu','_fpui','_xloci','_xcf','_trn','_rst','_tfe','_tfe_g','marker','_tra','_tra_theta','I1m','errN','JSr','tc','tct']
    if not bub_fp: exclude_names = exclude_names + bubfpnames
    if not pics:
        exclude_names.append('vis_C1')
        exclude_names.append('vis_C2')
        exclude_names.append('vis_A')
        exclude_names.append('vis_S')
    try:
        # Connect to the remote server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        hostname, port, username, password, _, _ = get_online_pass(ssh_pass,output=False)
        ssh_client.connect(hostname, port, username, password)

        # Open an SFTP session
        sftp = ssh_client.open_sftp()

        #stdin, stdout, stderr = ssh_client.exec_command("hostname; whoami; pwd; echo $HOME; ls -ld / /home /scratch /work 2>/dev/null")
        #print(stdout.read().decode())
        #print(stderr.read().decode())


        def download_recursive(remote_path, local_path):
            if not os.path.exists(local_path): os.mkdir(local_path)

            for item in sftp.listdir(remote_path):
                #print(item)
                remote_item_path = os.path.join(remote_path, item)
                local_item_path = os.path.join(local_path, item)

                newitem = False
                # Check if the item should be excluded
                for exclude_string in exclude_names:
                    if exclude_string in item: newitem=True
                if newitem: continue
                # Check if the item is a directory
                if stat.S_ISDIR(sftp.lstat(remote_item_path).st_mode):
                    os.makedirs(local_item_path, exist_ok=True)
                    download_recursive(remote_item_path, local_item_path)
                else:
                    if not os.path.exists(local_item_path):
                        print(f'- New file found: Dowonloading {item}')
                        sftp.get(remote_item_path, local_item_path)
                    else:
                        if any(substring in item for substring in update_list):
                            print(f'- Updating {item}')
                            sftp.get(remote_item_path, local_item_path)
                            #print('done')

        # Start downloading from the remote directory
    
        download_recursive(remote_path, local_path)

        bdata = '/input_bubble_data.dat'
        edata = '/input_ellipsoid_data.dat'
        idata = '/input.dat'
        if not os.path.exists(local_path + bdata):
            sftp.get( remote_path + '/..' + bdata, local_path + bdata)
        if not os.path.exists(local_path + idata):
            sftp.get( remote_path + '/..' + idata, local_path + idata)
        if not os.path.exists(local_path + edata):
            sftp.get( remote_path + '/..' + edata, local_path + edata)

        # Close the SFTP session and transport
        sftp.close()

    except FileNotFoundError:
        print("Directory does not exist")
    except IOError as e:
        print(f"IOError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def download_fluid(rpath,lpath,ssh_pass,I=-1):
    try:
        # Connect to the remote server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        hostname, port, username, password,_,_ = get_online_pass(ssh_pass)
        ssh_client.connect(hostname, port, username, password)

        # Open an SFTP session
        sftp = ssh_client.open_sftp()

        if not os.path.exists(lpath+'/fluid'): os.mkdir(lpath+'/fluid')
        rpath = rpath + '/fluid'
        lpath = lpath + '/fluid'
        fltoget = sort_files(sftp.listdir(rpath))
        for fl in fltoget:
            if fl=='x.dat' or fl=='y.dat' or fl=='z.dat':
                flRpath = os.path.join(rpath,fl)
                flLpath = os.path.join(lpath,fl)
                print(f'Downloading fluid data - {fl}')
                sftp.get(flRpath, flLpath)
            elif I!=-1:
                fl_id = extract_numeric(fl)
                if fl_id==I or I==0:
                    flRpath = os.path.join(rpath,fl)
                    flLpath = os.path.join(lpath,fl)
                    if not os.path.exists(flLpath):
                        print(f'Downloading fluid data - {fl}')
                        sftp.get(flRpath, flLpath)

        sftp.close()


    except Exception as e:
        print(f"An error occurred: {e}")


def ssh_get_files(sim,files):
    try:
        # Connect to the remote server
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        hostname, port, username, password, _, _ = get_online_pass(sim.ssh_pass)
        ssh_client.connect(hostname, port, username, password)

        # Open an SFTP session
        sftp = ssh_client.open_sftp()
        if not isinstance(files, list): files = [files]
        for file in files:
            lpath = sim.resultpath + '/' +file
            ldir = os.path.abspath(os.path.dirname(lpath))
            os.makedirs(ldir,exist_ok = True)
            rpath = sim.rpath + '/' +file
            print(f'Downloading: {os.path.basename(file)}')
            sftp.get(rpath, lpath)

        sftp.close()

    except Exception as e:
        print(f"An error occurred: {e}")


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
            print("no content")
        return 10000
