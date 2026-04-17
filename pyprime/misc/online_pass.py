def get_online_pass(key,output=True):
    if key=='barnard1':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = 'dataport1.hpc.tu-dresden.de'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/data/horse/ws/s5941119-pfo-draeger/'
        local_filesystem = '/media/paul/PortableSSD/Diplom/barnard_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    if key=='huegel':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = '141.30.240.180'
#        hostname = '141.30.241.76'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/scratch/schoppmann_rebel_draeger/'
#        remote_file_path = '/home/s5941119/Dokumente/Diplomarbeit/'
        local_filesystem = '/media/paul/PortableSSD/Diplom/huegel_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    if key=='berg':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = '141.30.240.180'
#        hostname = '141.30.241.76'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/home/s5941119/Dokumente/WHK/'
#        remote_file_path = '/home/s5941119/Dokumente/Diplomarbeit/'
        local_filesystem = '/media/paul/PortableSSD1/Diplom/huegel_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    if key=='blanc':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = '141.30.240.180'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/home/s5941119/Dokumente/Diplomarbeit/'
        local_filesystem = '/media/paul/PortableSSD/Diplom/huegel_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    elif key=='barnard2':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = 'dataport2.hpc.tu-dresden.de'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/data/horse/ws/s5941119-pfo-draeger/'
        local_filesystem = '/media/paul/PortableSSD/Diplom/barnard_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    elif key=='barnardD':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = 'dataport1.hpc.tu-dresden.de'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/data/horse/ws/s5941119-diplom/'
        local_filesystem = '/media/paul/PortableSSD/Diplom/barnard_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    elif key=='barnardDD':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = 'dataport1.hpc.tu-dresden.de'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/data/horse/ws/s5941119-diplom/'
        local_filesystem = '/media/paul/PortableSSD/Diplom/barnard_mirror2/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    elif key=='barnardDR':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = 'dataport1.hpc.tu-dresden.de'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/data/horse/ws/s5941119-diplom/'
        local_filesystem = '/media/paul/PortableSSD/Diplom/barnard_res/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    elif key=='barnardER':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = 'dataport1.hpc.tu-dresden.de'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/data/horse/ws/s5941119-electrolysis/'
        local_filesystem = '/media/paul/PortableSSD/Diplom/barnard_res/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    elif key=='barnardE':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = 'dataport1.hpc.tu-dresden.de'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/data/horse/ws/s5941119-electrolysis/'
        local_filesystem = '/media/s5941119/PortableSSD1/Diplom/barnard_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem

    elif key=='PSMbarnardE':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = 'dataport2.hpc.tu-dresden.de'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/data/horse/ws/s5941119-electrolysis/'
        local_filesystem = '/media/s5941119/PortableSSD/Diplom/barnard_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem

    if key=='PSMberg':
        if output: print(f'SSH Pass valid: "{key}"')
#        hostname = '141.30.240.180'
        hostname = '141.30.241.76'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/home/s5941119/Dokumente/WHK/'
#        remote_file_path = '/home/s5941119/Dokumente/Diplomarbeit/'
        local_filesystem = '/media/s5941119/PortableSSD1/Diplom/barnard_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem

    if key=='PSMtanne':
        if output: print(f'SSH Pass valid: "{key}"')
#        hostname = '141.30.240.180'
        hostname = '141.30.240.186'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/tanneberg_scratch_ssd/draeger/001_bubble_method/'
#        remote_file_path = '/home/s5941119/Dokumente/Diplomarbeit/'
        local_filesystem = '/media/s5941119/PortableSSD/Diplom/barnard_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem



    if key=='PSMZauberberg':
        if output: print(f'SSH Pass valid: "{key}"')
        hostname = '141.30.241.76'
#        hostname = '141.30.241.76'
        port = 22
        username = 's5941119'
        password = 'b0:UF]53qZc'
        remote_file_path = '/home/s5941119/Dokumente/WHK/'
#        remote_file_path = '/home/s5941119/Dokumente/Diplomarbeit/'
        local_filesystem = '/media/s5941119/PortableSSD1/Diplom/barnard_mirror/'
        return hostname, port, username, password, remote_file_path, local_filesystem


    else:
        print('Error: Online pass not found.')
        return None
