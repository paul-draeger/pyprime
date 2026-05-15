# pyprime

Python package for the pyprime analysis library.

## Installation

2. Add the Package to your environment 
```bash
python -m pip install -e . --no-deps
```
using `--no-deps` is recommended.

## Configuration
Before the Backend can be used, the user has to configure a "remote_config.json".
1. Copy the "remote_config.example.json" and rename it to "remote_config.json".
2. Configure a SSH connection and mirror directory
  Example: Horse filesystem @ZIH,TU Dresden
  1. Creat a SSH key pair (https://compendium.hpc.tu-dresden.de/access/ssh_login/)
  2. Add an entry to the "remote_config.json".
    ```
      "barnardExample": {
        "hostname": "dataport1.hpc.tu-dresden.de",
        "sshhost": "login1.barnard.hpc.tu-dresden.de",
        "port": 22,
        "username": "YOUR_USERNAME",
        "remote_file_path": "/path/to/workspace/",
        "local_filesystem": "/path/to/local/mirror/"
      }
    ``` 
  4. Set your ZIH username (e.g. s*******)
  5. Set the path to your workspace. Add "/" at the end of the path.
  6. Set the local mirror directory. Make sure it has enough space (using an external SSD drive is recommended).

## Usage
1. Import the package
```python
import pyprime as pp
```  
3. Create a "sim" object
```python
s1 = pp.sim("path/to/results/folder/in/your/workspace",ssh_pass="barnardExample",ssh_update=update,newbinary=True)
```
This will create a local mirror and construct the object "s1". Running this once will allow using the following options to save time. 
- Set "ssh_update=False" to load simulation data from the mirror directory without updating it.
- Set "newbinary=False" to reload monitoring files from numpy-binary format without updating them. 
4. Access "ellipsoid" related data
```python
If=1
f1 = s1.f(If)
```
This will create a "fluid" object. "If" is the index of the fluid file. The corresponding time is stored in the s1.tf array. The object "f1" now contains the fluid velocity field and the pressure field.
```
u = f1.u
v = f1.v
w = f1.w
p = f1.p
```
The fields are stored as numpy matrices (e.g. u[1:nx,1:ny,1:nz]). The grid coordinates can be acesses through the sim class.
```
Xu = s1.Xu
Yu = s1.Yu
Zu = s1.Zu
```
The meshgrid consits of the matrices Xu,Yu,Zu. Here, "u" denotes the grid for the u-velocity, since PRIME uses a staggered grid. Equivalently, "Xv","Xw",... are available. The center coordinates are stored as "Xp,Yp,Zp". 

5. Access "ellipsoid" related data
```python
Ie=1
e1 = s1.e(Ie)
```
This will create an "ellipsoid" object. "Ie" is the ellipsoid ID starting at 1.
6. Access "bubble" related data
```python
Ib=1
b1 = s1.b(Ib)
```
This will create a "bubble" object. "Ib" is the bubble ID starting at 1.


