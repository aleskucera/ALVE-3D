import os
import yaml
import subprocess

# # ===================================== Define the parameters =====================================
USER = 'kuceral4'
CONDA_ENV = 'ALVE-3D'
SINGULARITY_IMAGE = os.path.join(os.getcwd(), 'singularity/alve-3d.sif')

RECIPE_FILE = os.path.join(os.getcwd(), 'singularity/recipe.def')
ENVIRONMENT_FILE = os.path.join(os.getcwd(), 'environment.yaml')

EXCLUDED_PACKAGES = ['pytorch3d']

print('===============================================')
print('\tSINGULARITY IMAGE BUILD SCRIPT')
print('===============================================')
print('\nINFO: Parameters:')
print(f'\t- working directory: {os.getcwd()}')
print(f'\t- conda environment: {CONDA_ENV}')
print(f'\t- singularity image: {SINGULARITY_IMAGE}')
print(f'\t- recipe file: {RECIPE_FILE}')
print(f'\t- environment file: {ENVIRONMENT_FILE}')
print(f'\t- excluded packages:')
for package in EXCLUDED_PACKAGES:
    print(f'\t\t- {package}')
print(f'\t- ssh user: {USER}')

# ===================================== Export the conda environment =====================================

print('\n---- Exporting the Anaconda environment ----\n')

# Check if the conda environment exists
result = subprocess.run(['conda', 'env', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result.returncode != 0:
    print('ERROR: Could not list the conda environments')
    exit(1)
out = result.stdout.decode('utf-8')
err = result.stderr.decode('utf-8')
if CONDA_ENV not in out:
    print('ERROR: Could not find the conda environment')
print(f'INFO: Found conda environment {CONDA_ENV}, exporting it to {ENVIRONMENT_FILE}')

# Export the conda environment
result = subprocess.run(['conda', 'env', 'export', '--no-builds', '--name', CONDA_ENV, '--file', ENVIRONMENT_FILE],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if result.returncode != 0:
    print('ERROR: Could not export the conda environment')
    exit(1)
print(f'INFO: Environment file written: {ENVIRONMENT_FILE}')

# ===================================== Remove excluded packages =====================================

print('\n---- Removing excluded packages from the environment file ----\n')

with open(ENVIRONMENT_FILE, 'r') as f:
    environment = yaml.safe_load(f)

# Remove the excluded packages from the environment file
for package in EXCLUDED_PACKAGES:
    if package in environment['dependencies']:
        for conda_package in environment['dependencies']:
            if package in conda_package:
                print(f'INFO: Removing package {package} from conda dependencies')
                environment['dependencies'].remove(conda_package)
    elif 'pip' in environment['dependencies'][-1].keys():
        for pip_package in environment['dependencies'][-1]['pip']:
            if package in pip_package:
                print(f'INFO: Removing package {pip_package} from pip dependencies')
                environment['dependencies'][-1]['pip'].remove(pip_package)

# Remove the prefix from the environment file
if 'prefix' in environment.keys():
    environment.pop('prefix')

# Save the updated environment file
with open(ENVIRONMENT_FILE, 'w') as f:
    yaml.dump(environment, f)

# ===================================== Build the singularity image =====================================

print('\n---- Building the Singularity image ----\n')

result = subprocess.run(['sudo', 'singularity', 'build', '--nv', SINGULARITY_IMAGE, RECIPE_FILE])
if result.returncode != 0:
    print('ERROR: Could not build the singularity image')
    exit(1)
print(f'INFO: Singularity image built: {SINGULARITY_IMAGE}')

# ===================================== Copy the image to the RCI server =====================================

answer = input('\nWould you like to copy the image to the login3.rci.cvut.cz server? [y/n] ')
if answer == 'y':
    result = subprocess.run(
        ['scp', SINGULARITY_IMAGE, 'kuceral4@login3.rci.cvut.cz:/home/kuceral4/ALVE-3D/singularity'])

    if result.returncode != 0:
        print('ERROR: Could not copy the image to the login3.rci.cvut.cz server')
        exit(1)

    print('INFO: Image copied to the login3.rci.cvut.cz server')

else:
    print('INFO: Image not copied to the login3.rci.cvut.cz server')
