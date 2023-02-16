import os
import yaml
import subprocess

user = 'kuceral4'
conda_env = 'ALVE-3D'
singularity_image = 'alve-3d2.sif'

recipe_file = os.path.join(os.getcwd(), 'singularity/recipe.def')
environment_file = os.path.join(os.getcwd(), 'environment.yaml')

excluded_packages = ['open3d', 'pytorch3d', 'jakteristics']

print('===============================================')
print('\tSINGULARITY IMAGE BUILD SCRIPT')
print('===============================================')
print('\nParameters:')
print(f'\t- working directory: {os.getcwd()}')
print(f'\t- conda environment: {conda_env}')
print(f'\t- singularity image: {singularity_image}')
print(f'\t- recipe file: {recipe_file}')
print(f'\t- environment file: {environment_file}')
print(f'\t- excluded packages:')
for package in excluded_packages:
    print(f'\t\t- {package}')
print(f'\t- ssh user: {user}')

print('\n---- Exporting the Anaconda environment ----\n')

result = subprocess.run(['conda', 'env', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if result.returncode != 0:
    print('ERROR: Could not list the conda environments')
    exit(1)

out = result.stdout.decode('utf-8')
err = result.stderr.decode('utf-8')

if conda_env not in out:
    print('ERROR: Could not find the conda environment')
print(f'INFO: Found conda environment {conda_env}, exporting it to {environment_file}')

result = subprocess.run(['conda', 'env', 'export', '--no-builds', '--name', conda_env, '--file', environment_file],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if result.returncode != 0:
    print('ERROR: Could not export the conda environment')
    exit(1)

print(f'INFO: Environment file written: {environment_file}')

with open(environment_file, 'r') as f:
    environment = yaml.safe_load(f)

print('\n---- Removing excluded packages from the environment file ----\n')
# Remove the excluded packages from the environment file
for package in excluded_packages:
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

# Write the environment file
with open(environment_file, 'w') as f:
    yaml.dump(environment, f)

print('\n---- Building the Singularity image ----\n')

result = subprocess.run(['sudo', 'singularity', 'build', '--nv', singularity_image, recipe_file])

if result.returncode != 0:
    print('ERROR: Could not build the singularity image')
    exit(1)

answer = input('Would you like to copy the image to the login3.rci.cvut.cz server? [y/n] ')

if answer == 'y':
    result = subprocess.run(
        ['scp', singularity_image, 'kuceral4@login3.rci.cvut.cz:/home/kuceral4/ALVE-3D/singularity'])

    if result.returncode != 0:
        print('ERROR: Could not copy the image to the login3.rci.cvut.cz server')
        exit(1)

    print('INFO: Image copied to the login3.rci.cvut.cz server')

else:
    print('INFO: Image not copied to the login3.rci.cvut.cz server')
