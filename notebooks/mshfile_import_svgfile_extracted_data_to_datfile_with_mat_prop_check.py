## Store mesh cell midpoints to dat file
import os
import glob
import meshio
import numpy as np

def convert_msh_to_dat(msh_file, dat_file):
    # Read the mesh file
    mesh = meshio.read(msh_file)

    # Get the cell midpoints
    cell_midpoints = []
    for cell_block in mesh.cells:
        cell_type = cell_block.type
        cell_data = cell_block.data
        if cell_type == "triangle":
            for cell_vertices in cell_data:
                # Calculate the midpoint of the cell
                midpoint = np.mean(mesh.points[cell_vertices], axis=0)
                cell_midpoints.append(midpoint)

    # Convert the cell midpoints to a numpy array
    cell_midpoints = np.array(cell_midpoints)

    # Save the cell midpoints to the .dat file
    np.savetxt(dat_file, cell_midpoints, fmt="%.6f")

def convert_msh_files_in_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of .msh file paths in the input folder
    msh_files = glob.glob(os.path.join(input_folder, "*.msh"))

    # Convert each .msh file to .dat
    for msh_file in msh_files:
        # Get the base name of the .msh file without extension
        file_name = os.path.splitext(os.path.basename(msh_file))[0]

        # Specify the corresponding .dat file path
        dat_file = os.path.join(output_folder, file_name + ".dat")

        # Convert .msh to .dat
        convert_msh_to_dat(msh_file, dat_file)

# Specify the input folder and output folder paths
input_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/msh"  # Specify the path to the input folder containing the .msh files
output_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/dat"  # Specify the path to the output folder to save the .dat files

# Convert .msh files in the input folder and save them in the output folder
convert_msh_files_in_folder(input_folder, output_folder)

## Take the flipped svg image file, mesh file and dat file to store color
import os
import glob
import numpy as np
import meshio
import cairosvg
import io
import imageio

def convert_image_to_dat(image_file, msh_file, dat_file):
    # Check if the image is in .svg format
    if image_file.lower().endswith('.svg'):
        # Load the SVG image and render it to a numpy array
        svg_data = cairosvg.svg2png(url=image_file)
        image = imageio.v2.imread(io.BytesIO(svg_data))
    # Load the mesh from .msh file
    mesh = meshio.read(msh_file)

    # Get the cell midpoints
    cell_midpoints = []
    for cell_block in mesh.cells:
        cell_type = cell_block.type
        cell_data = cell_block.data
        if cell_type == "triangle":
            for cell_vertices in cell_data:
                # Calculate the midpoint of the cell
                midpoint = np.mean(mesh.points[cell_vertices], axis=0)
                cell_midpoints.append(midpoint)

    # Convert the image to .dat
    with open(dat_file, "w") as file:
        for midpoint in cell_midpoints:
            # Determine the color of the cell
            x_index = np.clip(int(midpoint[0]), 0, image.shape[1] - 1)
            y_index = np.clip(int(midpoint[1]), 0, image.shape[0] - 1)
            color = image[y_index, x_index][:3]

            # Determine the fourth column value based on color
            fourth_column = 1 if color[0] >= 127.5 else 0

            # Write the XYZ coordinates and fourth column value to the .dat file
            file.write(f"{midpoint[0]} {midpoint[1]} {midpoint[2]} {fourth_column}\n")


def convert_files_in_folders(msh_folder, image_folder, dat_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(dat_folder, exist_ok=True)

    # Get a list of file paths in the respective input folders
    msh_files = glob.glob(os.path.join(msh_folder, "*.msh"))
    image_files = glob.glob(os.path.join(image_folder, "*.svg"))
    dat_files = glob.glob(os.path.join(dat_folder, "*.dat"))

    # Sort the file paths in alphabetical order
    msh_files.sort()
    image_files.sort()
    dat_files.sort()

    # Iterate over the files and convert them
    for msh_file, image_file, dat_file in zip(msh_files, image_files, dat_files):
        convert_image_to_dat(image_file, msh_file, dat_file)

# Specify the input folders and output folder paths
msh_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/msh"  # Path to the folder containing the .msh files
image_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/images/svg"  # Path to the folder containing the images
dat_folder = "/mnt/d/Research Projects/FEniCS/plasticity/a/dat"  # Path to the folder to save the .dat files

# Convert the files in the folders
convert_files_in_folders(msh_folder, image_folder, dat_folder)

##The below code will be just for a test purpose to check the material is correctly assigned to mesh cell or not using paraview or not
import meshio
mesh_from_file = meshio.read("/mnt/d/Research Projects/FEniCS/mspaint/partha svgfiles for demo/msh/2.msh")

import numpy
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
    return out_mesh

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write("p2.xdmf", triangle_mesh)

from dolfin import * 
mesh = Mesh()
with XDMFFile("p2.xdmf") as infile:
    infile.read(mesh)

plot(mesh);

def extract_fourth_column(dat_file):
    fourth_column = []
    with open(dat_file, "r") as file:
        for line in file:
            elements = line.strip().split()
            fourth_column.append(int(elements[3]))  #fourth column is at index 3
    return fourth_column

# Provide the path to the DAT file
dat_file = "/mnt/d/Research Projects/FEniCS/mspaint/partha svgfiles for demo/dat/2.dat"

# Extract the fourth column elements
fourth_column_array = extract_fourth_column(dat_file)

import dolfin
import numpy as np
materials = MeshFunction('double', mesh, 2)

G = VectorFunctionSpace(mesh, "DG", 0)
g = Function(G)

local_values_material = np.zeros_like(g.vector().get_local())

for cell in cells(mesh):
    midpoint = cell.midpoint().array()
    i = (midpoint[0])
    j = (midpoint[1])
    k = (midpoint[2])
    local_values_material[cell.index()] = fourth_column_array[cell.index()]
    materials[cell] = int(local_values_material[cell.index()])
    
g.vector().set_local(local_values_material)

dolfin.XDMFFile(dolfin.MPI.comm_world, "p2.xdmf").write_checkpoint(g,"g",0)

import dolfin as dol
class al(dol.UserExpression):
    def __init__(self, materials, al0, al1, **kwargs):
        super().__init__(**kwargs)
        self.materials = materials
        self.k_0 = al0
        self.k_1 = al1
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0
        else:
            values[0] = self.k_1
            
# matrix material properties
E1 = 21e3 
nu1 = 0.3

# reinforcement material properties
E2 = 2100e3 
nu2 = 0.25

E = al(materials, E1, E2, degree = 0)
nu = al(materials, nu1, nu2, degree = 0)
File('mat_p2.pvd') << materials
