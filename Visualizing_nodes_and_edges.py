from devsim import *  # Import DEVSIM functions
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Define device and region names
# =============================================================================
device = "MyDevice"
region = "silic"

# =============================================================================
# 2. Create a simple 2D rectangular mesh
# =============================================================================
#    We'll make a 5×5 grid from (0,0) to (1,1) with 6 lines each, i.e. 25 square cells
create_2d_mesh(mesh="dio")  

# Add vertical mesh lines (x = constant)
for x in np.linspace(0.0, 1.0, 6):  
    # ps, ns = positive and negative spacing contributions (uniform here)
    add_2d_mesh_line(mesh="dio", dir="x", pos=x, ps=0.1, ns=0.1)

# Add horizontal mesh lines (y = constant)
for y in np.linspace(0.0, 1.0, 6):
    add_2d_mesh_line(mesh="dio", dir="y", pos=y, ps=0.1, ns=0.1)

# =============================================================================
# 3. Define a single silicon region covering the entire mesh
# =============================================================================
add_2d_region(
    mesh="dio",
    material="Silicon",
    region=region,
    yl=0.0, yh=1.0,  # y lower/upper bounds
    xl=0.0, xh=1.0   # x lower/upper bounds
)

# =============================================================================
# 4. Add two contacts (bottom and top) for completeness (no bias applied here)
# =============================================================================
add_2d_contact(
    mesh="dio", name="bot", region=region,
    material="metal", yl=0.0, yh=0.0, xl=0.0, xh=1.0
)
add_2d_contact(
    mesh="dio", name="top", region=region,
    material="metal", yl=1.0, yh=1.0, xl=0.0, xh=1.0
)

# Finalize the mesh and create the device in DEVSIM
finalize_mesh(mesh="dio")
create_device(mesh="dio", device=device)

# =============================================================================
# 5. Extract node coordinates
# =============================================================================
#    DEVSIM stores node coordinates in node models "x" and "y"
x_nodes = np.array(get_node_model_values(device=device, region=region, name="x"))
y_nodes = np.array(get_node_model_values(device=device, region=region, name="y"))

# =============================================================================
# 6. Extract edge endpoints
# =============================================================================
#    For each edge, DEVSIM provides x@n0, y@n0 (start) and x@n1, y@n1 (end)
#    First, create the edge models from node models

edge_from_node_model(device=device, region=region, node_model="x")
edge_from_node_model(device=device, region=region, node_model="y")

#x_n0[i], y_n0[i] : Represent the first node of the i_th edge created.
#x_n1[i], y_n1[i] : Represent the second node of the i_th edge created.

x_n0 = np.array(get_edge_model_values(device=device, region=region, name="x@n0"))
y_n0 = np.array(get_edge_model_values(device=device, region=region, name="y@n0"))
x_n1 = np.array(get_edge_model_values(device=device, region=region, name="x@n1"))
y_n1 = np.array(get_edge_model_values(device=device, region=region, name="y@n1"))

# =============================================================================
# 7. Calculate edge direction vectors and midpoints
# =============================================================================
# Direction vector for each edge (from n0 to n1)
dx = x_n1 - x_n0
dy = y_n1 - y_n0

# Edge midpoints for arrow placement
x_mid = 0.5 * (x_n0 + x_n1)
y_mid = 0.5 * (y_n0 + y_n1)

# Edge lengths for normalization
edge_length = np.sqrt(dx**2 + dy**2)
# Avoid division by zero for degenerate edges
edge_length[edge_length == 0] = 1.0

# Normalize direction vectors
dx_norm = dx / edge_length
dy_norm = dy / edge_length

# =============================================================================
# 8. Plot nodes, edges, and edge directions using Matplotlib
# =============================================================================
plt.figure(figsize=(12, 10))

# Plot edges: draw a line between each edge's endpoints
for i, (x0, y0, x1, y1) in enumerate(zip(x_n0, y_n0, x_n1, y_n1)):
    plt.plot([x0, x1], [y0, y1], color='lightgray', linewidth=1.5, alpha=0.8)

# Plot edge directions using quiver (arrows at edge midpoints)
# Scale arrows to be visible but not overwhelming
arrow_scale = 0.08  # Adjust this to make arrows larger or smaller
plt.quiver(x_n0, y_n0, 
           dx_norm * arrow_scale, dy_norm * arrow_scale, 
           angles='xy', scale_units='xy', scale=1, 
           color='blue', width=0.003, alpha=0.7, 
           label='Edge Direction (n0→n1)')

# Plot nodes: scatter points at each node
plt.scatter(x_nodes, y_nodes, color='red', s=40, label='Mesh Nodes', zorder=5)

# =============================================================================
# 9. Label selected nodes and edges to avoid clutter
# =============================================================================
# Label some nodes (every 3rd node to show ordering without clutter)
for idx, (xn, yn) in enumerate(zip(x_nodes, y_nodes)):
    if idx % 3 == 0:  # label every 3rd node to avoid clutter
        plt.text(xn + 0.02, yn + 0.02, f'N{idx}', 
                color='darkred', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

# Label some edges (every 5th edge to show indexing)
for idx, (xm, ym) in enumerate(zip(x_mid, y_mid)):
    if idx % 5 == 0:  # label every 5th edge to avoid clutter
        plt.text(xm + 0.01, ym - 0.03, f'E{idx}', 
                color='darkblue', fontsize=8, style='italic',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))

# =============================================================================
# 10. Formatting and display
# =============================================================================
plt.title("DEVSIM 2D Mesh Visualization:\nNodes (red), Edges (gray), and Edge Directions (blue arrows)", 
          fontsize=14, fontweight='bold')
plt.xlabel("x-coordinate", fontsize=12)
plt.ylabel("y-coordinate", fontsize=12)
plt.axis('equal')   # Ensure scaling is equal on both axes
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=10)

# Add text box with mesh information
textstr = f'Nodes: {len(x_nodes)}\nEdges: {len(x_n0)}\nGrid: 6×6 lines'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# =============================================================================
# 11. Print mesh statistics and edge examples
# =============================================================================
print(f"Mesh Statistics:")
print(f"Total nodes: {len(x_nodes)}")
print(f"Total edges: {len(x_n0)}")
print(f"Node coordinate range: x=[{x_nodes.min():.1f}, {x_nodes.max():.1f}], y=[{y_nodes.min():.1f}, {y_nodes.max():.1f}]")

print(f"\nFirst 5 edges (direction from n0 to n1):")
for i in range(len(x_n0)):
    print(f"Edge {i}: ({x_n0[i]:.2f}, {y_n0[i]:.2f}) → ({x_n1[i]:.2f}, {y_n1[i]:.2f})")
    print(f"         Direction vector: ({dx[i]:.2f}, {dy[i]:.2f})")

