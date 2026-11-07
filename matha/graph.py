import numpy as np

def _apply_custom_style(ax, title, xlabel, ylabel, grid, g_style, g_alpha, axh_c, axh_w, axv_c, axv_w):
    """Applies labels, grids, and axis lines consistently."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, linestyle=g_style, alpha=g_alpha)
    
    # 2D-only styling (avoids crashing 3D plots)
    if not hasattr(ax, 'zaxis'):
        if axh_c: ax.axhline(0, color=axh_c, linewidth=axh_w)
        if axv_c: ax.axvline(0, color=axv_c, linewidth=axv_w)

# basic plots

def histogram(data, bins=10, title="Histogram", color='tab:blue', edgecolor='black', 
              xlabel="Value", ylabel="Frequency", grid=True, g_style='-', g_alpha=0.3):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor)
    _apply_custom_style(plt.gca(), title, xlabel, ylabel, grid, g_style, g_alpha, None, 0, None, 0)
    plt.show()

def boxplot(data, title="Boxplot", ylabel="Values", color='tab:cyan', grid=True, g_style='-', g_alpha=0.3):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    bp = plt.boxplot(data, vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
    _apply_custom_style(plt.gca(), title, "", ylabel, grid, g_style, g_alpha, None, 0, None, 0)
    plt.show()

def violin_plot(data, title="Violin Plot", ylabel="Values", color='tab:purple', grid=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    parts = plt.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    _apply_custom_style(plt.gca(), title, "", ylabel, grid, '-', 0.2, None, 0, None, 0)
    plt.show()

def density_plot(data, title="Density Plot", xlabel="Value", ylabel="Density", 
                 color='tab:red', fill_alpha=0.3, grid=True, g_style='-', g_alpha=0.3):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    data = np.array(data)
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 500)
    y_vals = kde(x_vals)
    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, y_vals, color=color)
    plt.fill_between(x_vals, y_vals, alpha=fill_alpha, color=color)
    _apply_custom_style(plt.gca(), title, xlabel, ylabel, grid, g_style, g_alpha, None, 0, None, 0)
    plt.show()

def hexbin_plot(x, y, gridsize=20, title="Hexbin Density Plot", 
                xlabel="X", ylabel="Y", cmap='viridis', grid=True, g_alpha=0.1):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1)
    plt.colorbar(label='Count in Bin')
    _apply_custom_style(plt.gca(), title, xlabel, ylabel, grid, '-', g_alpha, None, 0, None, 0)
    plt.show()

def scatter_plot(x, y, title="Scatter Plot", xlabel="X", ylabel="Y", 
                 color='tab:blue', alpha=0.7, grid=True, g_style='--', g_alpha=0.3):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, color=color, alpha=alpha)
    _apply_custom_style(plt.gca(), title, xlabel, ylabel, grid, g_style, g_alpha, None, 0, None, 0)
    plt.show()

def scatter_3d(x, y, z, title="3D Scatter Plot", xlabel="X", ylabel="Y", zlabel="Z", cmap='plasma'):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap=cmap)
    plt.colorbar(sc, label=zlabel)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()

def scatter_matrix(data, labels=None, title="Scatter Matrix", color='tab:blue', 
                   grid=True, g_style=':', g_alpha=0.2):
    import matplotlib.pyplot as plt
    data = np.array(data)
    if data.shape[0] > data.shape[1]: 
        data = data.T
    dims = data.shape[0]
    fig, axes = plt.subplots(dims, dims, figsize=(3 * dims, 3 * dims))
    for i in range(dims):
        for j in range(dims):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[i], color=color, edgecolor='black', alpha=0.7)
            else:
                ax.scatter(data[j], data[i], s=15, color=color, alpha=0.5)
            if labels:
                if j == 0: ax.set_ylabel(labels[i])
                if i == dims - 1: ax.set_xlabel(labels[j])
            if grid: ax.grid(True, linestyle=g_style, alpha=g_alpha)
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

# math and solver plots

def plot_function(f, x_min, x_max, resolution=500, title="Function Plot", xlabel="x", ylabel="f(x)",
                  line_color='tab:blue', lw=2, grid=True, g_style='-', g_alpha=0.3,
                  axh_c='black', axh_w=0.8, axv_c='black', axv_w=0.8):
    import matplotlib.pyplot as plt
    x = np.linspace(x_min, x_max, resolution)
    try: y = f(x)
    except: y = np.array([f(val) for val in x])
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color=line_color, linewidth=lw)
    _apply_custom_style(plt.gca(), title, xlabel, ylabel, grid, g_style, g_alpha, axh_c, axh_w, axv_c, axv_w)
    plt.show()

def plot_multiple_functions(functions, x_min, x_max, resolution=500, title="Multi-Function Plot",
                            grid=True, g_style='-', g_alpha=0.2):
    import matplotlib.pyplot as plt
    x = np.linspace(x_min, x_max, resolution)
    plt.figure(figsize=(8, 5))
    for i, f in enumerate(functions):
        try: y = f(x)
        except: y = np.array([f(val) for val in x])
        plt.plot(x, y, label=f"f{i+1}(x)")
    _apply_custom_style(plt.gca(), title, "x", "y", grid, g_style, g_alpha, 'black', 0.5, 'black', 0.5)
    plt.legend()
    plt.show()

def plot_points(points: list[tuple], draw_line=True, title="Data Point Plot", 
                xlabel="X", ylabel="Y", color='tab:blue', grid=True, g_alpha=0.4):
    import matplotlib.pyplot as plt
    if not points: return
    xs, ys = zip(*points)
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, color=color, zorder=3)
    if draw_line: plt.plot(xs, ys, color=color, linestyle='--', alpha=0.6)
    _apply_custom_style(plt.gca(), title, xlabel, ylabel, grid, '-', g_alpha, 'black', 0.5, 'black', 0.5)
    plt.show()

def plot_root(func, root, x_range=(-10, 10), title="Root Visualization", color="red",
              grid=True, g_style='--', g_alpha=0.5, axh_c='black', axh_w=0.8):
    import matplotlib.pyplot as plt
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.array([func(i) for i in x])
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.scatter([root], [0], color=color, s=100, zorder=5, label=f"Root: {root:.4f}")
    _apply_custom_style(plt.gca(), title, "x", "f(x)", grid, g_style, g_alpha, axh_c, axh_w, None, 0)
    plt.legend()
    plt.show()

def plot_regression(x_data, y_data, slope, intercept, title="Linear Regression", 
                    line_color='red', scatter_color='blue', grid=True, g_style=':', g_alpha=0.5):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, color=scatter_color, alpha=0.6)
    x_line = np.array([min(x_data), max(x_data)])
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color=line_color, linewidth=2, label='Fit')
    _apply_custom_style(plt.gca(), title, "X", "Y", grid, g_style, g_alpha, None, 0, None, 0)
    plt.legend()
    plt.show()

def plot_ode(t, y, title="ODE Solution", xlabel="Time (t)", ylabel="State (y)", grid=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    if y.ndim > 1:
        for i in range(y.shape[0]): plt.plot(t, y[i], label=f'State {i+1}')
    else: plt.plot(t, y)
    _apply_custom_style(plt.gca(), title, xlabel, ylabel, grid, '-', 0.3, None, 0, None, 0)
    if y.ndim > 1: plt.legend()
    plt.show()

# num plots for complex nums
def plot_complex_2d(numbers, title="Complex Plane", grid=True, g_alpha=0.2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 7))
    reals, imags = [z.real for z in numbers], [z.imag for z in numbers]
    plt.scatter(reals, imags, color='red', zorder=5)
    for r, i in zip(reals, imags):
        plt.quiver(0, 0, r, i, angles='xy', scale_units='xy', scale=1, alpha=0.3)
    _apply_custom_style(plt.gca(), title, "Real", "Imaginary", grid, '-', g_alpha, 'black', 1, 'black', 1)
    plt.show()

def plot_complex_surface(func, x_range=(-2, 2), y_range=(-2, 2), res=100, cmap='viridis'):
    import matplotlib.pyplot as plt
    x, y = np.linspace(x_range[0], x_range[1], res), np.linspace(y_range[0], y_range[1], res)
    X, Y = np.meshgrid(x, y)
    Z = np.abs(func(X + 1j*Y))
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
    ax.set_title(f"Surface of {getattr(func, '__name__', 'Function')}")
    fig.colorbar(surf)
    plt.show()
