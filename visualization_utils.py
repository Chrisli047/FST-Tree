import numpy as np
from matplotlib import pyplot as plt


def plot_linear_equations(records, x_range=(-1, 1)):
    """
    Plots multiple 2D linear equations of the form ax1 + bx2 + c = 0.

    Parameters:
        records (list of tuples): Each tuple represents coefficients (a, b, c) of a linear equation.
        x_range (tuple): Range of x1 and x2, default is (0, 10).
    """
    plt.figure(figsize=(10, 8))
    x1 = np.linspace(x_range[0], x_range[1], 100)  # Generate x1 values within the range

    for record in records:
        a, b, c = record
        if b != 0:
            # Calculate x2 based on the equation
            x2 = -(a * x1 + c) / b
            x2 = np.clip(x2, x_range[0], x_range[1])  # Clip x2 to the defined range
            plt.plot(x1, x2, label=f'{a}x1 + {b}x2 + {c} = 0', linewidth=2)
        else:
            # Handle vertical lines where b = 0
            x = -c / a if a != 0 else None
            if x is not None and x_range[0] <= x <= x_range[1]:
                plt.plot([x, x], x_range, label=f'{a}x1 + {b}x2 + {c} = 0', linewidth=2)

    # Configure the plot
    plt.xlim(x_range)
    plt.ylim(x_range)
    plt.xlabel('x1', fontsize=20)
    plt.ylabel('x2', fontsize=20)
    plt.title('Visualization of Linear Equations', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

