import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

# Load arrays ✓
header = 'Apr_2023_cmp_heat_maps/True gn/IMPJ_20000_'
g1_two_in = np.load(header + 'g1_hm_matrix_two_inputs.npy')
g2_two_in = np.load(header + 'g2_hm_matrix_two_inputs.npy')
g3_two_in = np.load(header + 'g3_hm_matrix_two_inputs.npy')

g1_three_in = np.load(header + 'g1_hm_matrix_three_inputs.npy')
g2_three_in = np.load(header + 'g2_hm_matrix_three_inputs.npy')
g3_three_in = np.load(header + 'g3_hm_matrix_three_inputs.npy')


# Calculate percentage of conflicting instances in diagonal
def calc_diag_ci_perc(array):
    array_sum = np.sum(array)
    diag_sum = 0
    for j in range(0, array.shape[0]):
        for i in range(0, array.shape[1]):
            if i == j:
                if i == 0:
                    continue
                diag_sum += array[j][i]

    diag_perc = (diag_sum/array_sum)*100
    return diag_perc


g1_diag_perc = calc_diag_ci_perc(g1_two_in)
g2_diag_perc = calc_diag_ci_perc(g2_two_in)
g3_diag_perc = calc_diag_ci_perc(g3_two_in)
print("finish")


# Sum conflicting instances and calculate percentage improvement ✓
def calc_improvement(two_in_array, three_in_array):
    two_in_sum = np.sum(two_in_array)
    three_in_sum = np.sum(three_in_array)
    perc = (1 - (three_in_sum/two_in_sum))*100
    return two_in_sum, three_in_sum, perc


# g1_two_in_sum, g1_three_in_sum, g1_perc = calc_improvement(g1_two_in, g1_three_in)
# g2_two_in_sum, g2_three_in_sum, g2_perc = calc_improvement(g2_two_in, g2_three_in)
# g3_two_in_sum, g3_three_in_sum, g3_perc = calc_improvement(g3_two_in, g3_three_in)
# print('finish')


# Matrix subtraction ✓
g1_diff = np.subtract(g1_two_in, g1_three_in)
g2_diff = np.subtract(g2_two_in, g2_three_in)
g3_diff = np.subtract(g3_two_in, g3_three_in)


# Create colour arrays ✓
def create_colour_array(diff_array):

    # Record which elements are negative
    neg_zero_pos_array = np.full_like(diff_array, np.nan)
    for j in range(diff_array.shape[0]):
        for i in range(diff_array.shape[1]):
            if diff_array[j][i] < 0:
                neg_zero_pos_array[j][i] = -1
            elif diff_array[j][i] == 0:
                neg_zero_pos_array[j][i] = 0
            else:
                neg_zero_pos_array[j][i] = 1

    # Take absolute value of all elements
    colour_array = np.absolute(diff_array)
    for j in range(colour_array.shape[0]):
        for i in range(colour_array.shape[1]):
            if colour_array[j][i] == 0:
                colour_array[j][i] += 1
    colour_array = np.log10(colour_array)

    # Convert negative elements back to negative
    for j in range(colour_array.shape[0]):
        for i in range(colour_array.shape[1]):
            if neg_zero_pos_array[j][i] == -1:
                colour_array[j][i] = colour_array[j][i] * -1

    return colour_array


# Create heat maps ✓
def plot_hm_subplots(diff1, colour1, diff2, colour2, diff3, colour3):

    # Function for plotting one heat map subplot
    def plot_hm_subplot(diff_array, colour_array, ax):
        cmap = sb.diverging_palette(10, 150, s=100, as_cmap=True)
        # hm = sb.heatmap(colour_array, ax=ax, cmap=cmap, annot=diff_array,
        #                 annot_kws={'rotation': 45, 'fontsize': 8}, linewidths=0.5,
        #                 cbar_kws={'location': 'bottom', 'ticks': np.linspace(-3.5, 3.5, 15)},
        #                 vmin=-3.5, vmax=3.5, linecolor='black')

        hm = sb.heatmap(colour_array, ax=ax, cmap=cmap, linewidths=0.5,
                        cbar_kws={'location': 'bottom', 'ticks': np.linspace(-3.5, 3.5, 15)},
                        vmin=-3.5, vmax=3.5, linecolor='black')

        hm.set(xticklabels=[], yticklabels=[])
        hm.tick_params(bottom=False, left=False)

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plot_hm_subplot(diff1, colour1, ax1)
    plot_hm_subplot(diff2, colour2, ax2)
    plot_hm_subplot(diff3, colour3, ax3)
    plt.show()


g1_colour = create_colour_array(g1_diff)
g2_colour = create_colour_array(g2_diff)
g3_colour = create_colour_array(g3_diff)
plot_hm_subplots(g1_diff, g1_colour, g2_diff, g2_colour, g3_diff, g3_colour)
print('finish')
