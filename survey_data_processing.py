import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data_for_frequency_histogram(data, ax, lims):
    data = data[data != 0] - 1  # exclude entries with 0 (no response) and shift the values of the responses by 1
    n = len(data)  # number of all responses
    levels = [0, 1, 2, 3, 4]  # responses: 0 - no influence, ..., 4 - great influence
    frequencies = []  # frequencies of responses on a given level
    for level in levels:
        frequencies.append((data == level).sum() / n)

    mean_value = np.mean(data)  # mean value of the responses
    std_value = np.std(data, ddof=1)  # (sample) standard diviation of the responses

    # plot
    ax.bar(levels, frequencies)
    ax.set_ylim(lims)
    ax.set_xticks(levels)
    ax.tick_params(direction='in')
    ax.plot([mean_value, mean_value], lims, 'r')

    return levels, frequencies, mean_value, std_value, n


df = pd.read_excel("survey_data.xlsx")
labels = list(df.columns)

data = df.to_numpy()
fig, axs = plt.subplots(3, 3)

mean_susceptibilities = [0] * 9  # mean values of corresponding susceptibilities (names stored in labels)
std_susceptibilities = [
                           0] * 9  # (sample) standard diviations of corresponding susceptibilities (names stored in labels)
ns = [0] * 9  # number of responses

levels, frequencies, mean_susceptibilities[0], std_susceptibilities[0], ns[0] = get_data_for_frequency_histogram(
    data[:, 0], axs[0, 0], [0, 0.3])
levels, frequencies, mean_susceptibilities[3], std_susceptibilities[3], ns[3] = get_data_for_frequency_histogram(
    data[:, 3], axs[0, 1], [0, 0.3])
levels, frequencies, mean_susceptibilities[6], std_susceptibilities[6], ns[6] = get_data_for_frequency_histogram(
    data[:, 6], axs[0, 2], [0, 0.3])
axs[0, 0].set_ylabel('media')

axs[0, 0].set_title('HEV')
axs[0, 1].set_title('PHEV')
axs[0, 2].set_title('BEV')

levels, frequencies, mean_susceptibilities[2], std_susceptibilities[2], ns[2] = get_data_for_frequency_histogram(
    data[:, 2], axs[1, 0], [0, 0.5])
levels, frequencies, mean_susceptibilities[5], std_susceptibilities[5], ns[5] = get_data_for_frequency_histogram(
    data[:, 5], axs[1, 1], [0, 0.5])
levels, frequencies, mean_susceptibilities[8], std_susceptibilities[8], ns[8] = get_data_for_frequency_histogram(
    data[:, 8], axs[1, 2], [0, 0.5])
axs[1, 0].set_ylabel('local')

levels, frequencies, mean_susceptibilities[1], std_susceptibilities[1], ns[1] = get_data_for_frequency_histogram(
    data[:, 1], axs[2, 0], [0, 0.3])
levels, frequencies, mean_susceptibilities[4], std_susceptibilities[4], ns[4] = get_data_for_frequency_histogram(
    data[:, 4], axs[2, 1], [0, 0.3])
levels, frequencies, mean_susceptibilities[7], std_susceptibilities[7], ns[7] = get_data_for_frequency_histogram(
    data[:, 7], axs[2, 2], [0, 0.3])
axs[2, 0].set_ylabel('global')

fig.tight_layout()

for i in range(9):
    print(
        f'{labels[i]}: mean: {mean_susceptibilities[i]:.2f} std:{std_susceptibilities[i]:.2f} (based on {ns[i]} responses)')

plt.show()

driving_ranges = data[:,9]
driving_ranges = driving_ranges[driving_ranges != 0]  # exclude entries with 0 (no response)

# plot
fig, axs = plt.subplots()
axs.boxplot(driving_ranges)
axs.tick_params(direction='in')
axs.set_yscale('log')
axs.set_ylim([1e2, 1.6e6])
#axs.set_yticks([1e2,1e3,1e4,1e5,1e6])

# summary statistics
print('Summary statistics for:', labels[9])
print('number of responses:', len(driving_ranges))
print(f'mean: {np.mean(driving_ranges):.0f}')
print(f'std: {np.std(driving_ranges, ddof=1):.0f}')
print('min:', np.min(driving_ranges))
print('25:', np.percentile(driving_ranges,25))
print('50:', np.percentile(driving_ranges,50))
print('75:', np.percentile(driving_ranges,75))
print('max:', np.max(driving_ranges))

plt.show()

q75, q25 = np.percentile(driving_ranges, [75, 25])  # get 75 and 25 pwrcentiles to carculate IQR (Inter Quartile Range)
iqr = q75 - q25  # Inter Quartile Range (IQR)

distance_limit = q75 + (1.5 * iqr)  # limit above which all distances are consider to be distant journies (for which dp = 1)

# driving pattern calculations
dp = driving_ranges / distance_limit
print(f'Number of outliers: {np.sum(dp > 1)}')
dp[dp > 1] = 1  # all distances longer that distance_limit are considered as distant journies (dp = 1)

mean_dp = dp.mean()  # mean driving patter
std_dp = np.std(dp, ddof=1)

# plot
fig, axs = plt.subplots()
axs.hist(dp, bins=20)
axs.set_ylim([0, 140])
axs.plot([mean_dp, mean_dp], [0, 140], 'r')
axs.tick_params(direction='in')
axs.set_title('Histogram of driving patterns')

# summary statistics
print('Summary statistics')
print('number of responses:', len(dp))
print(f'mean: {np.mean(dp):.3f}')
print(f'std: {np.std(dp, ddof=1):.3f}')
print(f'min: {np.min(dp):.3f}')
print(f'25: {np.percentile(dp,25):.3f}')
print(f'50: {np.percentile(dp,50):.3f}')
print(f'75: {np.percentile(dp,75):.3f}')
print(f'max: {np.max(dp):.3f}')

plt.show()

# save driving patterns
# np.savetxt('dp.txt', dp, delimiter='\n')

fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 3]})

axs[0].boxplot(driving_ranges)
axs[0].tick_params(direction='in')
axs[0].set_yscale('log')
axs[0].set_ylim([1e2, 1.6e6])

axs[1].hist(dp, bins=20)
axs[1].set_ylim([0, 140])
axs[1].set_xlim([0, 1])
axs[1].plot([mean_dp, mean_dp], [0, 140], 'r')
axs[1].tick_params(direction='in')

plt.show()