import binascii
from tkinter.filedialog import askopenfilename
from math import log10
import seaborn as sns;
sns.set()

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

plt.style.use('ggplot')

print('Embedding an image into a container')


def input_secret():
    print('Choose the file with the secret')
    secret_file_name = askopenfilename()
    fin = open(secret_file_name, "rb")
    secret = fin.read()
    fin.close()
    # convert each data byte to a 2-digit hex string
    hex_str = str(binascii.hexlify(secret))
    # create a string of bits from two-digit hexadecimal strings
    hex_list = []
    bin_list = []
    for ix in range(2, len(hex_str) - 1, 2):
        word = hex_str[ix] + hex_str[ix + 1]
        hex_list.append(word)
        bin_list_0 = bin(int(word, 16))[2:]
        bin_list_0 = bin_list_0.zfill(8)
        bin_list.append(bin_list_0)
    bin_str = "".join(bin_list)
    return bin_str, secret_file_name


def input_container():
    print('Choose the container file')
    container_img_puth = askopenfilename()
    # convert container to array
    container_array = np.asarray(Image.open(container_img_puth).convert('RGB'))
    return container_array


def input_number_of_lsb():
    # number of embedded bits
    while True:
        number_of_lsb = input('Enter the number of embedded bits (from 1 to 8): ')
        try:
            number_of_lsb = int(number_of_lsb)
            if number_of_lsb < 0 or number_of_lsb > 8:
                print('Integer from 0 to 8 must be entered (inclusive)')
            else:
                return int(number_of_lsb)
        except ValueError:
            print('Integer must be entered!')


def input_sensitivity():
    # sensitivity - minimum standard standard deviation.
    # embedding occurs if the standard deviation is greater than sensitivity
    while True:
        sensitivity = input('Enter sensitivity (minimum allowed standard deviation): ')
        try:
            sensitivity = float(sensitivity)
            return float(sensitivity)
        except ValueError:
            print('Must enter numeric value!')


def input_step():
    while True:
        step = input('Enter the sensitivity reduction step: ')
        try:
            step = float(step)
            return float(step)
        except ValueError:
            print('Must enter numeric value!')


# embed image into container
def encode_img(sensitivity, container_without_secret, bin_str):
    i = 0  # the number of secret bits embedded in the container
    rows, cols, ch = container_without_secret.shape
    modified_container = np.zeros([rows, cols, ch])
    container_with_secret = container_without_secret.copy()
    for x in range(DISTANCE_FROM_CENTER_TO_EDGE, rows - DISTANCE_FROM_CENTER_TO_EDGE):
        if i == len(bin_str): break
        for y in range(DISTANCE_FROM_CENTER_TO_EDGE, cols - DISTANCE_FROM_CENTER_TO_EDGE):
            if i == len(bin_str): break
            for z in range(0, ch):
                data = np.std(
                    container_with_secret[x - DISTANCE_FROM_CENTER_TO_EDGE:x + (DISTANCE_FROM_CENTER_TO_EDGE + 1),
                    y - DISTANCE_FROM_CENTER_TO_EDGE:y + (DISTANCE_FROM_CENTER_TO_EDGE + 1), z])
                if data >= sensitivity:
                    if len(bin_str) - NUMBER_OF_LSB >= i:
                        bin_a5 = format(container_with_secret[x, y, z], '#010b')
                        container_with_secret[x][y][z] = int(bin_a5[0:-NUMBER_OF_LSB] + bin_str[i:i + NUMBER_OF_LSB],
                                                             base=0)
                        modified_container[x][y] = [255, 255, 255]
                        i = i + NUMBER_OF_LSB
                    else:
                        ii = len(bin_str) - i
                        bin_a5 = format(container_with_secret[x, y, z], '#010b')
                        container_with_secret[x][y][z] = int(bin_a5[0:-ii] + bin_str[i:], base=0)
                        modified_container[x][y] = [255, 255, 255]
                        i = i + ii
                if i == len(bin_str): break
    return container_with_secret, modified_container, i


def encode_2_img(sensitivity, container_without_secret, bin_str, number_of_lsb):
    i = 0  # the number of secret bits embedded in the container
    rows, cols, ch = container_without_secret.shape
    container_with_secret = container_without_secret.copy()
    for x in range(DISTANCE_FROM_CENTER_TO_EDGE, rows - DISTANCE_FROM_CENTER_TO_EDGE):
        if i == len(bin_str): break
        for y in range(DISTANCE_FROM_CENTER_TO_EDGE, cols - DISTANCE_FROM_CENTER_TO_EDGE):
            if i == len(bin_str): break
            for z in range(0, ch):
                data = np.std(
                    container_with_secret[x - DISTANCE_FROM_CENTER_TO_EDGE:x + (DISTANCE_FROM_CENTER_TO_EDGE + 1),
                    y - DISTANCE_FROM_CENTER_TO_EDGE:y + (DISTANCE_FROM_CENTER_TO_EDGE + 1), z])
                if data >= sensitivity:
                    if len(bin_str) - number_of_lsb >= i:
                        bin_a5 = format(container_with_secret[x, y, z], '#010b')
                        container_with_secret[x][y][z] = int(bin_a5[0:-number_of_lsb] + bin_str[i:i + number_of_lsb],
                                                             base=0)
                        i = i + number_of_lsb
                    else:
                        ii = len(bin_str) - i
                        bin_a5 = format(container_with_secret[x, y, z], '#010b')
                        container_with_secret[x][y][z] = int(bin_a5[0:-ii] + bin_str[i:], base=0)
                        i = i + ii
                if i == len(bin_str): break
    return container_with_secret


def calculation_psnr(original, compressed):
    mse = (np.square(original - compressed)).mean(axis=None)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    calculation_psnr = 10 * log10(max_pixel ** 2 / mse)
    return calculation_psnr


def calculation_capacity(secret_file_name, container):
    # capacity calculation: (number of secret pixels) / (number of container pixels)
    secret = np.asarray(Image.open(secret_file_name).convert('RGB'))
    rows_1, cols_1, ch_1 = secret.shape
    rows_2, cols_2, ch_2 = container.shape
    number_of_pixels_secret = rows_1 * cols_1
    number_of_pixels_container = rows_2 * cols_2
    capacity = (number_of_pixels_secret / number_of_pixels_container) * 100
    return capacity


def calculation_payload(container, number_of_bit_secret):
    # Payload calculation: (number of bits in secret) / ((number of pixels in container) * (number of channels RGB)))
    rows, cols, ch = container.shape
    number_of_pixels_and_channels_container = rows * cols * ch
    payload = number_of_bit_secret / number_of_pixels_and_channels_container
    return payload


# save the images
def save_output(container_and_secret, modified_container):
    im1 = Image.fromarray(np.uint8(container_and_secret))
    im1.save("stego.png")
    im2 = Image.fromarray(np.uint8(modified_container))
    im2.save("modified_container.png")
    print('\nContainer with built-in secret saved in file stego.png')
    print(
        'Changed container pixels are highlighted in white on the saved image modified_container.png')


def kpi_output(container, container_and_secret, secret, secret_file_name):
    mse = (np.square(container - container_and_secret)).mean(axis=None)
    print('mse (mean squared error):', mse)
    psnr = calculation_psnr(container, container_and_secret)
    print("psnr (peak signal-to-noise ratio):", psnr)
    container_1 = container.ravel()
    container_and_secret_1 = container_and_secret.ravel()
    pearson_r, p_value = stats.pearsonr(container_1, container_and_secret_1)
    print('Pearson correlation coefficient:', pearson_r)
    capacity = calculation_capacity(secret_file_name, container)
    payload = calculation_payload(container, len(secret))
    print("Capacity (%):", capacity)
    print('Payload:', payload)
    return mse, psnr, pearson_r


def chart_sensitivity(sensitivity, container_without_secret, bin_str, mse_0,
                      psnr_0,
                      pearson_r_0):
    sensitivitys = np.array([sensitivity])
    psnrs = np.array([psnr_0])
    mses = np.array([mse_0])
    pearsons_r = np.array([pearson_r_0])
    container_1 = container_without_secret.ravel()
    number_of_lsb = NUMBER_OF_LSB
    while sensitivity > 0:
        sensitivity = sensitivity - 0.1
        sensitivitys = np.append(sensitivitys, sensitivity)
        container_and_secret = encode_2_img(sensitivity, container_without_secret, bin_str, number_of_lsb)
        psnr_i = calculation_psnr(container, container_and_secret)
        psnrs = np.append(psnrs, psnr_i)
        mse_i = (np.square(container_without_secret - container_and_secret)).mean(axis=None)
        mses = np.append(mses, mse_i)
        container_and_secret_i = container_and_secret.ravel()
        pearson_r_0, p_value = stats.pearsonr(container_1, container_and_secret_i)
        pearsons_r = np.append(pearsons_r, pearson_r_0)
    plt.style.use('seaborn-whitegrid')
    fig_mses = plt.figure()
    ax = plt.axes()
    plt.plot(sensitivitys, mses, '-ok')
    plt.xlabel("Sensitivity")
    plt.ylabel("MSE")
    fig_mses.savefig('fig_mses.png')

    fig_psnrs = plt.figure()
    ax = plt.axes()
    plt.plot(sensitivitys, psnrs, '-ok')
    plt.xlabel("Sensitivity")
    plt.ylabel("PSNR")
    fig_psnrs.savefig('fig_psnrs.png')

    fig_pearsons_r = plt.figure()
    ax = plt.axes()
    plt.plot(sensitivitys, pearsons_r, '-ok')
    plt.xlabel("Sensitivity")
    plt.ylabel("Pearson correlation coefficient")
    fig_pearsons_r.savefig('fig_pearsons_r.png')
    print('\nMSE, PSNR and Pearson correlation coefficient graphs are ready')


def chart_number_of_lsb_pearsons_r(sensitivity, container_without_secret, bin_str, pearson_r_0):
    number_of_lsbs = np.array([NUMBER_OF_LSB])
    pearsons_r = np.array([pearson_r_0])
    number_of_lsb_i = NUMBER_OF_LSB
    container_1 = container_without_secret.ravel()
    while number_of_lsb_i < 8:
        number_of_lsb_i = number_of_lsb_i + 1
        number_of_lsbs = np.append(number_of_lsbs, number_of_lsb_i)
        container_and_secret = encode_2_img(sensitivity, container_without_secret, bin_str, number_of_lsb_i)
        container_and_secret_i = container_and_secret.ravel()
        pearson_r_0, p_value = stats.pearsonr(container_1, container_and_secret_i)
        pearsons_r = np.append(pearsons_r, pearson_r_0)
    fig_lsb_pearsons_r = plt.figure()
    ax = plt.axes()
    plt.plot(number_of_lsbs, pearsons_r, '-ok', alpha=0.7, lw=1, mec='k', mew=1, ms=2.5)
    plt.xlabel("Number of bits")
    plt.ylabel("Pearson correlation coefficient")
    fig_lsb_pearsons_r.savefig('fig_bit_pearson.png')
    print('\nThe chart of the Pearson correlation coefficient dependence on the number of bits to be replaced is ready')


def chart_3d(sensitivity_0, container_without_secret, bin_str):
    number_of_lsb_i = NUMBER_OF_LSB
    number_of_lsbs = np.array([], dtype=int)
    container_1 = container_without_secret.ravel()
    n = int(9 - NUMBER_OF_LSB)
    m = int(sensitivity_0 + 1)
    pearsons_r = np.zeros((n, m), dtype=float)
    for lsb_i in range(n):
        sensitivity_i = sensitivity_0
        sensitivitys = np.array([], dtype=int)
        number_of_lsbs = np.append(number_of_lsbs, number_of_lsb_i)
        for sens_j in range(m):
            sensitivitys = np.append(sensitivitys, sensitivity_i)
            container_and_secret = encode_2_img(sensitivity_i, container_without_secret, bin_str, number_of_lsb_i)
            container_and_secret_i = container_and_secret.ravel()
            pearson_r_0, p_value = stats.pearsonr(container_1, container_and_secret_i)
            pearsons_r[lsb_i][sens_j] = pearson_r_0
            sensitivity_i -= 1
        number_of_lsb_i += 1
    sensitivitys, number_of_lsbs = np.meshgrid(sensitivitys, number_of_lsbs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot a basic wireframe.
    ax.plot_wireframe(number_of_lsbs, sensitivitys, pearsons_r, rstride=2, cstride=2)

    # ax = Axes3D(fig)
    # ax.plot_surface(number_of_lsbs, sensitivitys, pearsons_r, rstride=1, cstride=1)
    # # Plot the surface.
    # surf = ax.plot_surface(number_of_lsbs, sensitivitys, pearsons_r, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # # Customize the z axis.
    # ax.set_zlim(0.9, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax.set_xlabel('Number of bits')
    # ax.set_ylabel('Sensitivity')
    # ax.set_zlabel('Pearson correlation coefficient')
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    print('\n3D chart is ready')
    plt.show()


DISTANCE_FROM_CENTER_TO_EDGE = 1
secret, secret_file_name = input_secret()
container = input_container()
NUMBER_OF_LSB = input_number_of_lsb()
sensitivity = input_sensitivity()
container_and_secret, modified_container, i = encode_img(sensitivity, container, secret)

if i < len(secret):
    # i - количество бит секрета, встроенных в контейнер
    if sensitivity > 0:
        print('Need to reduce sensitivity')
        step = input_step()
        while i < len(secret) and sensitivity >= 0:
            sensitivity = sensitivity - step
            container_and_secret, modified_container, i = encode_img(sensitivity, container, secret)
        if i == len(secret):
            save_output(container_and_secret, modified_container)
            mse, psnr, pearson_r = kpi_output(container, container_and_secret, secret, secret_file_name)
            print('Sensitivity reduced to', sensitivity)
            chart_sensitivity(sensitivity, container, secret, mse, psnr, pearson_r)
            # chart_number_of_lsb_pearsons_r(sensitivity, container, secret, pearson_r)
            # chart_3d(sensitivity, container, secret)
        else:
            print('\nUnable to embed. Container is too small')
    else:
        print('\nUnable to embed. Container is too small')
else:
    save_output(container_and_secret, modified_container)
    mse, psnr, pearson_r = kpi_output(container, container_and_secret, secret, secret_file_name)
    chart_sensitivity(sensitivity, container, secret, mse, psnr, pearson_r)
    # chart_number_of_lsb_pearsons_r(sensitivity, container, secret, pearson_r)
    # chart_3d(sensitivity, container, secret)
