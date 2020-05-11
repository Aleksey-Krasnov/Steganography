import cv2
import numpy as np
import binascii
from math import log10, sqrt
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.style.use('ggplot')

print('Встраивание изображения в контейнер')


def input_secret():
    secret_file_name = input('Введите имя файла с секретом:')
    fin = open(secret_file_name, "rb")
    secret = fin.read()
    fin.close()
    # преобразовать каждый байт данных в 2-значную шестнадцатеричную строку
    hex_str = str(binascii.hexlify(secret))
    # создать строку битов из двухзначных шестнадцатеричных строк
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
    container_img_puth = input('Введите имя файла-контейнера:')
    # преобразовать container в массив
    container_array = cv2.imread(container_img_puth)
    return container_array


def input_number_of_lsb():
    # количество встраиваемых бит
    while True:
        number_of_lsb = input('Введите количество встраиваемых бит (от 1 до 8): ')
        try:
            number_of_lsb = int(number_of_lsb)
            if number_of_lsb < 0 or number_of_lsb > 8:
                print('Необходимо ввести целое число от 0 до 8 (включительно)')
            else:
                return int(number_of_lsb)
        except ValueError:
            print('Необходимо ввести целое число!')


def input_sensitivity():
    # sensitivity - минимально допустимое среднеквадратичное отклонение.
    # встраивание выполняется, если  среднеквадратичное отклонение больше, чем sensitivity
    while True:
        sensitivity = input('Введите чувствительность (минимально допустимое среднеквадратичное отклонение): ')
        try:
            sensitivity = float(sensitivity)
            return float(sensitivity)
        except ValueError:
            print('Необходимо ввести числовое значение!')


def input_step():
    while True:
        step = input('Введите шаг уменьшения чувствительности: ')
        try:
            step = float(step)
            return float(step)
        except ValueError:
            print('Необходимо ввести числовое значение!')


# встроить изображение в контейнер
def encode_img(sensitivity, container_without_secret, bin_str, number_of_lsb):
    i = 0  # количество бит секрета, встроенных в контейнер
    rows, cols, ch = container_without_secret.shape
    modified_container = np.zeros([rows, cols, ch])
    container_with_secret = container_without_secret.copy()
    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            for z in range(0, ch):
                a1 = container_with_secret[x - 1][y - 1][z]
                a2 = container_with_secret[x - 1][y][z]
                a3 = container_with_secret[x - 1][y + 1][z]
                a4 = container_with_secret[x][y - 1][z]
                a5 = container_with_secret[x][y][z]
                a6 = container_with_secret[x][y + 1][z]
                a7 = container_with_secret[x + 1][y - 1][z]
                a8 = container_with_secret[x + 1][y][z]
                a9 = container_with_secret[x + 1][y + 1][z]
                data = [a1, a2, a3, a4, a5, a6, a7, a8, a9]
                if np.std(data) >= sensitivity:
                    if len(bin_str) - number_of_lsb >= i:
                        bin_a5 = format(a5, '#010b')
                        container_with_secret[x][y][z] = int(bin_a5[0:-number_of_lsb] + bin_str[i:i + number_of_lsb],
                                                             base=0)
                        modified_container[x][y] = [255, 255, 255]
                        i = i + number_of_lsb
                    else:
                        ii = len(bin_str) - i
                        bin_a5 = format(a5, '#010b')
                        container_with_secret[x][y][z] = int(bin_a5[0:-ii] + bin_str[i:], base=0)
                        modified_container[x][y] = [255, 255, 255]
                        i = i + ii
                if i == len(bin_str): break
            if i == len(bin_str): break
        if i == len(bin_str): break
    return container_with_secret, modified_container, i


def encode_2_img(sensitivity, container_without_secret, bin_str, number_of_lsb):
    i = 0  # количество бит секрета, встроенных в контейнер
    rows, cols, ch = container_without_secret.shape
    container_with_secret = container_without_secret.copy()
    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            for z in range(0, ch):
                a1 = container_with_secret[x - 1][y - 1][z]
                a2 = container_with_secret[x - 1][y][z]
                a3 = container_with_secret[x - 1][y + 1][z]
                a4 = container_with_secret[x][y - 1][z]
                a5 = container_with_secret[x][y][z]
                a6 = container_with_secret[x][y + 1][z]
                a7 = container_with_secret[x + 1][y - 1][z]
                a8 = container_with_secret[x + 1][y][z]
                a9 = container_with_secret[x + 1][y + 1][z]
                data = [a1, a2, a3, a4, a5, a6, a7, a8, a9]
                if np.std(data) >= sensitivity:
                    if len(bin_str) - number_of_lsb >= i:
                        bin_a5 = format(a5, '#010b')
                        container_with_secret[x][y][z] = int(bin_a5[0:-number_of_lsb] + bin_str[i:i + number_of_lsb],
                                                             base=0)
                        i = i + number_of_lsb
                    else:
                        ii = len(bin_str) - i
                        bin_a5 = format(a5, '#010b')
                        container_with_secret[x][y][z] = int(bin_a5[0:-ii] + bin_str[i:], base=0)
                        i = i + ii
                if i == len(bin_str): break
            if i == len(bin_str): break
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
    # расчёт ёмкости: (кол-во пикселей секрета) / (кол-во пикселей контейнера)
    secret = cv2.imread(secret_file_name)
    rows_1, cols_1, ch_1 = secret.shape
    rows_2, cols_2, ch_2 = container.shape
    number_of_pixels_secret = rows_1 * cols_1
    number_of_pixels_container = rows_2 * cols_2
    capacity = (number_of_pixels_secret / number_of_pixels_container) * 100
    return capacity


def calculation_payload(container, number_of_bit_secret):
    # расчёт полезной нагрузки: (кол-во бит в секрете) / (кол-во пикселей в контейнере)
    rows, cols, ch = container.shape
    number_of_pixels_container = rows * cols * ch
    payload = number_of_bit_secret / number_of_pixels_container
    return payload


# сохранить изображения
def save_output(container_and_secret, modified_container):
    cv2.imwrite("container_and_secret.png", container_and_secret, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite("modified_container.png", modified_container, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print('\nКонтейнер со встроенным секретом сохранён в файле container_and_secret.png')
    print(
        'Изменённые пиксели контейнера выделены белым цветом на изображении в сохранённом в файле modified_container.png')


def kpi_output(container, container_and_secret, secret, secret_file_name):
    mse = (np.square(container - container_and_secret)).mean(axis=None)
    print('mse (mean squared error):', mse)
    psnr = calculation_psnr(container, container_and_secret)
    print("psnr (peak signal-to-noise ratio):", psnr)
    container_1 = container.ravel()
    container_and_secret_1 = container_and_secret.ravel()
    pearson_r, p_value = stats.pearsonr(container_1, container_and_secret_1)
    print('pearson correlation coefficient:', pearson_r)
    capacity = calculation_capacity(secret_file_name, container)
    payload = calculation_payload(container, len(secret))
    print("Ёмкость (%):", capacity)
    print('Полезная нагрузка:', payload)
    return mse, psnr, pearson_r


def chart_sensitivity(sensitivity, container_without_secret, bin_str, number_of_lsb, mse_0,
                      psnr_0,
                      pearson_r_0):
    sensitivitys = np.array([sensitivity])
    psnrs = np.array([psnr_0])
    mses = np.array([mse_0])
    pearsons_r = np.array([pearson_r_0])
    container_1 = container_without_secret.ravel()
    while sensitivity > 0:
        sensitivity = sensitivity - 1
        sensitivitys = np.append(sensitivitys, sensitivity)
        container_and_secret = encode_2_img(sensitivity, container_without_secret, bin_str, number_of_lsb)
        psnr_i = calculation_psnr(container, container_and_secret)
        psnrs = np.append(psnrs, psnr_i)
        mse_i = (np.square(container_without_secret - container_and_secret)).mean(axis=None)
        mses = np.append(mses, mse_i)
        container_and_secret_i = container_and_secret.ravel()
        pearson_r_0, p_value = stats.pearsonr(container_1, container_and_secret_i)
        pearsons_r = np.append(pearsons_r, pearson_r_0)
    fig_mses = plt.figure()
    ax = plt.axes()
    plt.plot(sensitivitys, mses, '-ok')
    plt.xlabel("Чувствительность")
    plt.ylabel("MSE")
    fig_mses.savefig('fig_mses.png')

    fig_psnrs = plt.figure()
    ax = plt.axes()
    plt.plot(sensitivitys, psnrs, '-ok')
    plt.xlabel("Чувствительность")
    plt.ylabel("PSNR")
    fig_psnrs.savefig('fig_psnrs.png')

    fig_pearsons_r = plt.figure()
    ax = plt.axes()
    plt.plot(sensitivitys, pearsons_r, '-ok')
    plt.xlabel("Чувствительность")
    plt.ylabel("Коэффициент корреляции")
    fig_pearsons_r.savefig('fig_pearsons_r.png')
    print('\nГрафики готовы')


def chart_number_of_lsb_pearsons_r(sensitivity, container_without_secret, bin_str, number_of_lsb,
                                   pearson_r_0):
    number_of_lsbs = np.array([number_of_lsb])
    pearsons_r = np.array([pearson_r_0])
    container_1 = container_without_secret.ravel()
    while number_of_lsb < 8:
        number_of_lsb = number_of_lsb + 1
        number_of_lsbs = np.append(number_of_lsbs, number_of_lsb)
        container_and_secret = encode_2_img(sensitivity, container_without_secret, bin_str, number_of_lsb)
        container_and_secret_i = container_and_secret.ravel()
        pearson_r_0, p_value = stats.pearsonr(container_1, container_and_secret_i)
        pearsons_r = np.append(pearsons_r, pearson_r_0)
    fig_lsb_pearsons_r = plt.figure()
    ax = plt.axes()
    plt.plot(number_of_lsbs, pearsons_r, '-ok')
    plt.xlabel("Количество бит")
    plt.ylabel("Коэффициент корреляции")
    fig_lsb_pearsons_r.savefig('fig_bit_pearson.png')
    print('\nГрафик готов')


def chart_3d(sensitivity, container_without_secret, bin_str, number_of_lsb, pearson_r_0):
    sensitivitys = np.array([sensitivity])
    number_of_lsbs = np.array([number_of_lsb])
    container_1 = container_without_secret.ravel()
    m = int(sensitivity + 1)
    n = int(9 - number_of_lsb)
    pearsons_r = np.zeros((n, m), dtype=float)
    for x_i in range(n):
        for y_i in range(m):
            container_and_secret = encode_2_img(sensitivity, container_without_secret, bin_str, number_of_lsb)
            container_and_secret_i = container_and_secret.ravel()
            pearson_r_0, p_value = stats.pearsonr(container_1, container_and_secret_i)
            pearsons_r[x_i][y_i] = pearson_r_0
            sensitivity = sensitivity - 1
        number_of_lsb = number_of_lsb + 1
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # axes.plot_surface(sensitivitys, number_of_lsbs, pearsons_r, rstride=1, cstride=1)
    # plt.show()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(sensitivitys, number_of_lsbs, pearsons_r, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    print(pearsons_r, '\nГрафик готов')

secret, secret_file_name = input_secret()
container = input_container()
number_of_lsb = input_number_of_lsb()
sensitivity = input_sensitivity()
container_and_secret, modified_container, i = encode_img(sensitivity, container, secret, number_of_lsb)

if i < len(secret):
    # i - количество бит секрета, встроенных в контейнер
    if sensitivity > 0:
        print('Необходимо уменьшить чувствительность')
        step = input_step()
        while i < len(secret) and sensitivity >= 0:
            sensitivity = sensitivity - step
            container_and_secret, modified_container, i = encode_img(sensitivity, container, secret, number_of_lsb)
        if i == len(secret):
            save_output(container_and_secret, modified_container)
            mse, psnr, pearson_r = kpi_output(container, container_and_secret, secret, secret_file_name)
            print('Чувствительность уменьшена до', sensitivity)
            # chart_sensitivity(sensitivity, container, secret, number_of_lsb, mse, psnr, pearson_r)
            # chart_number_of_lsb_pearsons_r(sensitivity, container, secret, number_of_lsb, pearson_r)
            # chart_3d(sensitivity, container, secret, number_of_lsb, pearson_r)
        else:
            print('\nНевозможно встроить. контейнер слишком мал')
    else:
        print('\nНевозможно встроить. контейнер слишком мал')
else:
    save_output(container_and_secret, modified_container)
    mse, psnr, pearson_r = kpi_output(container, container_and_secret, secret, secret_file_name)
    # chart_sensitivity(sensitivity, container, secret, number_of_lsb, mse, psnr, pearson_r)
    # chart_number_of_lsb_pearsons_r(sensitivity, container, secret, number_of_lsb, pearson_r)
    # chart_3d(sensitivity, container, secret, number_of_lsb, pearson_r)
