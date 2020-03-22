import cv2
import numpy as np
import binascii
from sklearn.metrics import mean_squared_error

print ('Встраивание изображения в контейнер')

def input_secret():    
    secret_file_name = str (input('Введите имя файла с секретом:'))
    fin = open(secret_file_name, "rb")
    secret = fin.read()
    fin.close()
    # преобразовать каждый байт данных в 2-значную шестнадцатеричную строку
    hex_str = str(binascii.hexlify(secret))
    # создать строку битов из двухзначных шестнадцатеричных строк
    hex_list = []
    bin_list = []
    for ix in range(2, len(hex_str)-1, 2):
        word = hex_str[ix]+hex_str[ix+1]
        hex_list.append(word)
        bin_list_0=bin(int(word, 16))[2:]
        bin_list_0 = bin_list_0.zfill(8)
        bin_list.append(bin_list_0)
    bin_str = "".join(bin_list)
    return bin_str, secret_file_name

def input_container():    
    container_img = str (input('Введите имя файла-контейнера:'))
    # преобразовать container в массив
    container = cv2.imread(container_img)
    return container

def input_sensitivity(): 
    # sensitivity - минимально допустимое среднеквадратичное отклонение.
    # Встраивание выполняется, если  среднеквадратичное отклонение больше, чем sensitivity
    while True:
        sensitivity = input('Введите чувствительность (минимально допустимое среднеквадратичное отклонение): ')        
        try:
            sensitivity=float(sensitivity)
            return float(sensitivity)
        except ValueError:
            print('Необходимо ввести числовое значение!')
            
def input_step(): 
    while True:
        step = input('Введите шаг уменьшения чувствительности: ')
        try:
            step=float(step)
            return float(step)
        except ValueError:
            print('Необходимо ввести числовое значение!')
    
# встроить изображение в контейнер
def encode_img(sensitivity, container_without_secret, bin_str):
    i=0 # количество бит секрета, встроенных в контейнер
    value = 0    
    rows, cols, ch = container_without_secret.shape
    modified_container = np.zeros([rows, cols, ch])
    container_with_secret = container_without_secret.copy()
    for x in range(1, rows-1):
        for y in range(1, cols-1):
            for z in range(0, ch):
                a1=container_with_secret[x-1][y-1][z]
                a2=container_with_secret[x-1][y][z]
                a3=container_with_secret[x-1][y+1][z]
                a4=container_with_secret[x][y-1][z]
                a5=container_with_secret[x][y][z]
                a6=container_with_secret[x][y+1][z]
                a7=container_with_secret[x+1][y-1][z]
                a8=container_with_secret[x+1][y][z]
                a9=container_with_secret[x+1][y+1][z]
                data=[a1, a2, a3, a4, a5, a6, a7, a8, a9]            
                if np.std(data)>=sensitivity:
                    if int(bin_str[i])!=int(bin(a5)[-1]): # изменяем если значение бита секрета не равно значению НЗБ (LSB)
                        if a5!=255 and a5!=0: # замена, если не крайнее значение
                            if value%2==0:
                                container_with_secret[x][y][z] = a5+1                                
                            else:
                                container_with_secret[x][y][z] = a5-1                                
                            modified_container[x][y] = [255,255,255]
                            value=value+1
                            i=i+1
                        elif a5==255:  # замена, если крайнее верхнее значение
                            container_with_secret[x][y][z] = a5-1                                
                            modified_container[x][y] = [255,255,255]
                            value=value+1
                            i=i+1
                        else:         # замена, если крайнее нижнее значение
                            container_with_secret[x][y][z] = a5+1                                
                            modified_container[x][y] = [255,255,255]
                            value=value+1
                            i=i+1                            
                    else:
                        i=i+1                               
                if i==len(bin_str): break
            if i==len(bin_str): break
        if i==len(bin_str): break
    return container_with_secret, modified_container, i, value


def calculation_сapacity(secret, modified_container, number_of_bit_secret, number_of_byte_modified_container):
    # расчёт полезной нагрузки: кол-во пикселей секрета/ кол-во использованных пикселей контейнера
    secret = cv2.imread(secret)
    rows, cols, ch = secret.shape
    number_of_pixels_secret=rows*cols
    number_of_pixels_modified_container = np.sum(modified_container[1:-1,1:-1,0:1] == 255)
    сapacity = number_of_pixels_secret / number_of_pixels_modified_container
    # бит/байт
    сapacity_2 = number_of_bit_secret / number_of_byte_modified_container
    return сapacity, сapacity_2

# сохранить изображения
def save_output(container_and_secret, modified_container):
    cv2.imwrite("container_and_secret.png", container_and_secret)
    cv2.imwrite("modified_container.png", modified_container)                    
    print()
    print('Контейнер со встроенным секретом сохранён в файле container_and_secret.png')
    print('Изменённые пиксели контейнера выделены белым цветом на изображении в сохранённом в файле modified_container.png')
    
secret, secret_file_name = input_secret()
container = input_container()
sensitivity = input_sensitivity() 
container_and_secret, modified_container, i, value = encode_img(sensitivity, container, secret)
сapacity, сapacity_2 = calculation_сapacity(secret_file_name, modified_container, len(secret), value)

if i<len(secret):
    # i - количество бит секрета, встроенных в контейнер
    print('Необходимо уменьшить чувствительность')
    step = input_step()
    while i<len(secret) and sensitivity>=0:
        sensitivity=sensitivity-step
        container_and_secret, modified_container, i, value = encode_img(sensitivity, container, secret)
    if i==len(secret):
        save_output(container_and_secret, modified_container)
        correlation = np.sqrt(((container - container_and_secret) ** 2).mean())
        #correlation = mean_squared_error(container, container_and_secret)
        print('Correlation:', correlation)    
        сapacity, сapacity_2 = calculation_сapacity(secret_file_name, modified_container, len(secret), value)
        print('Чувствительность уменьшена до', sensitivity)
        print('Ёмкость (пикс):', сapacity)
        print('Ёмкость (бит/байт):', сapacity_2)
    if sensitivity<0:
        print()
        print('Невозможно встроить. Контейнер слишком мал')    
else:
    save_output(container_and_secret, modified_container)
    correlation = np.sqrt(((container - container_and_secret) ** 2).mean())
    #correlation = mean_squared_error(container, container_and_secret)
    print('Correlation:', correlation)
    сapacity, сapacity_2 = calculation_сapacity(secret_file_name, modified_container, len(secret), value)
    print('Ёмкость (пикс):', сapacity)
    print('Ёмкость (бит/байт):', сapacity_2)
