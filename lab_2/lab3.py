import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from numpy.fft import fft2, fftshift
from numpy.fft import ifft2, ifftshift

def rgb_to_gray(img):
    (W, H) = img.size
    grey_img = np.array([ [0]*W for i in range(H) ])
    pix = img.load()
    for i in range(W):
        for j in range(H):
            grey_img[j][i] = pix[i, j][0]*0.3 + pix[i, j][1]*0.59 + pix[i, j][2]*0.11
    return grey_img
def LowPassFilter(radius, img_gray):
    rows, cols = img_gray.shape
    crow,ccol = rows//2 , cols//2
    mask = np.zeros((rows,cols),np.uint8)
    for i in range(rows):
        for j in range(cols):
            if ((i-crow)**2 + (j-ccol)**2)**0.5 <= radius:
                mask[i,j] = 1
    return mask
def Butterworth_Low(radius, img_gray, n):
    rows, cols = img_gray.shape
    crow,ccol = rows//2 , cols//2
    mask = np.ones(img_gray.shape)
    for i in range(rows):
        for j in range(cols):
            D = ((i-crow)**2 + (j-ccol)**2)**0.5
            mask[i,j] = 1 / (1 + (D/radius)**(2*n))
    return mask
def Gauss_Low(radius, img_gray):
    rows, cols = img_gray.shape
    crow,ccol = rows//2 , cols//2
    mask = np.ones(img_gray.shape)
    for i in range(rows):
        for j in range(cols):
            D = ((i-crow)**2 + (j-ccol)**2)**0.5
            mask[i,j] = np.exp((-(D)**2)/(2*radius**2))
    return mask
def HighPassFilter(radius, img_gray):
    rows, cols = img_gray.shape
    crow,ccol = rows//2 , cols//2
    mask = np.ones((rows,cols),np.uint8)
    for i in range(rows):
        for j in range(cols):
            if ((i-crow)**2 + (j-ccol)**2)**0.5 <= radius:
                mask[i,j] = 0
    return mask
def Butterworth_High(radius, img_gray, n):
    rows, cols = img_gray.shape
    crow,ccol = rows//2 , cols//2
    mask = np.ones(img_gray.shape)
    for i in range(rows):
        for j in range(cols):
            D = ((i-crow)**2 + (j-ccol)**2)**0.5
            if D == 0:
                D = 10**(-10)
            mask[i,j] = 1 / (1 + (radius/D)**(2*n))
    return mask
def Gauss_High(radius, img_gray):
    rows, cols = img_gray.shape
    crow,ccol = rows//2 , cols//2
    mask = np.ones(img_gray.shape)
    for i in range(rows):
        for j in range(cols):
            D = ((i-crow)**2 + (j-ccol)**2)**0.5
            mask[i,j] = 1-np.exp((-(D)**2)/(2*radius**2))
    return mask

img_dir = 'D:\\Programms\\PyCharm\\PyCharm\\PyCharm_projects\\Theme_7\\image.jpg'   # путь к фото
save_path = 'D:\\Programms\\PyCharm\\PyCharm\\PyCharm_projects\\Theme_7\\DIP\\Lab2\\'
with Image.open(img_dir) as img:    # открытие файла
    img.load()  #считывание данных
# plt.imshow(img)
# plt.axis('off')
# plt.show()

img_gray = rgb_to_gray(img)
# plt.imshow(img_gray, cmap='gray')
# plt.axis('off')
# plt.savefig(save_path + 'img_gray')
# plt.show()

F = fft2(img_gray)
F_shift = fftshift(F)
ampl_spectr = (20*np.log(1+np.abs(F_shift))).clip(0,255)
ifft_shift = ifftshift(F_shift)
# fig = plt.figure()
# plt.subplot(131),plt.imshow(img_gray, cmap = 'gray')
# plt.title('Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow((20*np.log(1+np.abs(F))).clip(0,255), cmap = 'gray')
# plt.title('fft2'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(ampl_spectr, cmap = 'gray')
# plt.title('fftshift(fft2)'), plt.xticks([]), plt.yticks([])
# plt.show()

n = 2
D_0 = [5, 10, 50, 250]
res = []
res1 = []
res2 = []
# for i in D_0:
#     maska = LowPassFilter(i, img_gray)
#     res.append(abs(ifft2(ifftshift(maska * F_shift))).clip(0,255))
#     maska1 = Butterworth_Low(i, img_gray, n)
#     res1.append(abs(ifft2(ifftshift(maska1 * F_shift))).clip(0,255))
#     maska2 = Gauss_Low(i, img_gray)
#     res2.append(abs(ifft2(ifftshift(maska2 * F_shift))).clip(0, 255))
# res_all = [res, res1, res2]
# for j in range (3):
#     fig = plt.figure()
#     for i in range(4):
#         plt.subplot(2,2,(i+1)), plt.imshow(res_all[j][i], cmap='gray')
#         plt.title(f'D0 = {D_0[i]}'), plt.xticks([]), plt.yticks([])
#     plt.show()

# for i in D_0:
#     maska = HighPassFilter(i, img_gray)
#     res.append(abs(ifft2(ifftshift(maska * F_shift))).clip(0,255))
#     maska1 = Butterworth_High(i, img_gray, n)
#     res1.append(abs(ifft2(ifftshift(maska1 * F_shift))).clip(0,255))
#     maska2 = Gauss_High(i, img_gray)
#     res2.append(abs(ifft2(ifftshift(maska2 * F_shift))).clip(0, 255))
# res_all = [res, res1, res2]
# for j in range (3):
#     fig = plt.figure()
#     for i in range(4):
#         plt.subplot(2,2,(i+1)), plt.imshow(res_all[j][i], cmap='gray')
#         plt.title(f'D0 = {D_0[i]}'), plt.xticks([]), plt.yticks([])
#     plt.show()

img_dir = 'D:\\Programms\\PyCharm\\PyCharm\\PyCharm_projects\\Theme_7\\6.png'   # путь к фото
with Image.open(img_dir) as img:    # открытие файла
    img.load()  #считывание данных

img_gray = rgb_to_gray(img)

# fig = plt.figure()
# plt.subplot(1,2,1), plt.imshow(img)
# plt.axis('off')
# plt.title('Original')
# plt.subplot(1,2,2), plt.imshow(img_gray, cmap='gray')
# plt.axis('off')
# plt.title('Gray')
# plt.show()

# plt.figure(figsize=(11, 11))
# plt.imshow(img_gray, cmap="gray")
# plt.title('Gray'), plt.xticks([]), plt.yticks([])
# plt.show()

spectrum = fftshift(fft2(img_gray))
# plt.figure()
# plt.imshow((20*np.log(1+np.abs(spectrum))).clip(0,255), cmap="gray")
# plt.title('Спектр'), plt.xticks([]), plt.yticks([])
# plt.show()
# filtered = Gauss_Low(250, img_gray) * spectrum
# plt.figure()
# plt.imshow((20*np.log(1+np.abs(filtered))).clip(0,255), cmap="gray")
# plt.title('Отфильтрованный спектр'), plt.xticks([]), plt.yticks([])
# plt.show()
# img_back = abs(ifft2(ifftshift(filtered)))
# plt.figure()
# plt.imshow(img_back, cmap="gray")
# plt.title('Отфильтрованное изображение'), plt.xticks([]), plt.yticks([])
# plt.show()

# def cross_filter(wide, img_gray_shape):
#     mask = np.ones(img_gray_shape)
#     for i in range(img_gray_shape[0]):
#         for j in range(img_gray_shape[1]):
#             if (img_gray_shape[1]//2 - wide <= j <= img_gray_shape[1]//2 + wide):
#                 mask[i,j] = 0
#     mask[img_gray_shape[0]//2 - 3*wide : img_gray_shape[0]//2 + 3*wide, img_gray_shape[1]//2 - 3*wide : img_gray_shape[1]//2 + 3*wide] = 1
#     return mask
#
# spectr = cross_filter(50, img_gray.shape)
# fig = plt.figure()
# plt.imshow((20*np.log(1+np.abs(spectr))).clip(0,255), cmap="gray")
# plt.title('Маска'), plt.xticks([]), plt.yticks([])
# plt.show()
# fig = plt.figure()
# plt.imshow(abs(ifft2(ifftshift(spectr*spectrum))), cmap = 'gray')
# plt.title('Изображение после избирательного фильтра'), plt.xticks([]), plt.yticks([])
# plt.show()

dots = np.array([[1176, 1439], [1034, 1345],[1315, 1528],
                   [977, 1348],[761, 1179],[691, 1169],
                   [632, 1167],[501, 1080],[430, 1080],
                   [728, 1491],[749, 1447],[777, 1398],
                   [886, 1152],[898, 1120],[922, 1063],[942, 1023]])
dot = np.ones(spectrum.shape)
for i in range(len(dots)):
    h = dots[i][0]
    w = dots[i][1]
    for j in range(dot.shape[0]):
        for j2 in range(dot.shape[1]):
            if (j - h)**2 + (j2 - w)**2 <= 10**2:
                dot[j,j2] = 0
plt.figure()
plt.imshow((20*np.log(1+np.abs(spectrum*dot))).clip(0,255), cmap="gray")
plt.title('Спектр с точками'), plt.xticks([]), plt.yticks([])
plt.show()
fig = plt.figure()
plt.imshow(abs(ifft2(ifftshift(dot*spectrum))), cmap = 'gray')
plt.title('Изображение после избирательной фильтрации точками'), plt.xticks([]), plt.yticks([])
plt.show()
dot_paral = np.copy(dot)
for i in range(dot_paral.shape[0]):
    for j in range(dot_paral.shape[1]):
        if i in range(954-200,400+1147+1) and j in range(550-200,957+1):
            dot_paral[i,j]=0
plt.figure()
plt.imshow((20*np.log(1+np.abs(spectrum*dot_paral))).clip(0,255), cmap="gray")
plt.title('Спектр с точками и прямоугольником'), plt.xticks([]), plt.yticks([])
plt.show()
fig = plt.figure()
plt.imshow(abs(ifft2(ifftshift(dot_paral*spectrum))), cmap = 'gray')
plt.title('Изображение после избирательной фильтрации точками и прямоугольником'), plt.xticks([]), plt.yticks([])
plt.show()