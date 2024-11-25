import cv2
import matplotlib.pyplot as plt
import numpy as np

from source.utils import normalize
from source.preprocessing.preprocessing_extended_helper import convert_to_uint8

"""magn_map = np.load('/local/home/bjbraun/mask.npy')
fig, ax = plt.subplots()
ax.imshow(magn_map)
fig.show()"""

# magn_map_uint8 = np.asarray(magn_map, dtype=np.uint8)
img = np.load('/local/home/bjbraun/img_005.npy')
img = img / np.std(img)
img = convert_to_uint8(img)
img = cv2.medianBlur(img, 7)
img = 255 - img
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.set_title('Original Image')
fig.show()

# t_lower = int(np.quantile(img, 0.25))  # Lower Threshold
# t_upper = int(np.quantile(img, 0.75))  # Upper threshold
t_lower = 5  # 20
t_upper = 50  # 100
edge = cv2.Canny(img, t_lower, t_upper)
fig, ax = plt.subplots()
ax.imshow(edge, cmap='gray')
ax.set_title('Canny Edge Detection')
fig.show()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
# dilated = cv2.dilate(edge, (1, 1), iterations=10)
dilated = cv2.dilate(edge, kernel)
fig, ax = plt.subplots()
ax.imshow(dilated)
ax.set_title('Dilated')
fig.show()

def detect_ellipses(contours):
    ellipses = []
    for contour in contours:
        relevant_points = np.reshape(contour, (contour.shape[0], 2))
        ellipse = cv2.fitEllipse(relevant_points)
        ellipses.append(ellipse)
    return ellipses

cnts, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.set_title('Contours')
fig.show()

# relevant_points = np.reshape(cnt, (cnt.shape[0], 2))
# ellipse = cv2.fitEllipse(relevant_points)
ellipses = detect_ellipses(cnts)
# cv2.drawContours(img, ellipses, -1, (0, 255, 0), 2)
for ellipse in ellipses:
    cv2.ellipse(img, ellipse, (0, 255, 0), -2)
    # cv2.drawContours(mask, [ellipse], -1, 0, -1)
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.set_title('Ellipse')
fig.show()

"""quantile = np.quantile(magn_map, 0.75)
magn_map[magn_map > quantile] = 255
fig, ax = plt.subplots()
ax.imshow(magn_map)
fig.show()

mean_img = np.load('/local/home/bjbraun/mean_001.npy')
# mean_img[mean_img < np.quantile(mean_img, 0.1)] = 0
fig, ax = plt.subplots()
ax.imshow(mean_img)
fig.show()"""

"""mult = mean_img*(255-magn_map)
fig, ax = plt.subplots()
ax.imshow(mult)
fig.show()

quantile = np.quantile(mult, 0.75)
mult[mult < quantile] = 0
mult[mult > 0] = 1
# mult[:, 0:100] = 0
# mult[:, -100:] = 0
fig, ax = plt.subplots()
ax.imshow(mult)
fig.show()"""

"""t_lower = 60  # Lower Threshold
t_upper = 80  # Upper threshold
aperture_size = 5  # Aperture size
# img = cv2.imread("/local/home/bjbraun/image.png")
# img = np.expand_dims(mean_img, 2)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.load('/local/home/bjbraun/img_001.npy')
edge = cv2.Canny(gray, t_lower, t_upper)
fig, ax = plt.subplots()
ax.imshow(gray)
fig.show()
fig, ax = plt.subplots()
ax.imshow(edge)
fig.show()"""

# Create magn_map2, which sets all values to zero that are bigger than the quantile of magn_map
"""magn_map2 = magn_map.copy()
# quantile = np.quantile(magn_map, 0.5)
# magn_map2[magn_map > quantile] = 0
magn_map2 = 255 - magn_map2
quantile = np.quantile(magn_map2, 0.75)
magn_map2[magn_map2 < quantile] = 0

fig, ax = plt.subplots()
ax.imshow(magn_map2)
fig.show()"""

print('Ah')
