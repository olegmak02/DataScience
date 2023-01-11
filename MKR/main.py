import cv2
import numpy as np

# Функція для відновлення зображення
def automatic_correction(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


image = cv2.imread('example.jpeg')

# Псування та вивід зіпсованого зображення
image = cv2.convertScaleAbs(image, alpha=0.3, beta=15)
cv2.imshow('spoilt', image)

# Відновлення зіпсованого зображення
auto_result, alpha, beta = automatic_correction(image)
print('------------- Підібрані параметри для відновлення --------------')
print('alpha', alpha)
print('beta', beta)
cv2.imwrite('auto_result.jpeg', auto_result)
cv2.imshow('recovered', auto_result)
cv2.waitKey()


# Виділення контурів предметів на зображенні
hsv_min = np.array((0, 0, 0), np.uint8)
hsv_max = np.array((160, 160, 160), np.uint8)

img = cv2.imread("auto_result.jpeg")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )
thresh = cv2.inRange(hsv, hsv_min, hsv_max )
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 0)
cv2.drawContours(img, contours, -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 2)

# Вивід контурів на зображенні
cv2.imshow('borders', img)
cv2.imshow('thresh', thresh)
cv2.waitKey()
cv2.destroyAllWindows()
