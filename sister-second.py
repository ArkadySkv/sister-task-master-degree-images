import numpy as np
from PIL import Image

img = Image.open('./images/glioma1.jpg')
pixels = np.array(img)

height, width, channels = pixels.shape

gradient = np.zeros((height, width, channels))

for h in range(1, height-1):
    for w in range(1, width-1):
        for c in range(channels):
            gradient[h, w, c] = pixels[h+1, w, c] - pixels[h-1, w, c]

new_image = pixels + 2 * gradient

new_image = Image.fromarray(np.uint8(new_image))
new_image.show()

def roberts_and_sobel(img):
    mask1 = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]])
    mask2 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    for h in range(1, height-1):
        for w in range(1, width-1):
            for c in range(channels):
                gradient_x = np.sum(img[h-1:h+2, w-1:w+2, c] * mask1)
                gradient_y = np.sum(img[h-1:h+2, w-1:w+2, c] * mask2)
                gradient[h, w, c] = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient


combined_gradient = roberts_and_sobel(pixels)

new_image_2 = pixels + 2 * combined_gradient

new_image_2 = Image.fromarray(np.uint8(new_image_2))
new_image_2.show()
