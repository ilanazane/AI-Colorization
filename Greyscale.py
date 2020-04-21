def toGrey(image):
    #recolor each pixel
    for i in range(0,len(image)):
        for j in range(0, len(image[i])):
            image[i][j] = 0.21*image[i][j][0] + 0.72*image[i][j][1] + 0.07*image[i][j][2]

    return image
