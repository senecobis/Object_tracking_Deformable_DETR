from PIL import Image
import os


# Giving The Original image Directory
# Specified
image_folder = "/home/rpellerito/trackformer/data/EXCAV/test_raw"

images = []
list = sorted(os.listdir(image_folder))
for img in list:
    print(img)

    images.append(img)

    Original_Image = Image.open("/home/rpellerito/trackformer/data/EXCAV/test_raw/" + str(img))

    # Rotate Image By 90 Degree
    rotated_image1 = Original_Image.rotate(90)
    
    rotated_image1.save("/home/rpellerito/trackformer/data/EXCAV/test/" + str(img))
