import os
os.getcwd()

collection = "/home/rpellerito/trackformer/data/EXCAV/test/"

for i, filename in enumerate(sorted(os.listdir(collection))):
    print("\n", filename)
    os.rename("/home/rpellerito/trackformer/data/EXCAV/test/" + filename,
              "/home/rpellerito/trackformer/data/EXCAV/test_rename/" + (6-len(str(i)))*str(0)+ str(i) + ".png")