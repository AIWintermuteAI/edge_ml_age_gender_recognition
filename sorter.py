import os
i = 0
for filename in os.listdir("."):
    print(filename)
    os.rename(filename, str(i)+".jpg")
