import numpy as np
import cv2
import os 
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, prefix='appa-real/imgs/', shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.prefix = prefix
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def save_img(self, path, age, img):
        #img = img.astype(np.int8)
        filename = os.path.join("data", "imdb_processed", str(age)+"_"+os.path.basename(path))
        #print(filename)
        cv2.imwrite(filename, img)

    def __data_generation(self, batch_ids):
        X = []
        y_gender_list = []
        y_age_list = []
        for i in range(self.batch_size):
            img = cv2.imread(self.prefix + self.X_train[batch_ids[i]], 1)
            #print(self.prefix + self.X_train[batch_ids[i]])
            img = cv2.resize(img, (128, 128))
            if self.datagen:
                img = self.datagen.random_transform(img)
                img = self.datagen.standardize(img)
            #img_to_save = img

            img = img.astype(np.float32)
            img /= 255.
            img -= 0.5
            img *= 2.
            img = img[:, :, ::-1]

            #output_image = img[:, :, ::-1]
            #output_image /= 255.
            #output_image += 0.5
            #output_image *= 255.

            X.append(img) 
            
            y_gender = self.y_train[0][batch_ids[i]]
            y_age = self.y_train[1][batch_ids[i]]

            #self.save_img(self.X_train[batch_ids[i]][0][0], np.argmax(y_age), img_to_save)

            y_gender_list.append(y_gender)
            y_age_list.append(y_age)
        
        return np.array(X), [y_gender_list, y_age_list]
