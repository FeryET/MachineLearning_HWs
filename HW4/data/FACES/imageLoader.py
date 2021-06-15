import scipy.misc
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    train_data , train_label = loadImages('train')
    test_data , test_label = loadImages('test')

    print("number of train data images is" , train_data.shape[0] , "and number of features for each image is", train_data.shape[1])

    ### To show each images, you should reshape it to 256,256 and then use 'plt.imshow'
    plt.imshow(train_data[0].reshape(256,256) , cmap='gray')
    plt.show()

    ### Write your code here


    ###



def loadImages(dirName):
    # This function loads images from any directory
    # :param str dirName: is address of the directory (string)

    data = []
    label = []
    for root, dirs, files in os.walk(dirName):
        for file in files:
            face = scipy.misc.imread(os.path.join(root, file)) # Load image from a path
            face = face.reshape(256 * 256, ).tolist()          # Flatten image . Note: size of any image is 256,256
            data.append(face)
            label.append(file.split('.')[1])                   # Label of an image is in its fileName
    return np.asarray(data) , label



if __name__ == '__main__':
    main()