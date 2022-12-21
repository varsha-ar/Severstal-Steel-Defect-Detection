"""
# Steel Sheet Defects Detection using Vanilla U-Net

The data for the steel sheets defect image segmentation project was obtained from Severstal and Kaggle 
repositories (Kaggle Inc, 2019). Identifying different types of steel sheet defects is critical to improving Severstal’s 
automation, increasing efficiency, and maintaining high quality in their production. The company has taken major steps 
towards combining newer technologies with steel production. Please find the dataset kn Kaggle.

### Description of Dataset

The provided dataset contains train and test images with an area of 256 x 1600 pixels each, 
and the labels have pixel values where the defects are segmented. 
Each image can have potentially 4 kinds of defects (Types 1, 2, 3 and 4). 

The analysis of this project was possible with the help of xhlulu’s boilerplate code (https://www.kaggle.com/xhlulu/severstal-simple-keras-u-net-boilerplate). The author graciously agreed to share their work on public Kaggle kernels, and I was able to fork off of their work and introduce many of my own concepts.
"""

from keras.optimizers import *
from tqdm import tqdm
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import Callback, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import keras
import cv2
import json
import os
from google.colab import drive
drive.mount('/content/drive')

# Importing the required modules and libraries


"""### Utility Functions
Next, we go about defining some of the utility functions that will be used for various operations throughout the data analysis.
"""


def mask2rle(img):
    '''
    This function converts a pixel mask in the form of an array into an RLE encoded string
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(256, 1600)):
    '''
    Takes in the RLE of one image, creates a 1/0 mask, and returns the reshaped image.

    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def build_masks(rles, input_shape):
    """
    Takes in RLEs (rles: EncodedPixels) of one image with all 4 defects, and creates
    zero masks for that image in the same shape as input_shape, and then creates the 
    1/0 labelled masks corresponding to each defect. A (height, width, 4) size list is 
    returned corresponding to the 1/0 label map/mask for each defect.
    """
    depth = 4  # len(rles)   #4
    height, width = input_shape
    masks = np.zeros((height, width, depth))

    for i, rle in enumerate(rles):
        """Looping for each of the 4 types of defects"""
        if type(rle) is str:
            # calling the rle2mask function to build the mask label
            masks[:, :, i] = rle2mask(rle, (width, height))
        else:
            masks[:, :, i] = np.nan

    return masks


def build_rles(masks):
    """
    This function builds the RLEs for each (height, width, 4) size list containing the 
    1/0 label map/mask, and returns the RLEs
    """
    width, height, depth = masks.shape

    rles = [mask2rle(masks[:, :, i])
            for i in range(depth)]

    return rles


def percent_mask_per_image(mask):
    """
    Takes in a mask tuple of shape (height, width, 4), creates a dataframe for each of the
    4 defect masks and returns a dictionary of percentage areas of each of the 4 defect masks
    """

    percentage = dict()
    for i in range(4):
        # print('--------------------------')
        # print(mask[:,:,i])
        # print(mask[:,:,i].shape)
        # print('--------------------------')
        mask_df_defect = pd.DataFrame(mask[:, :, i])
        percentage[i] = mask_df_defect.sum().sum(
        ) / (mask_df_defect.shape[0] * mask_df_defect.shape[1])

    return percentage


def dice_coef(y_true, y_pred, smooth=1):
    """Definition of the Dice Coefficient"""
    # flattening the images into arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # the intersection is the sum of all the element-wise multiplied arrays
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# obtained from https://www.kaggle.com/xiejialun/keras-u-net-pre-post-processing


def dice_loss(y_true, y_pred):
    """Calculation of the Dice Loss (1 - Dice Coefficient)"""
    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    """Binary cross-entropy used as a loss measure to minimize"""
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


"""### Data Loading
We now begin the loading process, and prepare the data for exploratory data analysis. The data is stored in a dataframe *train_df*. 
"""

train_df = pd.read_csv(
    '/content/drive/MyDrive/severstal-steel-defect-detection/train.csv')
train_df['ImageId'] = train_df['ImageId_ClassId'].apply(
    lambda x: x.split('_')[0])
train_df['ClassId'] = train_df['ImageId_ClassId'].apply(
    lambda x: x.split('_')[1])
train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()

print(train_df.shape)
train_df.head()
# classifying whether the image has a mask, based on a null value in the EncodedPixels column
#train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()
#train_df = train_df[['EncodedPixels', 'ImageId', 'ClassId', 'hasMask']]

# Rearranging the images in terms of most masks per image: mask_count_df has the number of defect types per image

mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
mask_count_df.rename(columns={'hasMask': 'num_total_masks'}, inplace=True)
# mask_count_df.reset_index(inplace=True)

print(mask_count_df.head())
print(mask_count_df.shape)


#mask_count_df = train_df.groupby('ImageId').agg(np.sum).reset_index()
# Sorting the values in descending order and renaming the count of masks as 'num_total_masks'
##mask_count_df.sort_values('hasMask', ascending=False, inplace=True)
#mask_count_df.drop('ClassId', axis=1, inplace=True)
# mask_count_df.head()
# mask_count_df.shape
# mask_count_df.rename(columns={'hasMask':'num_total_masks'}, inplace=True)

# Here we read the sample_submission.csv file, split the Image and Class IDs, and create a test_imgs dataframe
# with unique imageId names
sub_df = pd.read_csv(
    '/content/drive/MyDrive/severstal-steel-defect-detection/sample_submission.csv')
sub_df['ImageId'] = sub_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
sub_df.head()

"""## Exploratory Data Analysis
### Encoding images with masks 
To start the EDA, we need to understand how images look with their defect masks on. 
"""

# For encoding all images with the masks
for i in range(4):
    mask_count_df['defect_%d_percentage' % i] = 0
mask_count_df.head()

"""Next we take each image name from the *mask_count_df*, filter it out from the original *train_df* dataframe, extract the *EncodedPixels* string into a variable rles. This rles is sent as an argument into the *build_masks* function, the return values are stored in *mask* variable. Also included in the loop is the calculation of the pixel area occupied by the defects (given by total 1s divided by total number of pixels).  """

for x, name in enumerate(mask_count_df['ImageId']):
    filename = name

    # extracting the image rows from the train_df dataframe
    image_df = train_df[train_df['ImageId'] == filename]
    # print(type(image_df['EncodedPixels'].iloc[0]))

    # building masks from the Encoded Pixels string, and storing the masks in mask variable
    rles = image_df['EncodedPixels'].to_numpy()
    # print('---------------------------------')
    # print(len(rles))
    # print('---------------------------------')
    mask = build_masks(rles, (256, 1600))

    # print(mask.shape)
    # calculating percentage areas of masks per image
    percent_mask = percent_mask_per_image(mask)

    # storing the percent areas as columns in the mask_count_df dataframe
    for i in range(2, 6):
        mask_count_df.iloc[x, i] = percent_mask[i-2]

# Examining the mask_count_df after the operation
mask_count_df.head()

"""Because there are a lot of images with no defects, an improvement on the model was to only consider images with at least one defect as an input into the training data. The previous iteration of the model with 0 defects also included showed about 20% dice coefficient (results not shown). Hence, a *non_zero_mask_df* is created as follows."""

non_zero_mask_df = mask_count_df[mask_count_df.hasMask > 0]

# Initializing color values on a 256 color spectrum for each of the four defects
colors = [191, 127, 0, 255]
plt.figure(figsize=(15, 15))

# Initializing dictionary to hold the axis information
img_ax = dict()

# 20 images
for y in range(0, 19):

    # for all the even-valued (left) columns, have the original image mapped out
    if y % 2 == 0:
        image_df = train_df[train_df['ImageId'] == mask_count_df.ImageId[y]]
        path = f"/content/drive/MyDrive/severstal-steel-defect-detection/train_images/{image_df['ImageId'].iloc[0]}"

        # reading the image path through OpenCV imread function into img
        img = cv2.imread(path)

        # converting the image from 3 colors into grayscale (to avoid unnecessary computations)
        altered_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # creating masks and obtaining arrays of each mask in mask_sample variable
        rles = image_df['EncodedPixels'].values
        mask_sample = build_masks(rles, (256, 1600))

        img_ax[y] = plt.subplot(10, 2, y+1)

        # showing the original image in the even column
        img_ax[y].imshow(img)
        plt.title('Original image' + mask_count_df.ImageId[y])
        img_ax[y].axis('off')

        # showing the picture with defects in the right column
        img_ax[y] = plt.subplot(10, 2, y+2)
        for i in range(4):
            """Filling in each of the four defects in the picture"""
            ind_mask = mask_sample[:, :, i]
            altered_img[ind_mask == 1] = colors[i]
            img_ax[y].imshow(altered_img)
            plt.title('Masked Image ' + mask_count_df.ImageId[y])
            img_ax[y].axis('off')

    if y == 19:
        break

plt.show()

"""## Analyzing the Data
Understanding the parameters of the data helps us create better models. 
"""

# Let's see how many defects per type are present
num_defects_types = train_df.groupby('ClassId').sum().rename(
    columns={'hasMask': 'num_defects'})
num_defects_types

# Visualizing the percentage of area covered by each of the defects

percent_defects = num_defects_types/12568  # Total number of images
percent_defects.plot(kind='bar')
plt.ylabel('Percentage of images')

"""As one can observe, Class 3 defects are quite common in images, and they consist of about 40% of all defects! Let us now examine the area covered by each of the defects and their distribution.

From the *mask_count_df* dataframe, we will extract the non-zero values for each defect type, and plot its spread as follows.
"""

plt.figure(figsize=(15, 10))
ax = dict()

# Looping over each of the 4 defects
for i in range(4):
    ax[i] = plt.subplot(2, 2, i+1)

    # the defect_column contains the values of the particular defect percentage area
    defect_column = mask_count_df['defect_{}_percentage'.format(i)]

    # label is the number of zero-values present in each defect
    label = (defect_column == 0).sum()

    # plotting histograms where defect_column has filtered out the zero-values
    # a log distribution is used because the areas are too small to observe using linear axes
    ax[i].hist(defect_column[defect_column > 0], log=True,
               bins=int(np.sqrt(12568)), color='green')

    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('Percentage of area covered by defect {}'.format(i+1))
    plt.ylabel('Frequency')
    print('The number of zero-values present in defect {} are '.format(i+1) + str(label))

"""As can be observed, the areas covered by both defects 3 and 4 occupy major portions of the image (between 0.01 and 0.1% of the total area), whereas defects 1 and 2 occupy tinier portions.

## Data Generator Class
As explained above, the dice coefficient is coded as a function dice_coef(y_true, y_pred). Two other functions bce_dice_loss and dice_loss are defined for calculating the log-loss of the model. 

To be able to load the large number of files in batches, a custom data generator class with the keras.utils.Sequence parameter is instantiated. The *__len__* method returns the number of batches per epoch, and *__get_item__*(index) returns the images (X) and the masks (y) associated with the indices of the training and the validation data. The *on_epoch_end()* function is used to trigger a shuffle of the indices at the end of an epoch, to create a more robust model.
"""


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, df, target_df=None, mode='fit',
                 base_path='/content/drive/MyDrive/severstal-steel-defect-detection/train_images/',
                 batch_size=32, dim=(256, 1600), n_channels=1,
                 n_classes=4, random_state=2019, shuffle=True):
        """
        All these values get instantiated once the DataGenerator class is called.
        """
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state

        # Needed for index updating after each epoch
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        # The __generate_X function is called
        X = self.__generate_X(list_IDs_batch)

        # The fit mode will call the __generate_y function that creates the mask labels
        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)
            return X, y

        # The predict mode will return the X values
        elif self.mode == 'predict':
            return X

        else:
            raise AttributeError(
                'The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        'Updates indexes after each epoch in a random order'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Using the filenames associated with the indexes, the images are fetched from the basepath
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_grayscale(img_path)

            # Store samples
            X[i, ] = img

        return X

    def __generate_y(self, list_IDs_batch):
        'Generates masks for the corresponding images'
        y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        # Using the filenames associated with the indexes, the masks are built from the basepath
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name]

            rles = image_df['EncodedPixels'].values
            masks = build_masks(rles, input_shape=self.dim)

            y[i, ] = masks

        return y

    def __load_grayscale(self, img_path):
        'To create images in grayscale'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        'To create images in RGB format'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img


# Instantiating train and validation values from the indexes randomly using a batch size of 32
BATCH_SIZE = 32

# The train and test split is 15%
train_idx, val_idx = train_test_split(
    non_zero_mask_df.index, random_state=2005653, test_size=0.15
)

train_generator = DataGenerator(
    train_idx,
    df=mask_count_df,
    target_df=train_df,
    batch_size=BATCH_SIZE,
    n_classes=4
)

val_generator = DataGenerator(
    val_idx,
    df=mask_count_df,
    target_df=train_df,
    batch_size=BATCH_SIZE,
    n_classes=4
)

"""In our particular example of image segmentation, we are using the U-Net Image segmentation model, developed by (Olaf Ronneberger, 2015): https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28. The architecture has been taken from the paper, and is shown below. The architecture analyzes an encoder-decoder system. Here the encoded vector is further expanded out into the original image size using upsampling techniques and concatenation with previous transformed data. This process is called decoding. The Adam optimizer was used for compiling at a learning rate of 0.001."""


def build_model(input_shape):
    inputs = Input(input_shape)

    c1 = Conv2D(8, (3, 3), activation='elu', padding='same')(inputs)
    c1 = Conv2D(8, (3, 3), activation='elu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(16, (3, 3), activation='elu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='elu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(32, (3, 3), activation='elu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='elu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(64, (3, 3), activation='elu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='elu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(64, (3, 3), activation='elu', padding='same')(p4)
    c5 = Conv2D(64, (3, 3), activation='elu', padding='same')(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    c55 = Conv2D(128, (3, 3), activation='elu', padding='same')(p5)
    c55 = Conv2D(128, (3, 3), activation='elu', padding='same')(c55)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c55)
    u6 = concatenate([u6, c5])
    c6 = Conv2D(64, (3, 3), activation='elu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='elu', padding='same')(c6)

    u71 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u71 = concatenate([u71, c4])
    c71 = Conv2D(32, (3, 3), activation='elu', padding='same')(u71)
    c61 = Conv2D(32, (3, 3), activation='elu', padding='same')(c71)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c61)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='elu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='elu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='elu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='elu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='elu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='elu', padding='same')(c9)

    outputs = Conv2D(4, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss=bce_dice_loss, metrics=[dice_coef])

    return model


# Calling the model will instantiate the model architecture and is shown below
model1 = build_model((256, 1600, 1))
model1.summary()

"""Next the actual model is trained and validated over 7 epochs and the dice coefficient calculated as follows. It is saved in a file called model1.h5. """

# First Model
checkpoint = ModelCheckpoint(
    'model1.h5',
    monitor='val_dice_coef',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

history = model1.fit(x=train_generator, validation_data=val_generator, callbacks=[
                     checkpoint], workers=1, epochs=7)

"""The loss values show large negative values. This could be the result of the function not averaging the pixel-wise losses, and instead showing the combined values of the pixel losses. The parameter of interest is the dice coefficient though, and it shows a 0.6868 value during its training and validation over 7 epochs. This is quite a good model for a first-time effort. """

# Understanding the losses and dice coefficients over epochs visually
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.title('Losses for Train and Validation Data')
history_df[['dice_coef', 'val_dice_coef']].plot()
plt.xlabel('Epochs')
plt.ylabel('Coefficients')
plt.title('Dice Coefficients for Train and Validation Data')

"""### Fitting the Test Data
Once the model has been trained, we will try and predict the masks for the test images from the Kaggle website. 
"""

# Importing the tqdm package for progress bar completions

# The weights from the trained model model1.h5 are loaded
model1.load_weights('model1.h5')
test_df = []

# Iniitalizing a loop for taking in 500 test images' indexes at a time
for i in range(0, test_imgs.shape[0], 500):

    # batch_idx is the incremental list of numbers over various batches
    batch_idx = list(range(i, min(test_imgs.shape[0], i + 500)))

    # creating DataGenerator object with mode = 'predict'
    test_generator = DataGenerator(
        batch_idx,
        df=test_imgs,
        shuffle=False,
        mode='predict',
        base_path='../input/severstal-steel-defect-detection/test_images',
        target_df=sub_df,
        batch_size=1,
        n_classes=4
    )

    # running the predict_generator function of the model class using the test_generator object created above
    batch_pred_masks = model1.predict_generator(
        test_generator,
        workers=1,
        verbose=1,
        use_multiprocessing=False
    )

    # appending the predicted masks to the test_df list created above
    for j, b in tqdm(enumerate(batch_idx)):

        # going over each filename in the test_imgs dataframe
        filename = test_imgs['ImageId'].iloc[b]

        # copying the rows corresponding to the filename in a new dataframe image_df
        image_df = sub_df[sub_df['ImageId'] == filename].copy()

        # creating masks from the batch_pred_masks object created above
        pred_masks = batch_pred_masks[j, ].round().astype(int)

        # building RLEs from the masks using the pred_masks above
        pred_rles = build_rles(pred_masks)

        # adding and appending pred_rles as EncodedPixels into the test_df dataframe
        image_df['EncodedPixels'] = pred_rles
        test_df.append(image_df)

# Next we concatenate all the list values of test_df together
test_df = pd.concat(test_df)
test_df.drop(columns='ImageId', inplace=True)

test_df.head()

# Here we convert the dataframe test_df into an Excel file
test_df.to_excel('submission.xlsx', index=False)

# Here we convert the dataframe test_df into a csv file
test_df.to_csv('submission.csv', chunksize=250, index=False)
