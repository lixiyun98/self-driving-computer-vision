import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import data_reader
from model import LeNet
import os
from scipy import ndimage
from skimage.transform import resize
from skimage.data import imread
from skimage import color

training_file = 'data/train.p'
validation_file = 'data/valid.p'
testing_file = 'data/test.p'

is_debug = True

data = data_reader.data(training_file, validation_file, testing_file)
data.print_data_info()

# according to the info we know that there is 43 classes of sign

# use a dictionary to manage the key-label
label_dict = {}
with open('signnames.csv', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        tmp = line.strip('\n')
        tmp = tmp.split(',')
        label_dict[tmp[0]] = tmp[1]
    f.close()


def subplot_show(img_list, label_list, gray=False):
    """
    show 16 pictures
    the img_list must be a list with 16 images
    """
    for i in range(len(img_list)):
        plt.subplot(4, 4, i + 1)
        plt.title(label_dict[str(int(label_list[i]))])
        if gray is True:
            plt.imshow(img_list[i], cmap='gray')
        else:
            plt.imshow(img_list[i])
    plt.show()

# # the original images
# subplot_show(data.test_x[:4])

# image after normalization and convert to gray scale
if is_debug:
    subplot_show(data.test_x[:16], data.test_y[:16], gray=True)

model = LeNet(data)

# the training stage
num_epoch = 20
batch_size = 128
save_dir = 'model_save_dir'
num_step = int(np.round(data.n_train/batch_size))
is_training = False

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_path = os.path.join(save_dir, 'convnet_model.ckpt')


def outputFeatureMap(image_input, sess, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    image_input = np.reshape(image_input, [1, 32, 32])
    activation = tf_activation.eval(session=sess, feed_dict={model.x_placeholder:image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15, 15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


with tf.Session(graph=model.graph) as sess:
    sess.run(tf.global_variables_initializer())

    if is_training:
        for epoch in range(num_epoch):
            for step in range(num_step):
                batch_x, batch_y = data.next_batch(batch_size)

                # define a feed dict to input minibatch
                feed_dict = {model.x_placeholder: batch_x, model.y_placeholder: batch_y, model.keep_p:0.5}

                sess.run(model.train_op, feed_dict=feed_dict)

            # eval stage
            val_loss, val_acc, merge, g_step = sess.run([model.loss, model.accuracy, model.merge, model.global_step],
                                                        feed_dict={model.x_placeholder:data.val_x,
                                                                   model.y_placeholder:data.val_y,
                                                                   model.keep_p:1.0})
            # add some training info to summary file
            model.summary_writer.add_summary(merge, global_step=epoch)

            print('validation loss at eopch %d: %f' % (epoch, val_loss))
            print('validation accuracy at eopch %d: %f' % (epoch, val_acc))

            model.saver.save(sess, save_path, global_step=epoch)

        # test stage
        print('Test accuracy:', model.accuracy.eval(
            feed_dict={model.x_placeholder: data.test_x, model.y_placeholder: data.test_y, model.keep_p:1.0}), '%')

    else:
        # load the pre-trained model and classify the images from web
        model.saver.restore(sess, save_path + '-19')
        # read the images and labels
        path_list = os.listdir('web_imgs')
        imgs_list = []
        imgs_arr = np.zeros([6, 32, 32])
        label_arr = np.zeros([6])
        for item in path_list:
            imgs_list.append(imread(os.path.join('web_imgs', item)))
        for i, item in enumerate(imgs_list):
            item = (item - 128.) / 128.
            item = color.rgb2gray(item)
            imgs_arr[i, :, :] = resize(item, (32, 32))
        for i, item in enumerate(path_list):
            item = item.split('.')[0]
            item = int(item)
            label_arr[i] = item
        # run the pre-trained model to predict
        output, acc, loss, top_k = sess.run([model.output, model.accuracy, model.loss, model.top_k], feed_dict={model.x_placeholder:imgs_arr,
                                                                                            model.y_placeholder:label_arr,
                                                                                            model.keep_p:1.0})

        print(top_k)

        print('the predict accuracy of web images is:', str(acc))
        predict = np.argmax(output, 1)
        for i in range(len(imgs_arr)):
            plt.subplot(2, 3, i + 1)
            plt.title('predict:'+label_dict[str(int(predict[i]))]+'\n truth:'+label_dict[str(int(label_arr[i]))])
            plt.imshow(imgs_arr[i, :, :], cmap='gray')
        plt.show()







