import os
import shutil
import numpy as np
import tensorflow as tf

# 下载得到的训练集图像
image_path = 'D:/train/train'
# 将猫狗分类保存的路径
train_path = 'afterClassify'

image_list = os.listdir(image_path)
# 读取1000张猫狗图像，按照图像名字分别保存
for image_name in image_list[0:1000]:
    class_name = image_name[0:3]
    save_path = os.path.join(train_path, class_name)
    if not (os.path.exists(save_path)):
        os.mkdir(save_path)
    file_name = os.path.join(image_path, image_name)
    save_name = os.path.join(save_path, image_name)
    shutil.copyfile(file_name, save_name)
for image_name in image_list[12500:13500]:
    class_name = image_name[0:3]
    save_path = os.path.join(train_path, class_name)
    if not (os.path.exists(save_path)):
        os.mkdir(save_path)
    file_name = os.path.join(image_path, image_name)
    save_name = os.path.join(save_path, image_name)
    shutil.copyfile(file_name, save_name)




class vgg16:
    def __init__(self, imgs):
        self.parameters = []
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc8

    def saver(self):
        return tf.train.Saver()

    def maxpool(self, name, input_data, trainable):
        out = tf.nn.max_pool(input_data, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name=name)
        return out

    def conv(self, name, input_data, out_channel, trainable):
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.get_variable("weights", [3, 3, in_channel, out_channel], dtype=tf.float32, trainable=False)
            biases = tf.get_variable("biases", [out_channel], dtype=tf.float32, trainable=False)
            conv_res = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
            res = tf.nn.bias_add(conv_res, biases)
            out = tf.nn.relu(res, name=name)
        self.parameters += [kernel, biases]
        return out

    def fc(self, name, input_data, out_channel, trainable=True):
        shape = input_data.get_shape().as_list()
        if len(shape) == 4:
            size = shape[-1] * shape[-2] * shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data, [-1, size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights", shape=[size, out_channel], dtype=tf.float32, trainable=trainable)
            biases = tf.get_variable(name="biases", shape=[out_channel], dtype=tf.float32, trainable=trainable)
            res = tf.matmul(input_data_flat, weights)
            out = tf.nn.relu(tf.nn.bias_add(res, biases))
        self.parameters += [weights, biases]
        return out

    def convlayers(self):
        # zero-mean input
        # conv1
        self.conv1_1 = self.conv("conv1re_1", self.imgs, 64, trainable=False)
        self.conv1_2 = self.conv("conv1_2", self.conv1_1, 64, trainable=False)
        self.pool1 = self.maxpool("poolre1", self.conv1_2, trainable=False)

        # conv2
        self.conv2_1 = self.conv("conv2_1", self.pool1, 128, trainable=False)
        self.conv2_2 = self.conv("convwe2_2", self.conv2_1, 128, trainable=False)
        self.pool2 = self.maxpool("pool2", self.conv2_2, trainable=False)

        # conv3
        self.conv3_1 = self.conv("conv3_1", self.pool2, 256, trainable=False)
        self.conv3_2 = self.conv("convrwe3_2", self.conv3_1, 256, trainable=False)
        self.conv3_3 = self.conv("convrew3_3", self.conv3_2, 256, trainable=False)
        self.pool3 = self.maxpool("poolre3", self.conv3_3, trainable=False)

        # conv4
        self.conv4_1 = self.conv("conv4_1", self.pool3, 512, trainable=False)
        #         self.conv4_2 = self.conv("convrwe4_2",self.conv4_1,512,trainable=False)
        #         self.conv4_3 = self.conv("conv4rwe_3",self.conv4_2,512,trainable=False)
        self.pool4 = self.maxpool("pool4", self.conv4_1, trainable=False)

        # conv5
        self.conv5_1 = self.conv("conv5_1", self.pool4, 512, trainable=False)
        #         self.conv5_2 = self.conv("convrwe5_2",self.conv5_1,512,trainable=False)
        #         self.conv5_3 = self.conv("conv5_3",self.conv5_2,512,trainable=False)
        self.pool5 = self.maxpool("poorwel5", self.conv5_1, trainable=False)

    def fc_layers(self):

        self.fc6 = self.fc("fc6", self.pool5, 2048)
        #         self.fc7 = self.fc("fc7", self.fc6, 4096)
        self.fc8 = self.fc("fc8", self.fc6, 2)

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i not in [30, 31]:
                sess.run(self.parameters[i].assign(weights[k]))
        print("-----------all done---------------")

img_width = 224
img_height = 224


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]
        if letter == 'cat':
            labels = np.append(labels, n_img * [0])
        else:
            labels = np.append(labels, n_img * [1])
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list


def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, img_width, img_height)
    image = tf.image.per_image_standardization(image)  # 将图片标准化
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch


def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


import VGG16_model as model
import create_and_read_TFRecord2 as reader2

if __name__ == '__main__':
    print("111")
    print("222")
    X_train, y_train = reader2.get_file("afterClass/train")  # 输入训练数据路径
    image_batch, label_batch = reader2.get_batch(X_train, y_train, 224, 224, 25, 256)
    x_imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_imgs = tf.placeholder(tf.int32, [None, 2])
    vgg = model.vgg16(x_imgs)
    fc3_cat_and_dog = vgg.probs
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc3_cat_and_dog, labels=y_imgs))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
    pre = tf.nn.softmax(fc3_cat_and_dog)
    correct_pred = tf.equal(tf.argmax(pre, 1), tf.argmax(y_imgs, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    vgg.load_weights('vgg16_weights.npz', sess)  # 输入VGG16权重
    saver = vgg.saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    import time

    start_time = time.time()

    for i in range(200):

        image, label = sess.run([image_batch, label_batch])
        labels = reader2.onehot(label)

        sess.run(optimizer, feed_dict={x_imgs: image, y_imgs: labels})
        if i % 10 == 0:
            loss_record = sess.run(loss, feed_dict={x_imgs: image, y_imgs: labels})
            print("now the loss is %f " % loss_record)
            print(sess.run(accuracy, feed_dict={x_imgs: image, y_imgs: labels}))
            end_time = time.time()
            print('time: ', (end_time - start_time))
            start_time = end_time
            print("----------epoch %d is finished---------------" % i)

    saver.save(sess, "model/")  # 保存模型路径
    print("Optimization Finished!")