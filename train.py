import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data


def getClassNameList(classFilePath):
    with open(classFilePath) as file:
        className_list = [k.strip() for k in file.readlines() if k.strip() != '']
    return className_list


def getAnchorList(anchorFilePath):
    with open(anchorFilePath) as file:
        anchor_list = [float(k) for k in file.read().split(',')]
    return np.array(anchor_list).reshape(-1, 2)


def main():
    classeFilePath = 'model_data/voc_classes.txt'
    anchorFilePath = 'model_data/yolo_anchors.txt'
    className_list = getClassNameList(classeFilePath)
    anchor_list = getAnchorList(anchorFilePath)

    # multiple of 32, height and width
    input_shape = (416,416)
    model = create_model(input_shape, anchor_list, len(className_list))
    annotationFilePath = 'dataset_train.txt'
    train(model, annotationFilePath, input_shape, anchor_list, len(className_list))


def create_model(input_shape,
                 anchor_list,
                 num_classes,
                 load_pretrained=True,
                 freeze_body=False,
                 weights_path='saved_model/trained_weights.h5'):
    """

    :param input_shape: 输入图片的尺寸，默认是(416, 416)
    :param anchor_list: 默认的9种anchor box，结构是(9, 2)
    :param num_classes: 类别个数。在网络中，类别值按0~n排列，同时，输入数据的类别也是用索引表示；
    :param load_pretrained:是否使用预训练权重
    :param freeze_body: 冻结模式，1或2。其中，1是冻结DarkNet53网络中的层，2是只保留最后3个1x1的卷积层，其余层全部冻结
    :param weights_path:
    :return:
    """

    # get a new session
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    height, width = input_shape
    num_anchors = len(anchor_list)   # 9

    """
       通过循环，创建3个Input层的列表，作为y_true,其张量（Tensor）结构，如下：
       Tensor("input_2:0", shape=(?, 13, 13, 3, 7), dtype=float32)
       Tensor("input_3:0", shape=(?, 26, 26, 3, 7), dtype=float32)
       Tensor("input_4:0", shape=(?, 52, 52, 3, 7), dtype=float32)
       其中，在真值y_true中，第1位是输入的样本数，第2~3位是特征图的尺寸，如13x13，
       第4位是每个图中的anchor数，第5位是：类别(n)+4个框值(xywh)+框的置信度(是否含有物体)
       """
    y_true = [Input(shape=(height // k,
                           width // k,
                           num_anchors // 3,
                           num_classes + 5)) for k in [32, 16, 8]]

    """
    通过传入，输入Input层image_input、每个尺度的anchor数num_anchors//3
    和类别数num_classes，构建YOLO v3的网络yolo_body
    在model_body中，最终的输入是image_input--(?, 416, 416, 3)
    最终的输出output是3个矩阵的列表
    [(?, 13, 13, 3,(2+5)), (?, 26, 26, 3, 7), (?, 52, 52, 3, 7)]
    """
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained and os.path.exists(weights_path):
        # 加载模型权重
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            # freeze_body=False 所以不冻结，重头训练
            num = len(model_body.layers)-7
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    """
        构建模型的损失层model_loss，其内容如下：
        Lambda是Keras的自定义层，输入为model_body.output和y_true，输出output_shape是(1,)，即一个损失值；
        自定义Lambda层的名字name为yolo_loss；
        层的参数是锚框列表anchors、类别数num_classes和IoU阈值ignore_thresh。
        其中，ignore_thresh用于在物体置信度损失（object confidence loss）中过滤IoU较小的框
        """
    model_loss = Lambda(yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss',
                        arguments={'anchors': anchor_list,
                                   'num_classes': num_classes,
                                   'ignore_thresh': 0.5
                                   }
                        )([*model_body.output, *y_true])

    """
    构建完整的算法模型，步骤如下：
    模型的输入层：model_body的输入（即image_input）和真值y_true；
    模型的输出层：自定义的model_loss层，其输出是一个损失值(None,1)；
    model_body.input是任意(?)个(416,416,3)的图片；y_true是已标注数据所转换的真值结构。
    """
    model = Model([model_body.input, *y_true], model_loss)
    return model


def train(model,
          annotationFilePath,
          input_shape,
          anchor_list,
          num_classes,
          logDirPath='saved_model/'):

    model.compile(optimizer='adam',
                  loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 2 * num_classes
    val_split = 0.05

    with open(annotationFilePath) as file:
        lines = file.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 在训练中，模型调用fit_generator方法，按批次创建数据，输入模型，进行训练
    model.fit_generator(
        data_generator(lines[:num_train], batch_size, input_shape, anchor_list, num_classes),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchor_list, num_classes),
        validation_steps=max(1, num_val // batch_size),
        epochs=200,
        initial_epoch=0)

    # when model training finished, save model
    if not os.path.isdir(logDirPath):
        os.makedirs(logDirPath)
    model_savedPath = os.path.join(logDirPath, 'trained_weights.h5')
    model.save_weights(model_savedPath)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """
    annotation_lines：标注数据的行，每行数据包含图片路径，和框的位置信息
    在第0次时，将数据洗牌shuffle，调用get_random_data解析annotation_lines[i]，
    生成图片image和标注框box，添加至各自的列表image_data和box_data中。
    """
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):  # 4
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        """  
        索引值递增i+1，当完成n个一轮之后，重新将i置0，再次调用shuffle洗牌数据。
        将image_data和box_data都转换为np数组，其中：
        image_data: (4, 416, 416, 3)
        box_data: (4, 20, 5) # 每个图片最多含有20个框
        """
        image_data = np.array(image_data)
        box_data = np.array(box_data)

        """
        将框的数据box_data、输入图片尺寸input_shape、anchor box列表anchors和类别数num_classes
        转换为真值y_true，其中y_true是3个预测特征的列表：
        [(4, 13, 13, 3, 7), (4, 26, 26, 3, 7), (4, 52, 52, 3, 7)]
        """
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)

        # 最终输出：图片数据image_data、真值y_true、每个图片的损失值np.zeros
        yield [image_data, *y_true], np.zeros(batch_size)


if __name__ == '__main__':
    main()
