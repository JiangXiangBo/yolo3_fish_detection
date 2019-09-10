# coding=utf-8
import xml.etree.ElementTree as ET
import os
import argparse
from sklearn.model_selection import train_test_split


def getClassNameList(classFilePath):
    with open(classFilePath) as file:
        className_list = [k.strip() for k in file.readlines() if k.strip() != '']
    return className_list


def getFilePathList(dirPath, partOfFileName=''):
    allFileName_list = list(os.walk(dirPath))[0][2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list


if __name__ == '__main__':
    classFilePath = 'model_data/voc_classes.txt'
    className_list = getClassNameList(classFilePath)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-dir', type=str, help='path for dataset directory')
    # argument_namespace = parser.parse_args()
    # datasetDirPath = argument_namespace.dir

    datasetDirPath = "images_416x416"
    xmlFilePath_list = getFilePathList(datasetDirPath, '.xml')
    train_xmlFilePath_list, test_xmlFilePath_list = train_test_split(xmlFilePath_list, test_size=0.1)
    dataset_list = [('dataset_train', train_xmlFilePath_list), ('dataset_test', test_xmlFilePath_list)]
    for dataset in dataset_list:
        txtFile_path = '%s.txt' %dataset[0]
        txtFile = open(txtFile_path, 'w')
        for xmlFilePath in dataset[1]:
            jpgFilePath = xmlFilePath.replace('.xml', '.jpg')
            txtFile.write(jpgFilePath)
            with open(xmlFilePath) as xmlFile:
                xmlFileContent = xmlFile.read()
            root = ET.XML(xmlFileContent)
            for obj in root.iter('object'):
                className = obj.find('name').text
                if className not in className_list:
                    print('error!! className not in className_list')
                    continue
                classId = className_list.index(className)
                bndbox = obj.find('bndbox')
                bound = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                         int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                txtFile.write(" " + ",".join([str(k) for k in bound]) + ',' + str(classId))
            txtFile.write('\n')
        txtFile.close()


