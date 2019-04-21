import os
import cv2
import keras
import numpy as np
from keras.layers import Input
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


class application:
    def __init__(self):
        self.name = ""
        self.preprocces_func = ""
        self.instance = None
        self.saveloadpath = ""
        self.shape = (224, 224, 3)
        self.input_tensor = Input(shape=self.shape)
        self.application = None
        self.defaultext = ".h5"

    def makemodel(self, name, application, preprocess_func, saveloadpath, shape=(224, 224, 3)):
        self.name = name
        self.preprocces_func = preprocess_func
        self.saveloadpath = saveloadpath
        self.shape = shape
        self.application = application

    def getinstance(self, trainable=False, loadfrom="", cancreate=True):
        if not loadfrom == "":
            instance = self.loadmodelinstance(loadfrom)
        else:
            if cancreate:
                instance = self.application(include_top=False, weights='imagenet', input_tensor=self.input_tensor,
                                            input_shape=self.shape)

            if not trainable:
                instance.trainable = False
                for layer in instance.layers:
                    layer.trainable = False
        self.instance = instance
        return instance

    def preprocess(self, x):
        return self.preprocces_func(x)

    def resizeAndPreprocess(self, x):
        data_upscaled = np.zeros((x.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        for i, img in enumerate(x):
            large_img = cv2.resize(img, dsize=(self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
            data_upscaled[i] = large_img
        return self.preprocess(data_upscaled)

    def loadmodelinstance(self, saveloadpath):
        p = saveloadpath
        if not os.path.exists(saveloadpath):
            if not "." in p:
                p += self.defaultext
        self.instance = load_model(p)
        return self.instance

    def getApplicationByName(self, name, saveloadpath='', hastoload=False):
        res = None
        loaded = False
        for a in self.getall():
            if a.name.lower() == name.lower():
                res = a
                if not saveloadpath == '':
                    res.saveloadpath = saveloadpath
                    if os.path.exists(saveloadpath):
                        res.loadmodelinstance(saveloadpath)
                        loaded = True
        if hastoload and not loaded:
            raise Exception('Model could not be loaded ' + saveloadpath)
        return res

    def getall(self):
        models = {}
        models['VGG19'] = [keras.applications.vgg19.VGG19, keras.applications.vgg19.preprocess_input, (224, 224, 3)]
        models['VGG16'] = [keras.applications.vgg16.VGG16, keras.applications.vgg16.preprocess_input, (224, 224, 3)]

        models['InceptionV3'] = [keras.applications.inception_v3.InceptionV3,
                                 keras.applications.inception_v3.preprocess_input, (299, 299, 3)]
        models['Mobilenet'] = [keras.applications.mobilenet.MobileNet, keras.applications.mobilenet.preprocess_input,
                               (224, 224, 3)]
        models['DenseNet201'] = [keras.applications.densenet.DenseNet201, keras.applications.densenet.preprocess_input,
                                 (224, 224, 3)]

        models['DenseNet121'] = [keras.applications.densenet.DenseNet121, keras.applications.densenet.preprocess_input,
                                 (224, 224, 3)]

        models['DenseNet169'] = [keras.applications.densenet.DenseNet169, keras.applications.densenet.preprocess_input,
                                 (224, 224, 3)]
        models['NASNetMobile'] = [keras.applications.nasnet.NASNetMobile, keras.applications.nasnet.preprocess_input,
                                  (224, 224, 3)]
        models['Xception'] = [keras.applications.xception.Xception, keras.applications.xception.preprocess_input,
                              (299, 299, 3)]
        models['InceptionResNetV2'] = [keras.applications.inception_resnet_v2.InceptionResNetV2,
                                       keras.applications.inception_resnet_v2.preprocess_input, (299, 299, 3)]

        models['ResNet50'] = [keras.applications.resnet50.resnet50,
                              keras.applications.resnet50.preprocess_input, (224, 224, 3)]

        models['NASNetLarge'] = [keras.applications.nasnet.NASNetLarge,
                                 keras.applications.nasnet.preprocess_input, (331, 331, 3)]

        models['MobileNetV2'] = [keras.applications.mobilenetv2.MobileNetV2,
                                 keras.applications.mobilenetv2.preprocess_input, (224, 224, 3)]

        if not os.path.exists('models'):
            os.makedirs('models')
        aps = []
        for k, v in models.items():
            app = application()
            app.input_tensor = Input(shape=v[2])
            app.name = k
            app.application = v[0]
            app.preprocces_func = v[1]
            app.shape = v[2]
            app.saveloadpath = "models/" + k
            aps.append(app)
        return aps

    def inference(self, imagepath, modelpath=''):
        if not modelpath == "":
            self.loadmodelinstance(modelpath)
        if self.instance == None:
            return "error", "error"
        image = cv2.imread(imagepath)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = self.resizeAndPreprocess(image)
        res = self.instance.predict(image)
        am = np.argmax(res)
        return am, res

    def makeinputmulti(self, apps, x):
        input = []
        for m in apps:
            x = m.resizeAndPreprocess(np.copy(x))
            input.append(x)
        return input

    def inference_combined(self, applist, SavePathCombinedModel, imagepath):
        pre = []
        for app in applist:
            pre.append(app.preprocces_func)
        model = load_model(SavePathCombinedModel)
        image = cv2.imread(imagepath)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        x = application().makeinputmulti(pre, image)
        res = model.predict(x)
        am = np.argmax(res)
        return am, res

    def getGenerators(self, imageCollectionPath, batch_size=32, split=0.2):
        datagen = ImageDataGenerator(validation_split=split, shear_range=0.2, horizontal_flip=True)
        train_generator = datagen.flow_from_directory(
            imageCollectionPath,
            target_size=(299, 299),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training'
        )

        test_generator = datagen.flow_from_directory(
            imageCollectionPath,
            target_size=(299, 299),
            batch_size=1000,
            class_mode='categorical',
            shuffle=True,
            subset='validation'
        )
        return train_generator, test_generator

    def AllAvailableModelsAsList(self):
        a = self.getall()
        names = []
        for app in a:
            names.append(app.name)
        return names

    def predictionNumberToLabel(self, dict, nr):
        prediction = "Unknown"
        for k, v in dict.items():
            if v == nr:
                prediction = k
        return prediction

    def Evaluate_one_saved(self, train_generator,savepath):
        return self.Evaluate_one(train_generator,self.loadmodelinstance(savepath))

    def Evaluate_one(self,train_generator):
        x, y = train_generator.next()
        x = self.resizeAndPreprocess(x)
        history = self.instance.evaluate(x, y, verbose=1)
        print(history)
        lastacc = history[1]
        lastloss = history[0]
        return lastacc,lastloss
