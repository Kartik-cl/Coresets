import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input,Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,LeakyReLU
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import shutil

class create_model:
    def __init__(self,params):
        self.exp_name = params[0]
        self.model_arch = params[1]
        self.num_classes = params[2]
        self.train_path = params[3]["train"]
        self.val_path = params[3]["val"]
        self.test_path = params[3]["test"]
        self.saved_model_path = params[4]["saved_model_path"]
        self.epochs = params[4]["epochs"]
        self.batch_size = params[4]["batch_size"]
        self.early_stopping_flag = params[4]["early_stopping"]
        self.image_shape = None

        if os.path.exists(self.saved_model_path) and os.path.isdir(self.saved_model_path):
            shutil.rmtree(self.saved_model_path)
        os.makedirs(os.path.join(self.saved_model_path), mode = 0o777)
    
    def plot_loss_curves(self,history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        epochs = range(len(history.history['loss']))

        # Plot loss
        plt.plot(epochs, loss, label='training_loss')
        plt.plot(epochs, val_loss, label='val_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.savefig(self.saved_model_path+"/loss_curve_"+str(self.exp_name)+"_e"+str(self.epochs)+"_b"+str(self.batch_size)+".png")

        # Plot accuracy
        plt.figure()
        plt.plot(epochs, accuracy, label='training_accuracy')
        plt.plot(epochs, val_accuracy, label='val_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.savefig(self.saved_model_path+"/accuracy_curve_"+self.exp_name+"_e"+str(self.epochs)+"_b"+str(self.batch_size)+".png")

    def create_data_generators(self):
        train_datagen=ImageDataGenerator(rescale=1/255.)
        val_datagen=ImageDataGenerator(rescale=1/255.)
        test_datagen=ImageDataGenerator(rescale=1/255.)

        train_data=train_datagen.flow_from_directory(self.train_path,
                                                    target_size=(self.image_shape,self.image_shape),
                                                    batch_size=self.batch_size,
                                                    class_mode='categorical')

        val_data=val_datagen.flow_from_directory(self.val_path,
                                                target_size=(self.image_shape,self.image_shape),
                                                batch_size=self.batch_size,
                                                class_mode='categorical')

        test_data=test_datagen.flow_from_directory(self.test_path,
                                                target_size=(self.image_shape,self.image_shape),
                                                batch_size=self.batch_size,
                                                class_mode='categorical')
        return train_data, val_data, test_data

    def init_model(self,head):
        act = 'relu'

        model = Sequential()
        model.add(head)
        model.add(MaxPooling2D(pool_size=(2,2)))    
        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation(act))
        # model.add(Dropout(0.25))
        model.add(Dense(64))

        model.add(BatchNormalization())
        model.add(Activation(act))
        # model.add(Dropout(0.25))
        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def train_the_model(self,model,train_data,val_data):
        # setting callbacks
        if self.early_stopping_flag:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=10)
            callback_val = [es] 
        else:
            callback_val = None

        history  = model.fit(train_data,
                    epochs=self.epochs,
                    steps_per_epoch=len(train_data),
                    validation_data=val_data,
                    validation_steps=len(val_data),
                    callbacks=callback_val
        )

        return history

    def make_model(self):
        # Model selection and initialization
        if self.model_arch == 'vgg':
            self.image_shape = 224
            input = Input(shape=(self.image_shape, self.image_shape, 3))
            head = VGG16(include_top=False, input_tensor= input)

        elif self.model_arch == 'inception':
            self.image_shape = 299
            input = Input(shape=(self.image_shape, self.image_shape, 3))
            head = InceptionV3(include_top=False, input_tensor= input)
            
        elif self.model_arch == 'resnet':
            self.image_shape = 224
            input = Input(shape=(self.image_shape, self.image_shape, 3))
            head = ResNet50(include_top=False, input_tensor= input)

        # Creating data generators to load images
        train_data, val_data, test_data = self.create_data_generators()

        # Making model
        model = self.init_model(head)
        
        # Model compilation
        model.compile(optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])
        
        print(model.summary())  

        history = self.train_the_model(model, train_data, val_data)
        

        #model.save(self.saved_model_path+"/"+self.exp_name+"_e"+str(self.epochs)+"_b"+str(self.batch_size)+".h5")
        print("\n-----------------------------------")
        print("Test Loss and Accuracy: ",model.evaluate(test_data))

        self.plot_loss_curves(history)
        