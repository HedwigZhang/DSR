##############################
# the main function, which will show the training process
##############################
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, History
from tensorflow.contrib.tpu.python.tpu import keras_support

from ResNet import *
from load_data import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import set_session
import pickle, os, time

graph = tf.get_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def lr_scheduler(epoch):
    lr = 0.01
    if epoch >= 20: lr /= 10.0
    if epoch >= 30: lr /= 10.0
    if epoch >= 40: lr /= 10.0
    return lr

def train():
    X_train, y_train = get_train_data()
    X_test, y_test = get_valid_data()
    # data generater
    train_gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
    test_gen = ImageDataGenerator(rescale=1.0/255)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # load network
    model = resnet50()
    #model.compile(Adam(0.001), "categorical_crossentropy", ["accuracy"])
    #model.compile(SGD(0.01, momentum = 0.9), "categorical_crossentropy", ["acc"])
    model.compile(SGD(0.01, momentum = 0.9), "categorical_crossentropy", ["acc", "top_k_categorical_accuracy"])
    model.summary()
    
    # set GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    set_session(session)

    # set
    batch_size = 128
    scheduler = LearningRateScheduler(lr_scheduler)
    hist = History()

    start_time = time.time()

    model.fit_generator(train_gen.flow(X_train, y_train, batch_size, shuffle=True),
                        steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=test_gen.flow(X_test, y_test, batch_size, shuffle=False),
                        validation_steps=X_test.shape[0]//batch_size,
                        callbacks=[scheduler, hist], max_queue_size=5, epochs=50)

    elapsed = time.time() - start_time
    print('training time', elapsed)
    
    history = hist.history
    history["elapsed"] = elapsed

    with open("resnet50_2_DS_with_res_no_bias.pkl", "wb") as fp:
        pickle.dump(history, fp)
    
if __name__ == "__main__":
    train()