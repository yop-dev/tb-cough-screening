import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
)
from tensorflow.keras.applications import (
    MobileNetV2, MobileNetV3Small, EfficientNetB0, EfficientNetB3,
    ResNet50, InceptionV3, DenseNet121
)

def _build_keras_classifier(base_model_cls, input_shape=(224,224,3), dropout=0.5):
    base = base_model_cls(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def MobileNetV2_Classifier(**kwargs):
    return _build_keras_classifier(MobileNetV2, **kwargs)

def MobileNetV3Small_Classifier(**kwargs):
    return _build_keras_classifier(MobileNetV3Small, **kwargs)

def EfficientNetB0_Classifier(**kwargs):
    return _build_keras_classifier(EfficientNetB0, **kwargs)

def EfficientNetB3_Classifier(**kwargs):
    return _build_keras_classifier(EfficientNetB3, **kwargs)

def ResNet50_Classifier(**kwargs):
    return _build_keras_classifier(ResNet50, **kwargs)

def InceptionV3_Classifier(**kwargs):
    return _build_keras_classifier(InceptionV3, **kwargs)

def DenseNet121_Classifier(**kwargs):
    return _build_keras_classifier(DenseNet121, **kwargs)
