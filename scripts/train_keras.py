import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Import model builders
from models.keras.base_models import (
    MobileNetV2_Classifier,
    MobileNetV3Small_Classifier,
    EfficientNetB0_Classifier,
    EfficientNetB3_Classifier,
    ResNet50_Classifier,
    InceptionV3_Classifier,
    DenseNet121_Classifier
)

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_npy(path, label, search_dirs):
    fname = path.numpy().decode('utf-8')
    for base in search_dirs:
        candidate = os.path.join(base, fname)
        if os.path.exists(candidate):
            arr = np.load(candidate)
            return arr.astype(np.float32), np.int32(label)
    raise FileNotFoundError(f"Couldn’t find {fname} in {search_dirs}")


def tf_load_npy(path, label, preprocess_fn, search_dirs, img_size):
    mel, lbl = tf.py_function(
        load_npy, [path, label, search_dirs], [tf.float32, tf.int32]
    )
    mel.set_shape([img_size[0], img_size[1]])
    lbl.set_shape([])
    # Expand and repeat to RGB
    mel = mel[..., tf.newaxis]
    mel = tf.repeat(mel, 3, axis=-1)
    # Scale and preprocess
    mel = mel * 255.0
    mel = preprocess_fn(mel)
    return mel, lbl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=[
        'mnet2','mnet3','effb0','effb3','res50','incep3','dnet121'
    ], required=True)
    parser.add_argument('--train-files', required=True)
    parser.add_argument('--train-labels', required=True)
    parser.add_argument('--val-files', required=True)
    parser.add_argument('--val-labels', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dirs', nargs='+', required=True,
                        help='Directories to search for .npy specs')
    parser.add_argument('--img-size', nargs=2, type=int, default=[224,224])
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()

    # Load splits
    train_files = np.load(args.train_files, allow_pickle=True)
    train_labels = np.load(args.train_labels, allow_pickle=True)
    val_files = np.load(args.val_files, allow_pickle=True)
    val_labels = np.load(args.val_labels, allow_pickle=True)

    # Select preprocess function and model builder
    if args.model == 'mnet2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        ModelBuilder = MobileNetV2_Classifier
    elif args.model == 'mnet3':
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        ModelBuilder = MobileNetV3Small_Classifier
    elif args.model == 'effb0':
        from tensorflow.keras.applications.efficientnet import preprocess_input
        ModelBuilder = EfficientNetB0_Classifier
    elif args.model == 'effb3':
        from tensorflow.keras.applications.efficientnet import preprocess_input
        ModelBuilder = EfficientNetB3_Classifier
    elif args.model == 'res50':
        from tensorflow.keras.applications.resnet import preprocess_input
        ModelBuilder = ResNet50_Classifier
    elif args.model == 'incep3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        ModelBuilder = InceptionV3_Classifier
    elif args.model == 'dnet121':
        from tensorflow.keras.applications.densenet import preprocess_input
        ModelBuilder = DenseNet121_Classifier

    # Build tf.data pipelines
    AUTOTUNE = tf.data.AUTOTUNE
    img_size = tuple(args.img_size)

    def make_ds(files, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((files, labels))
        if shuffle:
            ds = ds.shuffle(len(files), seed=SEED)
        ds = ds.map(lambda f, l: tf_load_npy(f, l, preprocess_input,
                                            args.data_dirs, img_size),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.batch(args.batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(train_files, train_labels, shuffle=True)
    val_ds   = make_ds(val_files, val_labels)

    # Instantiate and compile model
    model = ModelBuilder(input_shape=(*img_size,3), dropout=0.5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # Callbacks and training
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"best_{args.model}.h5")
    checkpoint = ModelCheckpoint(ckpt_path, save_best_only=True,
                                 monitor='val_auc', mode='max')
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=args.epochs,
                        callbacks=[checkpoint])

    # Save history
    hist_dict = history.history
    with open(os.path.join(args.output_dir, f"history_{args.model}.json"), 'w') as f:
        json.dump(hist_dict, f)

    print(f"Training complete. Model saved to {ckpt_path}")
