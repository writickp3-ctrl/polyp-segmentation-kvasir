import os
import numpy as np
import cv2
import tensorflow as tf
import gradio as gr
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

IMG_SIZE = 352
BASE_FILTERS = 32
MODEL_PATH = "best_model.h5"
THRESHOLD = 0.5

tf.config.optimizer.set_jit(False)


def squeeze_excite(x, ratio=8):
    filters = int(x.shape[-1])
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, filters))(se)
    se = Dense(max(filters // ratio, 1), activation="relu", use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", use_bias=False)(se)
    return Multiply()([x, se])


def residual_block(x, filters, strides=1):
    shortcut = x

    if int(x.shape[-1]) != filters or strides != 1:
        shortcut = Conv2D(
            filters, 1, strides=strides, padding="same",
            use_bias=False, kernel_initializer="he_normal"
        )(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters, 3, strides=strides, padding="same",
        use_bias=False, kernel_initializer="he_normal"
    )(x)

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(
        filters, 3, padding="same",
        use_bias=False, kernel_initializer="he_normal"
    )(x)

    x = squeeze_excite(x)
    return Add()([x, shortcut])


def resize_like(source, target, channels):
    target_size = Lambda(lambda t: tf.shape(t)[1:3])(target)

    target_h = int(target.shape[1]) if target.shape[1] is not None else None
    target_w = int(target.shape[2]) if target.shape[2] is not None else None

    return Lambda(
        lambda args: tf.image.resize(args[0], args[1], method="bilinear"),
        output_shape=(target_h, target_w, channels)
    )([source, target_size])


def aspp_block(x, filters):
    b0 = Conv2D(filters, 1, padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    def dilated(inp, rate):
        y = Conv2D(
            filters, 3, padding="same", dilation_rate=rate,
            use_bias=False, kernel_initializer="he_normal"
        )(inp)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        return y

    b1 = dilated(x, 1)
    b2 = dilated(x, 3)
    b3 = dilated(x, 6)
    b4 = dilated(x, 12)

    bg = GlobalAveragePooling2D()(x)
    bg = Reshape((1, 1, int(x.shape[-1])))(bg)
    bg = Conv2D(filters, 1, padding="same", use_bias=False)(bg)
    bg = BatchNormalization()(bg)
    bg = Activation("relu")(bg)
    bg = resize_like(bg, x, filters)

    x = Concatenate(axis=-1)([b0, b1, b2, b3, b4, bg])
    x = Conv2D(filters, 1, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def attention_gate(g, x, filters):
    theta_x = Conv2D(filters, 1, padding="same", use_bias=False)(x)
    phi_g = Conv2D(filters, 1, padding="same", use_bias=False)(g)

    phi_g = resize_like(phi_g, theta_x, filters)

    add = Add()([theta_x, phi_g])
    add = Activation("relu")(add)

    psi = Conv2D(1, 1, padding="same", use_bias=False)(add)
    psi = Activation("sigmoid")(psi)

    return Multiply()([x, psi])


def decoder_block(x, skip, filters):
    x = Conv2DTranspose(
        filters, 2, strides=2, padding="same",
        kernel_initializer="he_normal"
    )(x)

    skip = attention_gate(g=x, x=skip, filters=max(filters // 2, 1))

    x = Concatenate(axis=-1)([x, skip])
    x = residual_block(x, filters)
    return x


def build_resunet_plus_plus(input_shape=(352, 352, 3), base_filters=32):
    f = [base_filters * (2 ** i) for i in range(5)]

    inputs = Input(input_shape, name="input")

    x = Conv2D(
        f[0], 3, padding="same", use_bias=False,
        kernel_initializer="he_normal"
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    e1 = residual_block(x, f[0])
    p1 = MaxPool2D(pool_size=(2, 2))(e1)

    e2 = residual_block(p1, f[1])
    p2 = MaxPool2D(pool_size=(2, 2))(e2)

    e3 = residual_block(p2, f[2])
    p3 = MaxPool2D(pool_size=(2, 2))(e3)

    e4 = residual_block(p3, f[3])
    p4 = MaxPool2D(pool_size=(2, 2))(e4)

    bridge = aspp_block(p4, f[4])

    d4 = decoder_block(bridge, e4, f[3])
    d3 = decoder_block(d4, e3, f[2])
    d2 = decoder_block(d3, e2, f[1])
    d1 = decoder_block(d2, e1, f[0])

    out_main = Conv2D(
        1, 1, activation="sigmoid", name="output",
        dtype="float32"
    )(d1)

    out_d3 = Conv2D(1, 1, activation="sigmoid", dtype="float32")(d3)
    out_d3 = tf.keras.layers.Resizing(
        input_shape[0], input_shape[1],
        interpolation="bilinear",
        name="ds3"
    )(out_d3)

    out_d2 = Conv2D(1, 1, activation="sigmoid", dtype="float32")(d2)
    out_d2 = tf.keras.layers.Resizing(
        input_shape[0], input_shape[1],
        interpolation="bilinear",
        name="ds2"
    )(out_d2)

    return Model(inputs, [out_main, out_d3, out_d2], name="ResUNet_plusplus")


print("Building model architecture...")
model = build_resunet_plus_plus(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    base_filters=BASE_FILTERS
)

print("Loading weights from best_model.h5...")
model.load_weights(MODEL_PATH)
print("Weights loaded successfully.")


def predict(image):
    if image is None:
        return None, None

    image = np.array(image.convert("RGB"))
    original_h, original_w = image.shape[:2]

    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)

    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    prob = np.squeeze(pred).astype(np.float32)
    mask = (prob > THRESHOLD).astype(np.uint8) * 255

    mask_original = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255

    mask_bool = mask_original > 127
    overlay[mask_bool] = (
        overlay[mask_bool] * 0.55 + red[mask_bool] * 0.45
    ).astype(np.uint8)

    return Image.fromarray(mask_original), Image.fromarray(overlay)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Input Colonoscopy Image"),
    outputs=[
        gr.Image(type="pil", label="Binary Mask"),
        gr.Image(type="pil", label="Overlay"),
    ],
    title="Polyp Segmentation - ResUNet++",
    description="Upload a colonoscopy image to segment likely polyp regions.",
    allow_flagging="never"
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
