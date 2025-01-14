import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.metrics as metrics


def ge_layer(x_in, c, e=6, stride=1):
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), padding="same")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if stride == 2:
        x = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)

        y = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), strides=2, padding="same")(x_in)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2D(filters=c, kernel_size=(1, 1), padding="same")(y)
        y = layers.BatchNormalization()(y)
    else:
        y = x_in

    x = layers.DepthwiseConv2D(depth_multiplier=e, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=c, kernel_size=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, y])
    x = layers.Activation("relu")(x)
    return x


def stem(x_in, c):
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), strides=2, padding="same")(x_in)
    x = layers.BatchNormalization()(x)
    x_split = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=c // 2, kernel_size=(1, 1), padding="same")(x_split)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=c, kernel_size=(3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    y = layers.MaxPooling2D()(x_split)

    x = layers.Concatenate()([x, y])
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


def detail_conv2d(x_in, c, stride=1, k=3):
    x = layers.Conv2D(filters=c, kernel_size=(k, k), strides=stride, padding="same")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def context_embedding(x_in, c):
    x = layers.GlobalAveragePooling2D()(x_in)
    x = layers.BatchNormalization()(x)

    x = layers.Reshape((1, 1, c))(x)

    x = layers.Conv2D(filters=c, kernel_size=(1, 1), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # broadcasting no needed

    x = layers.Add()([x, x_in])
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), padding="same")(x)
    return x


def bilateral_guided_aggregation(detail, semantic, c):
    # detail branch
    detail_a = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same")(detail)
    detail_a = layers.BatchNormalization()(detail_a)

    detail_a = layers.Conv2D(filters=c, kernel_size=(1, 1), padding="same")(detail_a)

    detail_b = layers.Conv2D(filters=c, kernel_size=(3, 3), strides=2, padding="same")(detail)
    detail_b = layers.BatchNormalization()(detail_b)

    detail_b = layers.AveragePooling2D((3, 3), strides=2, padding="same")(detail_b)

    # semantic branch
    semantic_a = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same")(semantic)
    semantic_a = layers.BatchNormalization()(semantic_a)

    semantic_a = layers.Conv2D(filters=c, kernel_size=(1, 1), padding="same")(semantic_a)
    semantic_a = layers.Activation("sigmoid")(semantic_a)

    semantic_b = layers.Conv2D(filters=c, kernel_size=(3, 3), padding="same")(semantic)
    semantic_b = layers.BatchNormalization()(semantic_b)

    semantic_b = layers.UpSampling2D((4, 4), interpolation="bilinear")(semantic_b)
    semantic_b = layers.Activation("sigmoid")(semantic_b)

    # combining
    detail = layers.Multiply()([detail_a, semantic_b])
    semantic = layers.Multiply()([semantic_a, detail_b])

    # this layer is not mentioned in the paper !?
    # semantic = layers.UpSampling2D((4,4))(semantic)
    semantic = layers.UpSampling2D((4, 4), interpolation="bilinear")(semantic)

    x = layers.Add()([detail, semantic])
    x = layers.Conv2D(filters=c, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)

    return x


def seg_head(x_in, c_t, s, n):
    x = layers.Conv2D(filters=c_t, kernel_size=(3, 3), padding="same")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters=n, kernel_size=(3, 3), padding="same")(x)
    x = layers.UpSampling2D((s, s), interpolation="bilinear")(x)

    return x


def semantic_branch(x_in, l):  # noqa: E741
    # semantic branch
    # S1 + S2
    x = stem(x_in, 64 // l)

    # S3
    x = ge_layer(x, 128 // l, stride=2)
    x = ge_layer(x, 128 // l, stride=1)

    # S4
    x = ge_layer(x, 64, stride=2)
    x = ge_layer(x, 64, stride=1)

    # S5
    x = ge_layer(x, 128, stride=2)

    x = ge_layer(x, 128, stride=1)
    x = ge_layer(x, 128, stride=1)
    x = ge_layer(x, 128, stride=1)

    x = context_embedding(x, 128)
    return x


def detail_branch(x_in):
    y = detail_conv2d(x_in, 64, stride=2)
    y = detail_conv2d(y, 64, stride=1)

    # S2
    y = detail_conv2d(y, 64, stride=2)
    y = detail_conv2d(y, 64, stride=1)
    y = detail_conv2d(y, 64, stride=1)

    # S3
    y = detail_conv2d(y, 128, stride=2)
    y = detail_conv2d(y, 128, stride=1)
    y = detail_conv2d(y, 128, stride=1)
    return y


def instance_segmentation_branch(x_in, embed_dim):
    x = detail_conv2d(x_in, 64)
    x = layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = layers.UpSampling2D((8, 8), interpolation="bilinear")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters=embed_dim, kernel_size=(1, 1), strides=1, padding="same", name="instance")(x)
    return x


def binary_segmentation_branch(x_in, num_classes):
    x = detail_conv2d(x_in, 64)
    x = detail_conv2d(x_in, 128, k=1)
    x = layers.Conv2D(filters=1, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = layers.UpSampling2D((8, 8), interpolation="bilinear")(x)
    x = layers.Activation("sigmoid")(x)
    x = layers.Reshape((512, 1024), name="binary")(x)
    return x


class ArgmaxMeanIOU(metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def loss_instance_individual(label, output):
    delta_v = 0.5
    delta_d = 3.0
    param_var = 1.0
    param_dist = 1.0
    param_reg = 0.001
    param_scale = 1.0
    correct_label = tf.reshape(label, [label.shape[1] * label.shape[0]])
    reshaped_pred = tf.reshape(output, [output.shape[1] * output.shape[0], output.shape[2]])
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)
    segmented_sum = tf.math.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)
    mu = tf.math.divide(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1, ord=1)
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0.0, distance)
    distance = tf.square(distance)

    l_var = tf.math.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.math.divide(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.math.divide(l_var, tf.cast(num_instances, tf.float32))

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances * num_instances, output.shape[2]))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)
    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)
    mu_norm = tf.norm(mu_diff_bool, axis=1, ord=1)
    mu_norm = tf.subtract(2.0 * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0.0, mu_norm)
    mu_norm = tf.square(mu_norm)
    l_dist = tf.reduce_mean(mu_norm)
    l_reg = tf.reduce_mean(tf.norm(mu, axis=1, ord=1))
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg
    loss = param_scale * (l_var + l_dist + l_reg)
    return loss, l_var, l_dist, l_reg


def loss_instance(y_true, y_pred):
    def cond(label, batch, out_loss, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, i):
        disc_loss, l_var, l_dist, l_reg = loss_instance_individual(y_true[i], y_pred[i])
        out_loss = out_loss.write(i, disc_loss)
        return label, batch, out_loss, i + 1

    output_ta_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    _, _, out_loss_op, _ = tf.while_loop(cond, body, [y_true, y_pred, output_ta_loss, 0])

    out_loss_op = out_loss_op.stack()
    disc_loss = tf.reduce_mean(out_loss_op)
    return disc_loss
