import tensorflow as tf
import numpy as np

from tensorflow_addons.metrics import F1Score


class BioF1(F1Score):
    """
    A specific implementation of the F1-score that does not distinguish between I-class and B-class, as this leads to a
    double penalization of errors. E.g. OOBIIOO vs. OOOBIOO is actually one error, but would be seen as two by a naive
    F1-score.
    """
    def __init__(self, num_categories=None, num_classes=None, name='bio_f1_score', **kwargs):
        assert (num_categories is not None or num_classes is not None) and (num_categories is None or num_classes is None)
        num_categories = num_categories or (num_classes - 2)//2

        super().__init__(num_classes=num_categories + 1, name=name, **kwargs)
        self.matrix = np.zeros([2 + 2 * num_categories, num_categories + 1], dtype='float32')
        for i in range(num_categories + 1):
            self.matrix[2*i:2+2*i, i] = 1.

    def update_state(self, y_true, y_pred, **kwargs):
        y_true_relevant = tf.matmul(y_true, self.matrix)
        y_pred_relevant = tf.matmul(y_pred, self.matrix)
        y_true_relevant = tf.reshape(y_true_relevant, [-1, self.num_classes])
        y_pred_relevant = tf.reshape(y_pred_relevant, [-1, self.num_classes])
        super().update_state(y_true_relevant, y_pred_relevant, **kwargs)


if __name__ == "__main__":
    num_classes = 6
    num_samples = 10

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    bio_f1 = BioF1(num_classes=num_classes)
    f1 = F1Score(num_classes=num_classes)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy(), bio_f1, f1])

    data = np.random.random((num_samples, 32))
    labels = np.zeros((num_samples, num_classes))
    for row in labels:
        row[np.random.randint(2)] = 1

    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32)

    model.fit(dataset, epochs=2)
    result = model.predict(data)
    print(model.metrics[2](labels, result))
    print(model.metrics[3](labels, result))
    pass
