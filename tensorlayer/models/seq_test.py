import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer

class CustomisedModel(tl.models.Model):
    def __init__(self):
        super(CustomisedModel, self).__init__()
        self.rnnlayer1 = tl.layers.RNN(
            cell=tf.keras.layers.SimpleRNNCell(units=8, dropout=0.1),
            in_channels=4,
            return_last=True, return_state=True
        )
        self.rnnlayer2 = tl.layers.RNN(
            cell=tf.keras.layers.SimpleRNNCell(units=8, dropout=0.1),
            in_channels=4,
            return_last=True, return_state=True
        )
        self.dense = tl.layers.Dense(in_channels=8, n_units=1)

    def forward(self, x):
        _, state = self.rnnlayer1(x[:,:2,:])
        print("before: ",  state)
        print(x[:,2:,:])
        z, state = self.rnnlayer2(x[:,2:,:], initial_state=state)
        print("after: ", state)
        z = self.dense(z)
        return z


if __name__ == "__main__":
    
    batch_size = 2

    vocab_size = 20
    embedding_size = 4

    hidden_size = 8
    num_steps = 6
    data_x = np.random.random([batch_size, num_steps, embedding_size]).astype(np.float32)
    data_y = np.zeros([batch_size, 1]).astype(np.float32)

    rnn_model = CustomisedModel()
    print(rnn_model)
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    rnn_model.train()
    assert rnn_model.rnnlayer1.is_train
    assert rnn_model.rnnlayer2.is_train

    for epoch in range(50):
        with tf.GradientTape() as tape:
            pred_y = rnn_model(data_x)
            loss = tl.cost.mean_squared_error(pred_y, data_y)

        gradients = tape.gradient(loss, rnn_model.weights)
        optimizer.apply_gradients(zip(gradients, rnn_model.weights))

        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %f" % (epoch, loss))




    rnn_model.eval()
    print("inference mode")
    pred_y_1 = rnn_model(data_x)
    data_x[:,:2,:] = 1
    pred_y_2 = rnn_model(data_x)
    print(pred_y_1, pred_y_2)
