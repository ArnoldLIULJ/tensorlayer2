#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops


class Seq2seq(Model):
    def __init__(
            self,
            cell_dec,
            cell_enc,
            seq_step,
            embedding_layer=None,
            is_train=True,
            name="seq2seq"
    ):
        super(Seq2seq, self).__init__(name=name)
        self.embedding_layer = embedding_layer
        self.vocabulary_size = embedding_layer.vocabulary_size
        self.embedding_size = embedding_layer.embedding_size
        self.encoding = encoding
        self.decoding = decoding
        self.num_step = seq_step
        self.encoding_layer = tl.layers.RNN(cell=cell_enc, in_channels=self.embedding_size, return_state=True)
        self.decoding_layer = tl.layers.RNN(cell=cell_dec, in_channels=self.embedding_size)
        self.reshape_layer = tl.layers.Reshape([-1, cell_dec.units])
        self.dense_layer = tl.layers.Dense(n_units=self.vocabulary_size, act=tf.nn.softmax, in_channels=cell_dec.units)
        ### 这个reshape能保证变换可逆吗
        self.reshape_layer_after = tl.layers.Reshape([-1, self.num_step, self.vocabulary_size])
        self.reshape_layer_individual_sequence = tl.layers.Reshape([-1, 1, self.vocabulary_size])
        

    def inference(self, encoding, seq_length, start_token):

        # after embedding the encoding sequence, start the encoding_RNN, then transfer the state to decoing_RNN
        after_embedding_encoding = self.embedding_layer(encoding)
        enc_rnn_ouput, state = self.encoding_layer(after_embedding_encoding, return_state=True)

        
        # for the start_token, first create a batch of it, get[Batchsize, 1]. 
        # then embbeding, get[Batchsize, 1, embeddingsize]
        # then RNN, get[Batchsize, 1, RNN_units]
        # then reshape, get[Batchsize*1, RNN_units]
        # then dense, get[Batchsize*1, vocabulary_size]
        # then reshape, get[Batchsize, 1, vocabulary_size]
        # finally, get Argmax of the last dimension, get next_sequence[Batchsize, 1]
        # this next_sequence will repeat above procedure for the sequence_length time

        batch_size = encoding.shape[0]
        decoding = [[start_token] for i in range(batch_size)]
        decoding = np.array(decoding)
        
        after_embedding_decoding = self.embedding_layer(decoding)
        # 传进两个不同的encoding，在inference模式下，通过encoding_layer层后返回的state是不同的。
        # 但是，将这个不同的state传进decoding_layer后，返回的state相同。猜测---》state没传进去？
        # 在这里打印出来的state不同
        print("state before = ", state)
        feed_output, state = self.decoding_layer(after_embedding_decoding, state=state, return_state=True)
        # 在这里打印出来的state都相同
        print("state after = ", state)
        feed_output = self.reshape_layer(feed_output)
        feed_output = self.dense_layer(feed_output)
        feed_output = self.reshape_layer_individual_sequence(feed_output)
        feed_output = tf.argmax(feed_output, -1)
        
        final_output = feed_output
        
        for i in range(seq_length-1):
            feed_output = self.embedding_layer(feed_output)
            feed_output, state = self.decoding_layer(feed_output, state, return_state=True)
            feed_output = self.reshape_layer(feed_output)
            feed_output = self.dense_layer(feed_output)
            feed_output = self.reshape_layer_individual_sequence(feed_output)
            feed_output = tf.argmax(feed_output, -1)
            final_output = tf.concat([final_output,feed_output], 1)


        return final_output, state


    def forward(self, inputs, seq_length=8, start_token=None, return_state=False):


        if (self.is_train):
            encoding = inputs[0]
            after_embedding_encoding = self.embedding_layer(encoding)
            decoding = inputs[1]
            after_embedding_decoding = self.embedding_layer(decoding)
            enc_rnn_output, state = self.encoding_layer(after_embedding_encoding, return_state=True)
            
            dec_rnn_output, state = self.decoding_layer(after_embedding_decoding, initial_state=state, return_state=True)
            dec_output = self.reshape_layer(dec_rnn_output)
            denser_output = self.dense_layer(dec_output)

            ## reshape into [Batch_size, seq_step, vocabulary_size]
            output = self.reshape_layer_after(denser_output)
        else:
            encoding = inputs
            output, state = self.inference(encoding, seq_length, start_token)

        if (return_state):
            return output, state
        else:
            return output

def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).
  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, default: "sequence_loss_by_example".
  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.
  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(labels=target, logits=logit)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def cross_entropy_seq(logits, target_seqs, batch_size=None):  # , batch_size=1, num_steps=None):
    """Returns the expression of cross-entropy of two sequences, implement
    softmax internally. Normally be used for fixed length RNN outputs, see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`__.

    Parameters
    ----------
    logits : Tensor
        2D tensor with shape of `[batch_size * n_steps, n_classes]`.
    target_seqs : Tensor
        The target sequence, 2D tensor `[batch_size, n_steps]`, if the number of step is dynamic, please use ``tl.cost.cross_entropy_seq_with_mask`` instead.
    batch_size : None or int.
        Whether to divide the cost by batch size.
            - If integer, the return cost will be divided by `batch_size`.
            - If None (default), the return cost will not be divided by anything.

    Examples
    --------
    >>> import tensorlayer as tl
    >>> see `PTB example <https://github.com/tensorlayer/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py>`__.for more details
    >>> input_data = tf.placeholder(tf.int32, [batch_size, n_steps])
    >>> targets = tf.placeholder(tf.int32, [batch_size, n_steps])
    >>> # build the network
    >>> print(net.outputs)
    (batch_size * n_steps, n_classes)
    >>> cost = tl.cost.cross_entropy_seq(network.outputs, targets)

    """
    sequence_loss_by_example_fn = sequence_loss_by_example

    loss = sequence_loss_by_example_fn(
        [logits], [tf.reshape(target_seqs, [-1])], [tf.ones_like(tf.reshape(target_seqs, [-1]), dtype=tf.float32)]
    )
    # [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss)  # / batch_size
    if batch_size is not None:
        cost = cost / batch_size
    return cost


if __name__ == "__main__":
    encoding = np.random.randint(low=0,high=100,size=(50,5))
    decoding = np.random.randint(low=0,high=100,size=(50,8))
    target = np.random.randint(low=0,high=100,size=(50,8))
    model_ = Seq2seq(
        cell_enc=tf.keras.layers.SimpleRNNCell(units=9),
        cell_dec=tf.keras.layers.SimpleRNNCell(units=9), 
        embedding_layer=tl.layers.Embedding(vocabulary_size=100, embedding_size=5),
        seq_step = 8
        )

    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    model_.train()


    for i in range(100):
        
        with tf.GradientTape() as tape:
            ## compute outputs
            output = model_(inputs = [encoding, decoding])
            ## get the max and return shape[Batch_size, seq_step]
            output = tf.reshape(output, [-1, 100])
            ## compute loss and update model
            loss = cross_entropy_seq(output, target)
        ## print(model_.weights)
        grad = tape.gradient(loss, model_.weights)
        #print(grad)

        optimizer.apply_gradients(zip(grad, model_.weights))
        
    
    model_.eval()


    single_sentense = np.reshape(encoding[0], (1, -1))
    prediction = model_(inputs=single_sentense, seq_length=8, start_token=target[0,0])
    print(prediction)



    print("===== ========= ======== =====")
    single_sentense = np.reshape(encoding[1], (1, -1))
    prediction = model_(inputs=single_sentense, seq_length=8, start_token=target[0,0])
    print(prediction)

    
    # prediction = model_(inputs=encoding, seq_length=8, start_token=target[0,0])
    # print(prediction)



    # data_x = np.random.random([2, 3, 4]).astype(np.float32)
    # test_rnn = tl.layers.RNN(cell=tf.keras.layers.SimpleRNNCell(units=10), inputs_shape=(2,3,4))
    # outputs, state = test_rnn.forward(inputs=data_x, return_state=True)
    # outputs = test_rnn.forward(inputs=data_x, initial_state=state)
    
