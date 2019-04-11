#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.models import Model
from tensorlayer.layers import Dense, Dropout, Input
from tensorlayer.layers.core import Layer
from cross_entropy_seq_loss import *


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
        # # 传进两个不同的encoding，在inference模式下，通过encoding_layer层后返回的state是不同的。
        # # 但是，将这个不同的state传进decoding_layer后，返回的state相同。猜测---》state没传进去？
        # # 在这里打印出来的state不同
        # print("state before = ")
        # print(state)
        feed_output, state = self.decoding_layer(after_embedding_decoding, initial_state=state, return_state=True)
        # # 在这里打印出来的state都相同
        # print("state after = ")
        #(state)
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



if __name__ == "__main__":
    seq_step_input = 3
    seq_step_output = 5
    vocabulary_size = 10
    batch_size = 100


    encoding = np.random.randint(low=0,high=vocabulary_size,size=(batch_size,seq_step_input))
    decoding = np.random.randint(low=0,high=vocabulary_size,size=(batch_size,seq_step_output))
    target = np.random.randint(low=0,high=10,size=(batch_size,seq_step_output))
    model_ = Seq2seq(
        cell_enc=tf.keras.layers.SimpleRNNCell(units=9),
        cell_dec=tf.keras.layers.SimpleRNNCell(units=9), 
        embedding_layer=tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=5),
        seq_step = seq_step_output
        )
    
    optimizer = tf.optimizers.Adam(learning_rate=0.005)
    model_.train()


    for epoch in range(5000):
        
        with tf.GradientTape() as tape:
            ## compute outputs
            output = model_(inputs = [encoding, decoding])
            ## get the max and return shape[Batch_size, seq_step]
            output = tf.reshape(output, [-1, vocabulary_size])
            ## compute loss and update model
            loss = cross_entropy_seq(output, target)
        
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %f" % (epoch, loss))
        ## print(model_.weights)
        grad = tape.gradient(loss, model_.weights)
        #print(grad)

        optimizer.apply_gradients(zip(grad, model_.weights))
        
    
    model_.eval()


    single_sentense = np.reshape(encoding[0], (1, -1))
    prediction = model_(inputs=single_sentense, seq_length=seq_step_output, start_token=target[0,0])
    print(prediction)
    print(target[0])


    print("===== ========= ======== =====  ===== ========= ======== ===== ===== ========= ======== ===== ")
    single_sentense = np.reshape(encoding[1], (1, -1))
    prediction = model_(inputs=single_sentense, seq_length=seq_step_output, start_token=target[1,0])
    print(target[1])
    print(prediction)

    
    # prediction = model_(inputs=encoding, seq_length=8, start_token=target[0,0])
    # print(prediction)



    # data_x = np.random.random([2, 3, 4]).astype(np.float32)
    # test_rnn = tl.layers.RNN(cell=tf.keras.layers.SimpleRNNCell(units=10), inputs_shape=(2,3,4))
    # outputs, state = test_rnn.forward(inputs=data_x, return_state=True)
    # outputs = test_rnn.forward(inputs=data_x, initial_state=state)
    
