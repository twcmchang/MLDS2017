# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
#from six.moves import zip  # pylint: disable=redefined-builtin

#import tensorflow as tf
from lib.tf11_contrib_rnn import core_rnn
from lib.tf11_contrib_rnn import core_rnn_cell
#from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from lib import seq2seq as seq2seq_tf
#from tensorflow.python.framework import dtypes
#from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
#from tensorflow.python.ops import embedding_ops
#from tensorflow.python.ops import math_ops
#from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

def google_mt_seq2seq(encoder_inputs,
                      decoder_inputs,
                      use_lstm,
                      num_encoder_symbols,
                      num_decoder_symbols,
                      embedding_size,
                      num_layers=3,
                      num_heads=1,
                      output_projection=None,
                      feed_previous=False,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):

  def single_cell(size, lstm, residual=False):  
    if use_lstm:
      cell = core_rnn_cell.BasicLSTMCell(size)
    else:
      cell = core_rnn_cell.GRUCell(size)
    if residual:
      return core_rnn_cell.ResidualWrapper(cell)
    else:
      return cell
        
  with variable_scope.variable_scope(
      scope or "google_mt_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.   
    encoder_fw_cell = single_cell(embedding_size, use_lstm)
    encoder_bw_cell = single_cell(embedding_size, use_lstm)
    
    encoder_fw_cell = core_rnn_cell.EmbeddingWrapper(
      encoder_fw_cell, embedding_classes=num_encoder_symbols,
      embedding_size=embedding_size)
    
    encoder_bw_cell = core_rnn_cell.EmbeddingWrapper(
      encoder_bw_cell, embedding_classes=num_encoder_symbols,
      embedding_size=embedding_size)

    bi_encoder_outputs, state_fw, state_bw = core_rnn.static_bidirectional_rnn(
      encoder_fw_cell, encoder_bw_cell, encoder_inputs, dtype=dtype)
    
    encoder_cell = [single_cell(embedding_size, use_lstm)] 
    encoder_cell = encoder_cell + [single_cell(embedding_size, use_lstm, True) for l in xrange(num_layers-2)]
    encoder_cell = core_rnn_cell.MultiRNNCell(encoder_cell)
    
    encoder_outputs, encoder_state = core_rnn.static_rnn(
        encoder_cell, bi_encoder_outputs, dtype=dtype)
    
    encoder_state = (state_fw, ) + encoder_state

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [
        array_ops.reshape(e, [-1, 1, encoder_cell.output_size]) for e in encoder_outputs
    ]
    attention_states = array_ops.concat(top_states, 1)
    
    # Decoder.
    decoder_cell = [single_cell(embedding_size, use_lstm)] 
    decoder_cell = decoder_cell + [single_cell(embedding_size, use_lstm, True) for l in xrange(num_layers-1)]
    decoder_cell = core_rnn_cell.MultiRNNCell(decoder_cell)
      
    output_size = None
    if output_projection is None:
      decoder_cell = core_rnn_cell.OutputProjectionWrapper(decoder_cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      outputs, state = seq2seq_tf.embedding_attention_decoder(
          decoder_inputs,
          encoder_state,
          attention_states,
          decoder_cell,
          num_decoder_symbols,
          embedding_size,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)
      return outputs, state, encoder_state

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = seq2seq_tf.embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            decoder_cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state, encoder_state