# Exploring the Effects of Outcome of Each Decoder Layer of a Large Language Model for Text Generation
======
The purpose of this notebook is to show effects of layers on generation. More precisely, the goal is to identify, as the number of layer progresses, how the
generated outcome improves. With a set of practical experiments, you will have experience on the effect
of each layer of an LLM and eventually documenting your working process.
## Background
======
The decoder section of the Transformer architecture, as shown in the below diagram, consists of several
components that are stacked N times to form the full decoder. Each block of the decoder contains the
following components:
1. Masked Multi-Head Attention: This block receives the "Outputs" (which are embeddings of the
target sequence that have been shifted right), and it is "masked" to ensure that the predictions
for a certain position can only depend on the known outputs at positions before it. This is critical
in tasks like translation, where during training, the model shouldn't have access to the future
tokens it's trying to predict.
2. Add & Norm: After the masked multi-head attention, the attention output is added to the original
input through a residual connection, then layer normalization is applied. This step helps to
stabilize the learning process and allows for deeper networks.
3. Multi-Head Attention: This layer involves attention mechanisms that operate in parallel. It's
different from the masked multi-head attention in that it's not masked, and it attends to the entire
input sequence. Here, the decoder is allowed to attend to all positions in the input sequence. This
part is also known as "encoder-decoder attention" because the queries come from the previous
decoder layer, and the keys and values come from the encoder's output.
4. Add & Norm: Similar to the previous, the output of the multi-head attention is added back to its
input (residual connection) and normalized.
5. Feed Forward: This is a position-wise feed-forward network that consists of two linear
transformations with a ReLU activation in between. This network is applied to each position
separately and identically.
6. Add & Norm: Once again, the output of the feed-forward network is added back to its input and
normalized.
Language Head
• Linear Layer or Vocabulary Head: This layer is a simple linear transformation that projects the
decoder's output to a higher-dimensional space that is typically the size of the vocabulary.
• SoftMax: The final layer applies the softmax function to the output of the linear layer to obtain
the probabilities of each token in the vocabulary. The highest probability token is typically chosen
as the output at each position in the sequence.

In most transformer-based models, including LLaMA, the decoder processes the input sequentially, layer
by layer. Each layer of the decoder produces a set of hidden states that are passed to the next layer, and
the final layer's output is used to compute probabilities for the next token in the sequence. The log
likelihood of a sequence given a model can be computed by summing the log probabilities of each token
in the sequence, according to the model's predictions.


