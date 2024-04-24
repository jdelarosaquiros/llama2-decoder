# Exploring the Effects of Outcome of Each Decoder Layer of a Large Language Model for Text Generation

The purpose of this notebook is to show effects of layers on generation. More precisely, the goal is to identify, as the number of layer progresses, how the
generated outcome improves. With a set of practical experiments, you will have experience on the effect
of each layer of an LLM and eventually documenting your working process.

## Usage

The package requirements are in the first cell of the notebook. If you wish to test a different model, specify its name in the Load Model Section.

## Background

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

## Premature Layers (Early Exit Layers) vs. Mature Layer (Last Layer)
| Sample 1  | Sample 2 |
| ------------- | ------------- |
| ![image](https://github.com/jdelarosaquiros/llama2-decoder/assets/86489701/a39f008f-1018-4eef-bf5f-72ebbaa2781f) | ![image](https://github.com/jdelarosaquiros/llama2-decoder/assets/86489701/ca83cb12-6c32-4108-802f-7a0c75be7b6e) |
| ![image](https://github.com/jdelarosaquiros/llama2-decoder/assets/86489701/088cf9d2-aabb-432d-841b-9b32ed4154da) | ![image](https://github.com/jdelarosaquiros/llama2-decoder/assets/86489701/997e4850-8cbd-4c93-88e9-3e3a25ea32d7) |
| ![image](https://github.com/jdelarosaquiros/llama2-decoder/assets/86489701/fcb216fc-7520-483f-aa07-6a3af9eeb3d8) | ![image](https://github.com/jdelarosaquiros/llama2-decoder/assets/86489701/d8025664-76c7-4fdc-8883-feec9817ab6c) |
| ![image](https://github.com/jdelarosaquiros/llama2-decoder/assets/86489701/ea848fe3-1fff-4013-b674-b38b261b2870) | ![image](https://github.com/jdelarosaquiros/llama2-decoder/assets/86489701/1ff45527-5961-401d-a6f5-a327b913f9e3) |

## Discussion

As the charts above show, the probability of the tokens start fairly even, and they gradually become more skewed towards certain tokens. In other words, the model becomes more certain as the layers increase. With this in mind, I doubt that checking the consistency between layers to analyze the factuality would be helpful. I have two reasons to believe this. One is that the distribution of the first layers are far too even and random to matter in a consistency check. As for the subsequent layers, the same problem applies but to a lesser degree. The only exceptions might be the last few when the model becomes more certain, and the top weights might be more meaningful. However, between a popularity contest among less informed layers and a decision made by a single but most informed layer, I would choose the decision of the most informed layer. The second reason is that I believe one cannot know the factuality of a model simply gauging its confidence. It would be the equivalent of us deciding whether a fact is true or not based on our personal opinion of the fact. Bias is too big of a factor for factuality checks to be reliable. The most we could probably do by checking the consistency or confidence is to analyze whether the model is hallucinating. If a model is confident on a token, it probably has seen it being used in a similar case, but even if that’s true, it can still hallucinate. Take cooking for example, assume we tell the model to write a recipe with a list of ingredients that we have available. Then, the model proceeds to write the recipe, but it includes an ingredient not in our list because it is confident that the ingredient is typically used in the recipe. Despite being confident, it still hallucinated because the ingredient wasn’t part of our list.

## Results

| Layers | BLEU | Rouge-L | BERTScore |
| ------------- | ------------- | ------------- |
| Layer 8 | 0.000 | 0.004 | 0.728 |
| Layer 16 | 0.000 | 0.002 | 0.690 |
| Layer 24 | 0.008 | 0.086 | 0.760 |
| Layer 32 | 0.020 | 0.173 | 0.632 |

Note: Results were not great because the model was quantize to 4 bits.

As expected, the first layers had lower scores because their token probabilities were more evenly distributed than the subsequent ones, so the tokens were chosen more randomly. As the layers increased, the entropy of the probability distribution decreased, and some values started to stand out among the rest. My results show that this change led to an increase in performance, and this makes sense because the model was basically narrowing down the tokens to most likely ones to be the solution. The increase in performance was most noticeble in the BLEU and Rouge-L scores because compare exact words, and the higher layers typically selected more right words than the lower layers. Though, the BERTScore was less predictable probably because the outputs were not of enough quality for the BERT to reliably evaluate the outputs.



