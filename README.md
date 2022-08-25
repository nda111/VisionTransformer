# Vision Transformer
Implementation of 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale' ICLR 2020 ([arXiv](https://arxiv.org/abs/2010.11929), [PDF](https://arxiv.org/pdf/2010.11929)) 
and a documentation about its historical and technical background.
Some figures and equations are from their original paper.
![vision transformer](./archive/img/01.%20vision%20transformer.png)

## Index
> 1. Preceding Works
>    * Attention Mechanism
>    * Transformers 
> 2. Vision Transformers
>    * Overall Structure
>    * Patching
>    * Positional Embedding
>    * Layer Normalization
>    * GELU Activation
> 3. Experiments
>    * Environments
>    * Result

## Preceding Works
### Attention Mechanism
#### Attention in NLP
In the field of NLP, they pushed the limitations of RNNs by developing [attention mechanism](https://arxiv.org/abs/1409.0473). Attention mechanism is a method to literally pay attention to the important features. In NLP, for example, with the sentence "I love NLP because it is fascinating.", the word 'it' referes to 'NLP'. Thus, when you process 'it', you have to treat it as 'NLP'. 
> <img src='./archive/img/01. preceding works/01. attention mechanism/01. attention score_nlp.png' />
> <p>Figure1. An example of attention score between words in a sentence. You can see 'it' and 'NLP' has high attention score. This examples represents one of attention mechanisms called self-attention.</p>

Attention is basically inner product operation as similarity measurement. You have three following informations:
|Acronym|Name|Description|En2Ko example|
|:-:|-|-|-|
|Q|Query|An information to compare this with all keys to find the best-matching key.|because|
|K|Key|A set of key that leads to appropriate values.|I, love, NLP, because, it, is, fascinating|
|V|Value|The result of attention.|나, 자연어처리, 좋, 왜냐하면, 이것, 은, 흥미롭다|

When you translate "because" in "I love NLP because it is fascinating.", first you calculate the similarity between "because" and other words. Then, weight-sum the korean words with the similarity, you get an appropriate word vector.

To take attention mechanism further, Ashish Vaswani et al. introduced a framework named called "[Transformer](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)". Transformer takes a sequential sentence as a pile of words and extracts features through attention encoder and decoders with several types of attention. More details of transformer structure is explained on the "Transformer" section. While attention mechanism sounds quite reasonable with NLP examples, it's not trivial that it works in computer vision as well. Alexey Dosovitskiy from Google Research introduced transformer for computer vision and explained how it works.

#### Attention in Computer Vision
By attention mechanism, in NLP, the model calculates where to concentrate. On the other hand, in computer vision, Jie Hu et al. proposed a new attention structure for computer vision (Squeeze-and Excitation Networks, CVPR 2018, [arXiv](https://arxiv.org/abs/1709.01507), [PDF](https://arxiv.org/pdf/1709.01507.pdf)). The proposed architecture contains a sub-network that learns channel-wize attention score.

> <img src='./archive/img/01. preceding works/01. attention mechanism/02. se module.png' />
> <p>Figure2. This diagram shows the structure of squeeze-and-excitation module. A sub-network branch makes 1x1xC dimensional tensor of attention scores for the input of the module.</p>

Convolutional neural networks, is in a way, a channel-level ensemble. Each kernels extracts different features from an image for another features. But these features from different kernels are not always important equally. SE module helps emphasize the important features and reduce other features.

Another way to adapt attention is to split an image into fixed size patches and calculates similarity between all combinations of the patches. Detailed description of this method is in the "Transformer' section.

## Vision Transformer
### Overall Structure
1. Image patching
2. Linear projection
3. CLS token & Positional embedding
4. Transformer encoder
5. MLP head

### Patching
<img src = "archive/img/02. vision transformer/01. patching/vit_patching_2.png" width="300px"/> <br />
Figure3. Patching is a process corresponding to the red area above among the entire ViT structure.

> <img src="./archive/img/02.%20vision%20transformer/01.%20patching/vit_patching.png"/>
> <img src="./archive/img/02.%20vision%20transformer/01.%20patching/vit_patching_1.png"/>
> <p>Figure4. Divide the input image into fixed size patches. And flatten each patches to stack as columns. P, N stands for patch size and total number of patches respectively.</p>

C, H, and W represent the number of channels, height and width, respectively. And P represents the size of the patch. Each patch is size of (C, P, P) and N means the total number of patches. The total number of patches N can be obtained through $\frac{HW}{P^2}$.
If each patch is flattened into a 2D vector, the size of each vector becomes $1 \times P^2C$, and the sum of these vectors is called $x_p=N \times P^2C$.

### Linear Projection
<img src = "archive/img/02. vision transformer/02. linear projection/vit_linear_projection_2.png" width="300"> <br/>
Linear projection is a process corresponding to the red area above among the entire ViT structure.<br/><br/>

> <img src="./archive/img/02.%20vision%20transformer/02.%20linear%20projection/vit_linear_projection.png" />
> <p>Figure5. Flatten the patches and get it through a linear layer to form a tensor of specific size.</p>
- Each flattened 2d patch is multiplied by E, which is a Linear Projection Matrix, and the vector size is changed to a latent vector size (D).
- The shape of E becomes ($P^2C, D$), and when $x_p$ is multiplied by E, it has the size (N, D). If the batch size is also considered, a tensor having a size of (B, N, D) can be finally obtained.

### Class Token
Like BERT([arXiv](https://arxiv.org/abs/1810.04805), [PDF](https://arxiv.org/pdf/1810.04805.pdf)), transformer trains class token by passing it through multiple encoder blocks. The class token first initialized with zeros and appended to the input. Just like BERT, the NLP transformer also uses class token. Consequently, class token was inherited to vision transformer too. On vision transformer. The class token is a special symbol to train. Even if it looks like a part of the input, as long as it's a trainable parameter, it make more sense to treat it as a part of the model. 

### Positional Embedding
<img src = "archive/img/02. vision transformer/03. positional embedding/vit_pe_2.png" width="300"> <br/>
Positional embedding is a process corresponding to the red area above among the entire ViT structure.
First, you should understand that sequence is a kind of position. The authors of NLP transformer tried to embed fixed positional values to the input and the value was formulated as ${p_t}^{(i)} := \begin{cases} \sin(w_k \bullet t) \quad \mathrm{if} i=2k \\ \cos(w_k \bullet t) \quad \mathrm{if} i=2k+1 \\ \end{cases}, w_t = {1 \over {10000^{2k/d}}}$. This represents unique positional information to all tokens. On the other hand, vision transformer, set the positional information as another learnable parameter. 

> <img src="./archive/img/02.%20vision%20transformer/03.%20positional%20embedding/vit_pe.png" />
> <p>Figure6. Add the class token to the embedding result as shown in the figure above. Then a matrix of size (N, D) becomes of size (N+1, D).</p>
- Add CLS Token, a learnable random vector of 1xD size.
- Add $E_{pos}$, a learnable random vector of size (N+1)xD.
- The final Transformer input becomes $z_0$.  

After the training, the positional vector is looks like Figure7.
> <img src='./archive/img/01. preceding works/01. attention mechanism/05. positional embedding.png' />
> Figure7. Position embeddings of models trained with different hyperparameters.

### GELU Activation
They applied GELU activation function([arXiv](https://arxiv.org/abs/1606.08415), [PDF](https://arxiv.org/pdf/1606.08415.pdf)) proposed by Dan Hendrycks and Kevin Gimpel. They combined dropout, zoneout and ReLU activation function to formulate GELU. ReLU gives non-linearity by dropping negative outputs and os as GELU. Let $x\Phi(x) = \Phi(x) \times Ix + (1 - \Phi(x)) \times 0x$, then $x\Phi(x)$ defines decision boundary. Refer to the paper, loosely, this expression states that we scale $x$ by how much greater it is than other inputs. Since, the CDF of a Gaussian is often computed with the error function, they defiend Gaussian Error Linear Unit (GELU) as $\textrm{GELU}(x) = xP(X \le x) = x\Phi(x)=x\bullet {1 \over 2}[\textrm{erf}({x \over \sqrt{2}})]$. and we can approximate this with $\mathrm{GELU}(x) = 0.5x(1+\tanh[\sqrt{2 \over \pi}(x + 0.044715x^3)])$.

> <img src='./archive/img/01. preceding works/01. attention mechanism/03. gelu.png' />
> <p>Figure8. Graph comparison among GELU, ReLU and ELU activation functions.</p>

> <img src='./archive/img/01. preceding works/01. attention mechanism/04. gelu_performance.png' /> <br />
> <p>Figure9. MNIST Classification Results. Left are the loss curves without dropout, and right are curves with a dropout rate of 0.5. Each curve is the the median of five runs. Training set log losses are the darker, lower curves, and the fainter, upper curves are the validation set log loss curves.</p>

See the [paper](https://arxiv.org/abs/1606.08415) for more experiments.

### Layer Normalization
<img src="./archive/img/02.%20vision%20transformer/04.%20layer%20normalization/vit_ln.png" width="450px" /> <br />
Layer normalization is a stage to normalize the features of pixels and channels for one sample.
This normalization is sample independent and normalizes the total image tensor.

> <img src = "archive/img/02. vision transformer/04. layer normalization/vit_ln_2.png" width="500px"/>     
> <p>Figure10. Comparison of normalization target between batch normalization and layer normalization.</p>
- Batch Normalization operates only on N, H, and W. Therefore, the mean and standard deviation are calculated regardless of channel map C and normalized for batch N.
- Since Layer Normalization operates only on C, H, W, the mean and standard deviation are calculated regardless of batch N. That is, it is normalized to channel map C.
- Layer normalization is more used than batch normalization because the mini-batch length can be different in NLP's Transformer. ViT borrowed NLP's Transformer Layner normalization.
