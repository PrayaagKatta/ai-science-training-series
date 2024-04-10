# %% [markdown]
# # Large language models (LLMs): Part II
# 
# Author: Archit Vasan , including materials on LLMs by Varuni Sastri, and discussion/editorial work by Taylor Childers, Carlo Graziani, Bethany Lusch, and Venkat Vishwanath (Argonne)
# 
# Inspiration from the blog posts "The Illustrated Transformer" and "The Illustrated GPT2" by Jay Alammar, highly recommended reading.
# 
# Before you begin, make sure that you have your environment set up and your repo refreshed, as described in previous lessons, and reviewed in the accompanying 'Readme.md' file. Make sure that you select the kernel 'datascience/conda-2023-01-10' at the top-left of the Jupyter notebook.

# %% [markdown]
# ## Overview
# 1. Training and inference using Hugging Face
# 2. Elements of an LLM
# 3. Attention mechanisms
# 4. Positional encoding
# 5. Output layers
# 6. Training loops

# %%
import os
os.environ["HTTP_PROXY"]="http://proxy-01.pub.alcf.anl.gov:3128"
os.environ["HTTPS_PROXY"]="http://proxy-01.pub.alcf.anl.gov:3128"
os.environ["http_proxy"]="http://proxy-01.pub.alcf.anl.gov:3128"
os.environ["https_proxy"]="http://proxy-01.pub.alcf.anl.gov:3128"
os.environ["ftp_proxy"]="http://proxy-01.pub.alcf.anl.gov:3128" 

# %% [markdown]
# ## LLM training and inference using HuggingFace

# %% [markdown]
# <img src="images/hf-logo-with-title.png" alt="Drawing" style="width: 300px;"/>
# HuggingFace is a platform and community that provides open-source library tools and resources like pre-trained models and datasets.
# Refer to the following links for more information :
# 
# https://huggingface.co/docs/hub/index
# 
# https://huggingface.co/docs/transformers/en/index

# %% [markdown]
# Warning: _Large Language Models are only as good as their training data. They have no ethics, no judgement, or editing ability. We will be using some pretrained models from Hugging Face which used wide samples of internet hosted text. The datasets have not been strictly filtered to restrict all malign content so the generated text may be surprisingly dark or questionable. They do not reflect our core values and are only used for demonstration purposes._

# %% [markdown]
# ### Inference
# 
# We can use the Huggingface pipeline with a pretrained GPT2 model to generate text given a prompt.

# %%
from transformers import AutoTokenizer,AutoModelForCausalLM, AutoConfig
# input_text = "My dog really wanted to"
# from transformers import pipeline
# generator = pipeline("text-generation", model="openai-community/gpt2")
# generator(input_text, max_length=20, num_return_sequences=5)

# %% [markdown]
# We will cover  evaluation metrics,as well as safe and responsibilities practices when using LLMs in **Session 8**.

# %% [markdown]
# ### Training

# %% [markdown]
# We can also load in our own dataset and train a model with this data as follows:

# %%
# !pip install accelerate -U

# %%
from transformers import TextDataset,DataCollatorForLanguageModeling

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128) 
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

# %%
from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

train_dataset,test_dataset,data_collator = load_dataset('dataset/train_input.txt','dataset/test_input.txt', tokenizer)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./gpt2", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps = 40, # Number of update steps between two evaluations.
    save_steps=80, # after # steps model is saved 
    warmup_steps=50,# number of warmup steps for learning rate scheduler
    prediction_loss_only=True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# %% [markdown]
# ## What is going on below the hood?

# %% [markdown]
# There are two components that are "black-boxes" here:
# 1. The method for tokenization
# 2. The model that generates novel text.
# 
# Carlo Graziani already gave a great explanation of tokenization last week and how this affects embeddings (https://github.com/argonne-lcf/ai-science-training-series/blob/main/04_intro_to_llms/Sequential_Data_Models.ipynb)

# %% [markdown]
# Today we will take a closer look at how the model is designed to deal with language.

# %% [markdown]
# Let's look inside GPT2! GPT2 incorporates the `GPT2LMHeadModel` architecture so let's inspect this more closely.

# %%
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# print(model)

# %% [markdown]
# ## General elements of an LLM

# %% [markdown]
# GPT-2 is an example of the popular Transformer architecture. 
# 

# %% [markdown]
# <img src="images/decoder_only_block.png" alt="Drawing" style="width: 200px;"/>
# Image credit: https://arxiv.org/pdf/1706.03762.pdf

# %% [markdown]
# The gray section in this figure is the Transfomer Decoder and it is the main mechanism GPT2 uses to encode context of language into its predictions.
# 
# <img src="images/transformer-decoder-intro.png" alt="Drawing" style="width: 600px;"/>
# Image credit: https://jalammar.github.io/illustrated-gpt2/

# %% [markdown]
# The Transformer-Decoder is composed of Decoder blocks stacked ontop of each other where each contains two types of layers: 
# 1. Masked Self-Attention and 
# 2. Feed Forward Neural Networks.

# %% [markdown]
# You have already discussed Feed Forward Neural Networks in detail in the other lectures in this series. To review this, please look at https://github.com/argonne-lcf/ai-science-training-series/blob/main/02_intro_neural_networks/01_introduction_mnist.ipynb

# %% [markdown]
# In this lecture, we will 
# * First, discuss attention mechanisms at length as this is arguably the greatest contribution by Transformers.
# * Second, extend the discussion from last week (https://github.com/argonne-lcf/ai-science-training-series/blob/main/04_intro_to_llms/Sequential_Data_Models.ipynb) on embedding input data while taking into account position.
# * Third, discuss outputting real text/sequences from the models.
# * Fourth, build a training loop for a mini-LLM.

# %% [markdown]
# **Let's set up all the imports we will need**

# %%
## IMPORTS

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4 ## so head_size = 16
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# %% [markdown]
# ## Attention mechanisms
# 
# Suppose the following sentence is an input sentence we want to translate using an LLM:
# 
# `”The animal didn't cross the street because it was too tired”`
# 
# Last week, Carlo mentioned that the Transformer learns an embedding of all words allowing interpretation of meanings of words.
# 
# <img src="images/viz-bert-voc-verbs.png" alt="Drawing" style="width: 400px;"/>
# 
# So, if the model did a good job in token embedding, it will "know" what all the words in this sentence mean. 

# %% [markdown]
# But to understand a full sentence, the model also need to understand what each word means in relation to other words.
# 
# For example, when we read the sentence:
# `”The animal didn't cross the street because it was too tired”`
# we know intuitively that the word `"it"` refers to `"animal"`, the state for `"it"` is `"tired"`, and the associated action is `"didn't cross"`.
# 
# However, the model needs a way to learn all of this information in a simple yet generalizable way.
# What makes Transformers particularly powerful compared to earlier sequential architectures is how it encodes context with the **self-attention mechanism**.
# 
# As the model processes each word in the input sequence, attention looks at other positions in the input sequence for clues to a better understanding for this word.

# %% [markdown]
# <img src="images/transformer_self-attention_visualization.png" alt="Drawing" style="width: 300px;"/>

# %% [markdown]
# Image credit: https://jalammar.github.io/illustrated-transformer/

# %% [markdown]
# Self-attention mechanisms use 3 vectors to encode the context of a word in a sequence with another word:
# 1. Query: the word representation we score other words against using the other word's keys
# 2. Key: labels for the words in a sequence that we match against the query
# 3. Value: actual word representation. We will use the queries and keys to score the word's relevance to the query, and multiply this by the value. 
# 
# An analogy provided by Jay Alammar is thinking about attention as choosing a file from a file cabinet according to information on a post-it note. You can use the post-it note (query) to identify the folder (key) that most matches the topic you are looking up. Then you access the contents of the file (value) according to its relevance to your query.

# %% [markdown]
# <img src="images/self-attention-example-folders-3.png" alt="Drawing" style="width: 500px;"/>
# Image credit: https://jalammar.github.io/illustrated-gpt2/

# %% [markdown]
# In our models, we can encode queries, keys, and values using simple linear layers with the same size (`sequence length, head_size`). During the training process, these layers will be updated to best encode context.

# %%
C = 32 # channels
head_size = 16

key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

# %% [markdown]
# The algorithm for self-attention is as follows:
# 
# 1. Generate query, key and value vectors for each word
# 2. Calculate a score for each word in the input sentence against each other.
# 3. Divide the scores by the square root of the dimension of the key vectors to stabilize the gradients. This is then passed through a softmax operation.
# 4. Multiply each value vector by the softmax score.
# 5. Sum up the weighted value vectors to produce the output.
# 

# %% [markdown]
# <img src="images/self-attention-output.png" alt="Drawing" style="width: 450px;"/>

# %% [markdown]
# Image credit: https://jalammar.github.io/illustrated-transformer/

# %% [markdown]
# Let's see how attention is performed in the code.

# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# Here we want the wei to be data dependent - ie gather info from the past but in a data dependant way

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16) # each token here (totally B*T) produce a key and query in parallel and independently
q = query(x) # (B, T, 16)
v = value(x)

wei =  q @ k.transpose(-2, -1) * head_size**-0.5 # (B, T, 16) @ (B, 16, T) ---> (B, T, T). #
wei = F.softmax(wei, dim=-1) # exponentiate and normalize giving a nice distibution that sums to 1 and
                             # now it tells us that in a data dependent manner how much of info to aggregate from

out = wei @ v # aggregate the attention scores and value vector.


# %%
# print(out[0])

# %% [markdown]
# ### Multi-head attention
# 
# In practice, multiple attention heads are used which
# 1. Expands the model’s ability to focus on different positions and prevent the attention to be dominated by the word itself.
# 2. Have multiple “representation subspaces”. Have multiple sets of Query/Key/Value weight matrices

# %% [markdown]
# <img src="images/transformer_multi-headed_self-attention-recap.png" alt="Drawing" style="width: 700px;"/>

# %% [markdown]
# Image credit: https://jalammar.github.io/illustrated-transformer/

# %% [markdown]
# ### Let's see attention mechanisms in action!

# %% [markdown]
# We are going to use the powerful visualization tool bertviz, which allows an interactive experience of the attention mechanisms. Normally these mechanisms are abstracted away but this will allow us to inspect our model in more detail.

# %%
# !pip install bertviz

# %% [markdown]
# Let's load in the model, GPT2 and look at the attention mechanisms. 
# 
# **Hint... click on the different blocks in the visualization to see the attention**

# %%
from transformers import AutoTokenizer, AutoModel, utils, AutoModelForCausalLM

from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

model_name = 'openai-community/gpt2'
input_text = "No, I am your father"  
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
outputs = model(inputs)  # Run model
attention = outputs[-1]  # Retrieve attention from model outputs
tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
model_view(attention, tokens)  # Display model view

# %% [markdown]
# ## Positional encoding

# %% [markdown]
# Last week, Carlo discussed token embedding, which is when words are encoded into a vocabulary. Now, we just discussed attention mechanisms which account for context between words. Another question we should ask is how do we account for the order of words in an input sentence
# 
# Consider the following two sentences to see why this is important:
# 
# ``The man ate the sandwich.``
# 
# ``The sandwich ate the man.``
# 
# Clearly, these are two vastly different situations even though they have the same words. The Transformer can 
# 
# Transformers differentiate between these situations by adding a **Positional encoding** vector to each input embedding. These vectors follow a specific pattern that the model learns, which helps it determine the position of each word.

# %% [markdown]
# <img src="images/positional_encoding.png" alt="Drawing" style="width: 500px;"/>
# Image credit: https://medium.com/@xuer.chen.human/llm-study-notes-positional-encoding-0639a1002ec0

# %% [markdown]
# We set up positional encoding similarly as token embedding using the ``nn.Embedding`` tool. We use a simple embedding here but there are more complex positional encodings used such as sinusoidal. 
# 
# For an explanation of different positional encodings, refer to this post: https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

# %%
vocab_size = 65
n_embd = 64

token_embedding_table = nn.Embedding(vocab_size, n_embd)
block_size = 32 # what is the maximum context length for predictions?
position_embedding_table = nn.Embedding(block_size, n_embd)

# %% [markdown]
# You will notice the positional encoding size is `(block_size, n_embed)` because it encodes for the postion of a token within the sequence of size `block_size`

# %% [markdown]
# Then, the position embedding used is simply added to the token embedding to apply positional embedding.

# %% [markdown]
# Let's look at token embedding alone:

# %%
x = torch.tensor([1,3,15,4,7,1,4,9])
x = token_embedding_table(x)
# print(x[0])

# %% [markdown]
# And token + positional embeddings:

# %%
x = torch.tensor([1,3,15,4,7,1,4,9])
x= position_embedding_table(x) + token_embedding_table(x)
# print(x[0])

# %% [markdown]
# You can see a clear offset between these two embeddings.

# %% [markdown]
# During the training process, these embeddings will be learned to best encode the token and positional embeddings of the sequences.

# %% [markdown]
# ## Output layers
# 
# At the end of our Transformer model, we are left with a vector, so how do we turn this into a word?
# 
# <img src="images/transformer-decoder-intro.png" alt="Drawing" style="width: 400px;"/>
# 
# Using a final Linear layer and a Softmax Layer.
# The Linear layer projects the vector produced by the stack of decoders, into a larger vector called a logits vector.
# 
# If our model knows 10,000 unique English words learned from its training dataset the logits vector is 10,000 cells wide – each cell corresponds to the score of a unique word.
# 
# The softmax layer turns those scores into probabilities. The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

# %% [markdown]
# <img src="images/transformer_decoder_output_softmax.png" alt="Drawing" style="width: 450px;"/>

# %% [markdown]
# Image credit: https://jalammar.github.io/illustrated-transformer/

# %% [markdown]
# ## Training
# 
# How does an LLM improve over time?
# We want to compare the probabilitiy distribution for each token generated by our model to the ground truths. 
# Our model produces a probability distribution for each token. We want to compare these probability distributions to the ground truths. 
# For example, when translating the sentence: “je suis étudiant” into “i am a student” as can be seen in the example:

# %% [markdown]
# <img src="images/output_target_probability_distributions.png" alt="Drawing" style="width: 500px;"/>

# %% [markdown]
# Image credit: https://jalammar.github.io/illustrated-transformer/

# %% [markdown]
# The model can calculate the loss between the vector it generates and the ground truth vector seen in this example. A commonly used loss function is cross entropy loss:
# 
# $CE = -\sum_{x \in X} p(x) log q(x)$
# 
# where p(x) represents the true distribution and q(x) represents the predicted distribution.

# %%
from torch.nn import functional as F
logits = torch.tensor([0.5, 0.1, 0.3])
targets = torch.tensor([1.0, 0.0, 0.0])
loss = F.cross_entropy(logits, targets)
# print(loss)

# %% [markdown]
# Another important metric commonly used in LLMs is **perplexity**.
# 
# Intuitively, perplexity means to be surprised. We measure how much the model is surprised by seeing new data. The lower the perplexity, the better the training is.

# %% [markdown]
# Mathematically, perplexity is just the exponent of the negative cross entropy loss:
# 
# $\text{perplexity} = exp(\text{CE})$

# %%
perplexity = torch.exp(loss)
# print(perplexity)

# %% [markdown]
# In this example, we are using cross entropy loss.

# %% [markdown]
# ## Let's train a mini-LLM from scratch

# %% [markdown]
# ### Set up hyperparameters:

# %%
# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4 ## so head_size = 16
n_layer = 4
dropout = 0.0
# ------------

# %% [markdown]
# ### Load in data and create train and test datasets

# %% [markdown]
# We're going to be using the tiny Shakespeare dataset. 
# Data is tokenized according to a simple character based tokenizer.
# Data is split into a train and test set so we have something to test after performing training (9:1 split).

# %%
with open('dataset/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# %% [markdown]
# ### Set up the components of the Decoder block: 
# * MultiHeadAttention
# * FeedForward Network

# %%
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C) 16,32,16
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# %% [markdown]
# ### Combine components into the Decoder block

# %%
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # Communication
        x = x + self.ffwd(self.ln2(x))  # Computation
        return x

# %% [markdown]
# ### Set up the full Transformer model 
# This is a combination of the Token embeddings, Positional embeddings, a stack of Transformer blocks and an output block.

# %%
# super simple language model
class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



# %% [markdown]
# We will be training a larger LLM on distributed resources in session 6.

# %% [markdown]
# ## Homework
# 
# 1. In this notebook, we learned the various components of an LLM. 
#     Your homework this week is to take the mini LLM we created from scratch and run your own training loop. Show how the training and validation perplexity change over the steps.
#       
#     Hint: this function might be useful for you:

# %%
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# %% [markdown]
# 2. Run the same training loop but modify one of the hyperparameters from this list: 

# %%
# hyperparameters
n_embd = 64
n_head = 4 ## so head_size = 16
n_layer = 4

total_samples = len(train_data)

# Calculate the number of batches
num_batches = total_samples // batch_size
if total_samples % batch_size != 0:
    num_batches += 1

def train_one_epoch(model, num_batches):
    model.train()
    
    for batch in range(num_batches):
        X, Y = get_batch('train')

        optimizer.zero_grad()

        logits, loss = model(X, Y)
        
        loss.backward()

        optimizer.step()


llmodel = LanguageModel().to(device)
optimizer = torch.optim.AdamW(llmodel.parameters(), lr=learning_rate)

num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    train_one_epoch(llmodel, num_batches)

    losses = estimate_loss()
    train_loss = losses['train']
    val_loss = losses['val']
    
    print(f'Epoch {epoch + 1}, Train Perplexity: {torch.exp(train_loss)}, Val Perplexity: {torch.exp(val_loss)}')

# %%
# hyperparameters
n_embd = 16
n_head = 4 ## so head_size = 16
n_layer = 4

total_samples = len(train_data)

# Calculate the number of batches
num_batches = total_samples // batch_size
if total_samples % batch_size != 0:
    num_batches += 1

def train_one_epoch(model, num_batches):
    model.train()
    
    for batch in range(num_batches):
        X, Y = get_batch('train')

        optimizer.zero_grad()

        logits, loss = model(X, Y)
        
        loss.backward()

        optimizer.step()


llmodel = LanguageModel().to(device)
optimizer = torch.optim.AdamW(llmodel.parameters(), lr=learning_rate)

num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    train_one_epoch(llmodel, num_batches)

    losses = estimate_loss()
    train_loss = losses['train']
    val_loss = losses['val']
    
    print(f'Epoch {epoch + 1}, Train Perplexity: {torch.exp(train_loss)}, Val Perplexity: {torch.exp(val_loss)}')

# %%
# hyperparameters
n_embd = 64
n_head = 8 ## so head_size = 16
n_layer = 4

total_samples = len(train_data)

# Calculate the number of batches
num_batches = total_samples // batch_size
if total_samples % batch_size != 0:
    num_batches += 1

def train_one_epoch(model, num_batches):
    model.train()
    
    for batch in range(num_batches):
        X, Y = get_batch('train')

        optimizer.zero_grad()

        logits, loss = model(X, Y)
        
        loss.backward()

        optimizer.step()


llmodel = LanguageModel().to(device)
optimizer = torch.optim.AdamW(llmodel.parameters(), lr=learning_rate)

num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    train_one_epoch(llmodel, num_batches)

    losses = estimate_loss()
    train_loss = losses['train']
    val_loss = losses['val']
    
    print(f'Epoch {epoch + 1}, Train Perplexity: {torch.exp(train_loss)}, Val Perplexity: {torch.exp(val_loss)}')

# %%
# hyperparameters
n_embd = 64
n_head = 4 ## so head_size = 16
n_layer = 8

total_samples = len(train_data)

# Calculate the number of batches
num_batches = total_samples // batch_size
if total_samples % batch_size != 0:
    num_batches += 1

def train_one_epoch(model, num_batches):
    model.train()
    
    for batch in range(num_batches):
        X, Y = get_batch('train')

        optimizer.zero_grad()

        logits, loss = model(X, Y)
        
        loss.backward()

        optimizer.step()


llmodel = LanguageModel().to(device)
optimizer = torch.optim.AdamW(llmodel.parameters(), lr=learning_rate)

num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    train_one_epoch(llmodel, num_batches)

    losses = estimate_loss()
    train_loss = losses['train']
    val_loss = losses['val']
    
    print(f'Epoch {epoch + 1}, Train Perplexity: {torch.exp(train_loss)}, Val Perplexity: {torch.exp(val_loss)}')

# %%
# hyperparameters
n_embd = 64
n_head = 8 ## so head_size = 16
n_layer = 8

total_samples = len(train_data)

# Calculate the number of batches
num_batches = total_samples // batch_size
if total_samples % batch_size != 0:
    num_batches += 1

def train_one_epoch(model, num_batches):
    model.train()
    
    for batch in range(num_batches):
        X, Y = get_batch('train')

        optimizer.zero_grad()

        logits, loss = model(X, Y)
        
        loss.backward()

        optimizer.step()


llmodel = LanguageModel().to(device)
optimizer = torch.optim.AdamW(llmodel.parameters(), lr=learning_rate)

num_epochs = 5

# Training loop
for epoch in range(num_epochs):
    train_one_epoch(llmodel, num_batches)

    losses = estimate_loss()
    train_loss = losses['train']
    val_loss = losses['val']
    
    print(f'Epoch {epoch + 1}, Train Perplexity: {torch.exp(train_loss)}, Val Perplexity: {torch.exp(val_loss)}')

# %% [markdown]
# Run this at least 4 times with a different value and plot each perplexity over training step. Write a sentence on how the perplexity changed.

# %% [markdown]
# Bonus 1: output some generated text from each model you trained. Did the output make more sense with some hyperparameters than others? 

# %% [markdown]
# Bonus 2: We saw a cool visualization of attention mechanisms with BertViz. Take a more complicated model than GPT2 such as "meta-llama/Llama-2-7b-chat-hf" and see how the attention mechanisms are different 

# %% [markdown]
# ## References

# %% [markdown]
# Here are some recommendations for further reading and additional code for review.
# 
# * "The Illustrated Transformer" by Jay Alammar
# * "Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)"
# * "The Illustrated GPT-2 (Visualizing Transformer Language Models)"
# * "A gentle introduction to positional encoding"
# * "LLM Tutorial Workshop (Argonne National Laboratory)"
# * "LLM Tutorial Workshop Part 2 (Argonne National Laboratory)"

# %%



