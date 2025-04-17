import torch
from transformers import AutoModel
import argparse
import os
import sys


def naive_exporter(args):
    print("start model conversion")
    model = AutoModel.from_pretrained(args.model_name)
    model.eval() # export model in inference mode
    bs = 1
    seq_len = 512
    dummy_inputs = (torch.randint(1000, size=(bs, seq_len), dtype=torch.int32), torch.zeros(bs, seq_len, dtype=torch.int32))
    torch.onnx.export(
        model,
        dummy_inputs,
        args.save,
        export_params=True,
        opset_version=17,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 'attention_mask':  {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size'}},
    )
    print(f"{args.save} model conversion completed")


class SentenceEmbeddingModel(torch.nn.Module):
    def __init__(self, model_name):
        super(SentenceEmbeddingModel, self).__init__()
        self.transformer_model = AutoModel.from_pretrained(model_name)
        self.transformer_model.eval() # export model in inference mode

    def forward(self, input_ids, attention_mask):
        model_outputs = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embedding = model_outputs.pooler_output  # This is the pooled output
        # normalize embedding
        sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
        return sentence_embedding


'''
help(outputs)
 |      last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
 |          Sequence of hidden-states at the output of the last layer of the model.
 |      pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
 |          Last layer hidden-state of the first token of the sequence (classification token) after further processing
 |          through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
 |          the classification token after processing through a linear layer and a tanh activation function. The linear
 |          layer weights are trained from the next sentence prediction (classification) objective during pretraining.
 |      hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
 |          Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
 |          one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

- [What is a 'hidden state' in BERT output?](https://datascience.stackexchange.com/questions/66786/what-is-a-hidden-state-in-bert-output)
BERT is a transformer.
A transformer is made of several similar layers, stacked on top of each others.
Each layer have an input and an output. So the output of the layer n-1 is the input of the layer n.
The hidden state you mention is simply the output of each layer.
'''


def customized_exporter(args):
    model = SentenceEmbeddingModel(args.model_name)
    print("start model conversion")
    bs = 1
    seq_len = 512
    dummy_inputs = (torch.randint(1000, size=(bs, seq_len), dtype=torch.int32), torch.zeros(bs, seq_len, dtype=torch.int32))
    # Specify input and output names
    input_names = ["input_ids", "attention_mask"]
    output_names = ["sentence_embedding"]
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'sentence_embedding': {0: 'batch_size'}
    }
    torch.onnx.export(
        model, 
        dummy_inputs,
        args.save,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17  # Specify the ONNX opset version
    )
    print(f"{args.save} model conversion completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="/root/code/bge-large-zh-v1.5")
    parser.add_argument("--save", default="model.onnx")
    args = parser.parse_args()

    if os.path.exists(args.save):
        print(f"{args.save} already exists")
        sys.exit(0)

    # naive_exporter(args)
    customized_exporter(args)
