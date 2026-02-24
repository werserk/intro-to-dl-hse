import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def _run_recurrent(self, embeds: torch.Tensor, hidden=None):
        if embeds.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                if hidden is None:
                    return self.rnn(embeds)
                return self.rnn(embeds, hidden)
        if hidden is None:
            return self.rnn(embeds)
        return self.rnn(embeds, hidden)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, input length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, output length, vocab_size)
        """
        max_len = int(lengths.max().item())
        safe_indices = indices[:, :max_len]
        if (safe_indices < 0).any():
            safe_indices = safe_indices.masked_fill(safe_indices < 0, self.dataset.unk_id)
        embeds = self.embedding(safe_indices)
        rnn_out, _ = self._run_recurrent(embeds)
        logits = self.linear(rnn_out)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        device = next(self.parameters()).device
        self.eval()

        prefix_ids = self.dataset.text2ids(prefix)
        input_ids = [self.dataset.bos_id] + prefix_ids
        input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)

        embeds = self.embedding(input_tensor)
        output, hidden = self._run_recurrent(embeds)
        
        generated_ids = prefix_ids
        
        for _ in range(self.max_length - len(input_ids)):
            logits = self.linear(output[:, -1, :])
            
            probs = torch.softmax(logits / temp, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_id == self.dataset.eos_id:
                break
                
            generated_ids.append(next_id)
            
            current_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            embeds = self.embedding(current_tensor)
            output, hidden = self._run_recurrent(embeds, hidden)

        return self.dataset.ids2text(generated_ids)
