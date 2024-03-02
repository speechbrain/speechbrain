import torch

class AttentionMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x):
        x = self.layers(x)
        att_w = torch.nn.functional.softmax(x, dim=2)
        return att_w


class EmbeddingLayer(torch.nn.Module):
    def __init__(self, discrete_ssl_model, SSL_layers, num_clusters, emb_dim,pad_index=0,init=False, freeze=False):
       super(EmbeddingLayer, self).__init__()
       self.discrete_ssl_model = discrete_ssl_model
       self.num_clusters=num_clusters
       self.freeze= freeze

       self.layers = torch.nn.ModuleList()
       self.layer_num=[]
       for layer_num,  vocabulary in zip(self.discrete_ssl_model.ssl_layer_ids, self.discrete_ssl_model.vocabularies):
            if layer_num not in SSL_layers:
                    continue
            layer = torch.nn.Embedding(num_clusters+1, emb_dim, padding_idx=pad_index).requires_grad_(not self.freeze)
            self.layer_num.append(layer_num)
            if init:
                with torch.no_grad():
                   layer.weight[pad_index] = torch.zeros(emb_dim)
                   layer.weight[1:] = torch.from_numpy(vocabulary)
            
            self.layers.append(layer)

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze):
            embeddings_list =[]
            for i,layer in enumerate(self.layers):
                embeddings_list.append(layer(x[:,:,i] - self.layer_num[i]*self.num_clusters))
            embeddings = torch.stack(embeddings_list, dim =-2)
            return embeddings