import torch


def test_embedding():

    from speechbrain.nnet.embedding import Embedding

    # create one hot vector and consider blank as zero vector
    embedding_dim = 39
    blank_id = 39
    size_dict = 40
    emb = Embedding(
        num_embeddings=size_dict, consider_as_one_hot=True, blank_id=blank_id,
    )
    inputs = torch.Tensor([10, 5, 2, 0, 39]).long()
    output = emb(inputs, init_params=True)
    assert output.shape == (5, 39)

    # use standard embedding layer
    embedding_dim = 128
    emb = Embedding(num_embeddings=size_dict, embedding_dim=embedding_dim)
    inputs = torch.randint(0, 40, (5, 10))
    output = emb(inputs, init_params=True)
    assert output.shape == (5, 10, 128)
