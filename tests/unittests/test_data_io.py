def test_batch_pad_right():
    import torch
    from speechbrain.data_io.utils import batch_pad_right
    import numpy as np

    n_tensors = [2, 3, 4]
    n_dims = [2, 4, 5, 6]

    for i in range(len(n_tensors)):
        tensors = []
        shapes = []
        for t in range(n_tensors[i]):
            c_shape = [np.random.randint(1, 5) for x in range(n_dims[i])]
            tensors.append(torch.rand((c_shape)))
            shapes.append(c_shape)

        padded, valid_indx = batch_pad_right(tensors)

        # assert each c_shape has same len
        assert len(set([len(x) for x in shapes])) == 1

        # get the max shape for each element
        max_shapes = shapes[0]
        for dim in range(len(shapes[0])):
            for shape in range(1, len(shapes)):
                max_shapes[dim] = max(max_shapes[dim], shapes[shape][dim])

        assert list(padded.shape[1:]) == max_shapes

        for batch_indx in range(len(tensors)):
            padded_indxs = [
                x - y for x, y in zip(max_shapes, tensors[batch_indx].shape)
            ]
            for dim in range(len(padded_indxs)):
                assert (
                    valid_indx[batch_indx][dim]
                    == tensors[batch_indx].shape[dim]
                    / padded[batch_indx].shape[dim]
                )
