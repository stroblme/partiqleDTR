import torch as t


def construct_rel_recvs(ln_leaves, self_interaction=False, device=None):
    """
    ln_leaves: list of ints, number of leaves for each sample in the batch
    """
    pad_len = max(ln_leaves)
    rel_recvs = []
    for l in ln_leaves:
        rel_recv = t.eye(pad_len, device=device)  # (l, l)
        rel_recv[:, l:] = 0
        rel_recv = rel_recv.repeat_interleave(pad_len, dim=1).T  # (l*l, l)
        for j in range(l, pad_len):  # remove padding vertex edges TODO optimize
            rel_recv[j::pad_len] = 0

        if self_interaction == False:
            rel_recv[0::pad_len + 1] = 0

        rel_recvs.append(rel_recv)

    return t.stack(rel_recvs)


def construct_rel_sends(ln_leaves, self_interaction=False, device=None):
    """
    ln_leaves: list of ints, number of leaves for each sample in the batch
    """
    pad_len = max(ln_leaves)
    rel_sends = []
    for l in ln_leaves:
        rel_send = t.eye(pad_len, device=device).repeat(pad_len, 1)
        if self_interaction == False:
            rel_send[t.arange(0, pad_len*pad_len, pad_len+1)] = 0
            # rel_send = rel_send[rel_send.sum(dim=1) > 0]  # (l*l, l)

        # padding
        rel_send[:, l:] = 0
        rel_send[l*(pad_len):] = 0
        rel_sends.append(rel_send)
    return t.stack(rel_sends)

def pad_collate_fn(batch):
    ''' Collate function for batches with varying sized inputs

    This pads the batch with zeros to the size of the large sample in the batch

    Args:
        batch(tuple):  batch contains a list of tuples of structure (sequence, target)
    Return:
        (tuple): Input, labels, mask, all padded
    '''
    # First pad the input data
    data = [item[0] for item in batch]
    # Here we pad with 0 as it's the input, so need to indicate that the network ignores it
    data = t.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)  # (N, L, F)
    data = data.transpose(0, 1)  # (L, N, F)
    # Then the labels
    labels = [item[1] for item in batch]

    # Note the -1 padding, this is where we tell the loss to ignore the outputs in those cells
    target = t.zeros(data.shape[1], data.shape[0], data.shape[0], dtype=t.long) - 1  # (N, L, L)
    # mask = t.zeros(data.shape[0], data.shape[1], data.shape[1])  # (N, L, L)

    # I don't know a cleaner way to do this, just copying data into the fixed-sized tensor
    for i, tensor in enumerate(labels):
        length = tensor.size(0)
        target[i, :length, :length] = tensor
        # mask[i, :length, :length] = 1

    return data, target  # mask


def rel_pad_collate_fn(batch, self_interaction=False):
    ''' Collate function for batches with varying sized inputs

    This pads the batch with zeros to the size of the large sample in the batch

    Args:
        batch(tuple):  batch contains a list of tuples of structure (sequence, target)
    Return:
        (tuple): Input, labels, rel_rec, rel_send, all padded
    '''
    lens = [sample[0].size(0) for sample in batch]

    data, target = pad_collate_fn(batch)

    rel_recvs = construct_rel_recvs(lens, self_interaction=self_interaction)
    rel_sends = construct_rel_sends(lens, self_interaction=self_interaction)

    return (data, rel_recvs, rel_sends), target