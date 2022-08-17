import torch as t
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional, OrderedDict, Tuple
from kedro.io import AbstractDataSet
import dill


class HybridTorchModel:
    def __init__(
        self,
        setup_fn: Callable,
        *args,
        state_dict: Optional[OrderedDict[str, t.Tensor]] = None,
        **kwargs,
    ):
        self._setup_fn = setup_fn
        self._model = self._setup_fn(*args, **kwargs)
        if state_dict is not None:
            self._model.load_state_dict(state_dict)

    @property
    def model(self) -> t.nn.Sequential:
        return self._model


class HybridTorchModelDataset(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = PurePosixPath(filepath)

    def _load(self):
        weights = t.load(Path(self._filepath, "weights.pt"))
        with open(Path(self._filepath, "setup_fn.pkl"), "rb") as dill_file:
            setup_fn = dill.load(dill_file)
        model = HybridTorchModel(setup_fn=setup_fn, state_dict=weights)
        return model

    def _save(self, data: HybridTorchModel):
        if not self._exists():
            Path(self._filepath.as_posix()).mkdir()
        with open(Path(self._filepath.as_posix(), "setup_fn.pkl"), "wb") as output:
            output.write(dill.dumps(data._setup_fn))
        t.save(data.model.state_dict(), Path(self._filepath.as_posix(), "weights.pt"))

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        ...


def construct_rel_recvs(ln_leaves, self_interaction=False, device=None):
    """
    ln_leaves: list of ints, number of leaves for each sample in the batch
    """
    pad_len = max(ln_leaves)
    rel_recvs = []
    for l in ln_leaves:
        rel_recv = t.eye(pad_len, device=device)  # (l, l), identity matrix
        rel_recv[
            :, l:
        ] = 0  # set everything "behind" the actual number of classes to zero
        rel_recv = rel_recv.repeat_interleave(
            pad_len, dim=1
        ).T  # (l*l, l) # repeat each row of this matrix l times

        # we trim of the "intermediate" ones here
        for j in range(l, pad_len):  # remove padding vertex edges TODO optimize
            rel_recv[j::pad_len] = 0

        # and here we remove the "ones" on the diagonal
        if self_interaction == False:
            rel_recv[0 :: pad_len + 1] = 0

        rel_recvs.append(rel_recv)  # append to form a batch

    return t.stack(rel_recvs)  # convert list to tensor


def construct_rel_sends(ln_leaves, self_interaction=False, device=None):
    """
    ln_leaves: list of ints, number of leaves for each sample in the batch
    """
    pad_len = max(ln_leaves)
    rel_sends = []
    for l in ln_leaves:
        rel_send = t.eye(pad_len, device=device).repeat(pad_len, 1)
        if self_interaction == False:
            rel_send[t.arange(0, pad_len * pad_len, pad_len + 1)] = 0
            # rel_send = rel_send[rel_send.sum(dim=1) > 0]  # (l*l, l)

        # padding
        rel_send[
            :, l:
        ] = 0  # set everything "behind" the actual number of classes to zero
        rel_send[
            l * (pad_len) :
        ] = 0  # set everything "below" the actual number of classes repeated pads to zero
        rel_sends.append(rel_send)
    return t.stack(rel_sends)


def pad_collate_fn(batch):
    """Collate function for batches with varying sized inputs

    This pads the batch with zeros to the size of the large sample in the batch

    Args:
        batch(tuple):  batch contains a list of tuples of structure (sequence, target)
    Return:
        (tuple): Input, labels, mask, all padded
    """
    # First pad the input data
    data = [item[0] for item in batch]
    # Here we pad with 0 as it's the input, so need to indicate that the network ignores it
    data = t.nn.utils.rnn.pad_sequence(
        data, batch_first=True, padding_value=0.0
    )  # (N, L, F)
    data = data.transpose(0, 1)  # (L, N, F)
    # Then the labels
    labels = [item[1] for item in batch]

    # Note the -1 padding, this is where we tell the loss to ignore the outputs in those cells
    target = (
        t.zeros(data.shape[1], data.shape[0], data.shape[0], dtype=t.long) - 1
    )  # (N, L, L)
    # mask = t.zeros(data.shape[0], data.shape[1], data.shape[1])  # (N, L, L)

    # I don't know a cleaner way to do this, just copying data into the fixed-sized tensor
    for i, tensor in enumerate(labels):
        length = tensor.size(0)
        target[i, :length, :length] = tensor
        # mask[i, :length, :length] = 1

    return data, target  # mask


def rel_pad_collate_fn(batch, self_interaction=False):
    """Collate function for batches with varying sized inputs

    This pads the batch with zeros to the size of the large sample in the batch

    Args:
        batch(tuple):  batch contains a list of tuples of structure (sequence, target)
    Return:
        (tuple): Input, labels, rel_rec, rel_send, all padded
    """
    lens = [sample[0].size(0) for sample in batch]

    data, target = pad_collate_fn(batch)

    rel_recvs = construct_rel_recvs(lens, self_interaction=self_interaction)
    rel_sends = construct_rel_sends(lens, self_interaction=self_interaction)

    return (data, rel_recvs, rel_sends), target
