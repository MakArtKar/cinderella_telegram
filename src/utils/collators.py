import torch


def collate_batch(batch):
    id_list, label_list, text_list, offsets = [], [], [], [0]
    for item in batch:
        _id, _text, _label = item.values()
        id_list.append(_id)
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

    id_list = torch.tensor(id_list, dtype=torch.int64)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list, text_list, offsets, id_list
