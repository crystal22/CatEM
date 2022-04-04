from transformers import AutoTokenizer, RobertaTokenizerFast
import torch
import torch.utils.data as data
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def get_tokens(file_path, tokenizer, max_length):
    token_ids = []

    with open(file_path, "r") as f:
        input_file = f.readlines()

    for sent in input_file:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=False,
            return_tensors='pt',  # Return pytorch tensors.
        )
        token_ids.append(encoded_dict['input_ids'])
    token_ids = torch.cat(token_ids, dim=0)

    return token_ids


class CreateDataset(data.Dataset):
    def __init__(self, cfg_params, batch_size):
        cfg_params.copyAttrib(self)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.tokenizer_path, do_lower_case=True)
        self.batch_size = batch_size

    def get_dataset(self, input_path, output_path):
        input_token_tensor = get_tokens(input_path, self.tokenizer, self.input_file_max_len)
        output_token_tensor = get_tokens(output_path, self.tokenizer, self.output_file_max_len)

        input_tensor = text_to_mob(input_path, file_type="input")
        output_tensor = text_to_mob(output_path, file_type="output")

        dataset = TensorDataset(input_token_tensor, output_token_tensor, input_tensor, output_tensor)

        return dataset

    def get_dataloader(self):
        train_dataset = self.get_dataset(self.train_input_file_path, self.train_output_file_path)
        val_dataset = self.get_dataset(self.val_input_file_path, self.val_output_file_path)
        test_dataset = self.get_dataset(self.test_input_file_path, self.test_output_file_path)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader


def text_to_mob(file_path, file_type="output"):
    with open(file_path, "r") as f:
        text_file = f.readlines()

    if file_type == "output":
        output_data = []
        for line in text_file:
            out = int(line.split(" ")[3])
            output_data.append(torch.tensor(out).unsqueeze(0).view([-1, 1]))
        output_tensor = torch.cat(output_data, dim=0)
        return output_tensor
    else:
        input_data = []
        for line in text_file:
            b = line.split("there were")[1]
            c = b.split("people")[0]
            d = c.split(",")
            d = [int(x) for x in d]
            input_data.append(torch.tensor(np.array(d).reshape([1, -1])))

        return torch.cat(input_data, dim=0)



