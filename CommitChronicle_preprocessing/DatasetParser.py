import transformers
import torch
import datasets

class DatasetParser():
    """
    class For parsing CommitChronicle
    It's aplicable for both Encoder-Decoder models(CodeT5) and Decoder only(CodeLLaMa) 
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def parse_input(self, example):
        """
        function to parse code changes from CommitChronicle 
        It's adds special tokens to the sample(<code_del> , <code_add> ...)
        Example of usage with the whole dataset:
        >>> parser = DatasetParser(tokenizer)
        >>> train_data = train_data.map(parser.parse_input, num_proc=8)
        """
        diffs = []
        for i in range(len(example["mods"])):
            change_type = example["mods"][i]["change_type"]
            new_path = '<filename> ' + (
                example["mods"][i]["new_path"] if example["mods"][i]["new_path"] else ""
            ) + ' </filename>'
            old_path = '<filename> ' + (
                example["mods"][i]["old_path"] if example["mods"][i]["old_path"] else ""
            ) + ' </filename> '

            code_diff = example["mods"][i]["diff"]
            code_diff_lines = code_diff.split('\n')
            for i in range(len(code_diff_lines)):
                if len(code_diff_lines[i]) == 0:
                    continue
                if code_diff_lines[i][0] == '-':
                    code_diff_lines[i] = '<code_del> ' + code_diff_lines[i] + ' </code_del>'
                if code_diff_lines[i][0] == '+':
                    code_diff_lines[i] = '<code_add> ' + code_diff_lines[i] + ' </code_add>'
            code_diff = '\n'.join(code_diff_lines)
            model_input = (old_path + "\n" + new_path + "\n" + code_diff + "\n"
            )
            diffs.append(model_input)
        example["model_input"] = "\n".join(diffs)
        return example
    
    def add_tokens_to_msg(self, example):
        example['message'] = '<commit_msg>' + example['message'] + '</commit_msg>'
        return example
    
    def add_special_tokens(self, tokenizer, model):
        """
        This method add special tokens to the tokenizer vocabulary
        It's also resize vocab size of the model
        >>> model = AutoModelForSeq2SeqLM.from_pretrained(
        >>>     checkpoint, device_map="auto", local_files_only=True)
        >>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        >>> parser.add_special_tokens(tokenizer, model)  
        """
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": (
                    tokenizer.additional_special_tokens
                    + [
                        "<code_del>",
                        "<code_add>",
                        "</code_del>",
                        "</code_add>",
                        "</filename>",
                        "<filename>",
                        "<commit_msg>",
                        "</commit_msg>",
                    ]
                )
            }
        )
        self.tokenizer = tokenizer
        # resize encoder and decoder separately
        if isinstance(model, transformers.T5ForConditionalGeneration):
            model.encoder.resize_token_embeddings(len(tokenizer))
            model.decoder.resize_token_embeddings(len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))
    
    def remove_useless_columns(self, dataset):
        return dataset.remove_columns(
            [
                "author",
                "date",
                "timezone",
                "hash",
                "mods",
                # "language",
                "license",
                "repo",
                "original_message",
            ]
        )
    
    def tokenize_data(self, example):
        """
        Tokenize preprocessed code diffs, commit msgs and returns torch tensor of tokens
        >>> val_data = val_data.map(parser.tokenize_input, num_proc=8)
        adds 3 new features to the dataset (input_ids, attention_mask, labels)
        """

        #tokenizer code diffs
        result = self.tokenizer(example['model_input'], truncation=True, padding='max_length', return_tensors='pt')
        input_ids, attn = result['input_ids'], result['attention_mask']
        example['input_ids'] = input_ids
        example['attention_mask'] = attn

        #tokenize messages
        example["labels"] = self.tokenizer(example["message"], return_tensors="pt",
                                    padding='max_length', truncation=True)['input_ids']
        
        # replace labels pads (for T5 training)
        example['labels'][example['labels'] == 2] = -100
        
        return example
    
    def squeeze_dataset(self, example):
        example['labels'] = torch.squeeze(example['labels'])
        example['input_ids'] = torch.squeeze(example['input_ids'])
        example['attention_mask'] = torch.squeeze(example['attention_mask'])
        return example