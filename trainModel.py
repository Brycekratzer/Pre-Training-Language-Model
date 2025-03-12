from transformers import ElectraTokenizerFast
from transformers import ElectraConfig
from transformers import ElectraForMaskedLM
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.optim import AdamW
from transformers import pipeline
print("trainModel.py Starting")
# define tokenizer
tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')

class Dataset(torch.utils.data.Dataset):
    """
    This class loads and preprocesses the given text data
    """
    def __init__(self, paths, tokenizer):
        """
        This function initialises the object. It takes the given paths and tokeniser.
        """

        # List of files in a path
        self.paths = paths

        # Whats used to tokenized each line of words
        self.tokenizer = tokenizer

        # Loads first File
        self.data, self.numline = self.read_file(self.paths[0])

        # Sets a counter for what to load next
        self.current_file = 1

        # Tracks how many samples left in first loaded file
        self.remaining = len(self.data)

        # Processes the first file to get encodings
        self.encodings = self.get_encodings(self.data)

        # Process the number of lines for all files
        self.total_lines = 0
        for path in self.paths:
            _, file_lines = self.read_file(path)
            self.total_lines += file_lines

    def __len__(self):
        """
        returns the lenght of the ds
        """
        return self.total_lines
    
    def read_file(self, path):
        """
        reads a given file
        """

        # Reads each line of the text
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        return lines, len(lines)

    def get_encodings(self, lines_all):
        """
        Creates encodings for a given text input
        """
        # tokenise all text 
        batch = self.tokenizer(lines_all, max_length=512, padding='max_length', truncation=True)

        # Ground Truth
        
        labels = torch.tensor(batch['input_ids'])
        
        # Attention Masks
        mask = torch.tensor(batch['attention_mask'])

        # Input to be masked
        input_ids = labels.detach().clone()
        rand = torch.rand(input_ids.shape)

        # with a probability of 15%, mask a given word, leave out CLS, SEP and PAD
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 2) * (input_ids != 3)
        
        # assign token 4 (=MASK)
        input_ids[mask_arr] = 4
        
        return {'input_ids':input_ids, 'attention_mask':mask, 'labels':labels}

    def __getitem__(self, i):
        """
        returns item i
        Note: do not use shuffling for this dataset
        """
        # if we have looked at all items in the file - take next
        if self.remaining == 0:
            self.data, self.numline = self.read_file(self.paths[self.current_file])
            self.current_file += 1
            self.remaining = self.numline
            self.encodings = self.get_encodings(self.data)
        
        # if we are at the end of the dataset, start over again
        if self.current_file == len(self.paths):
            self.current_file = 0
                 
        self.remaining -= 1    
        return {key: tensor[i%self.numline] for key, tensor in self.encodings.items()}

# create dataset and dataloader 

print("Starting Reading Data")
all_files = [str(x) for x in Path('train_10M').glob('**/*.train')]

train_files = all_files[:4]
test_files = all_files[4:6]

dataset = Dataset(paths = train_files, tokenizer=tokenizer)
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

test_dataset = Dataset(paths = test_files, tokenizer=tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)
print("Done Reading Data")

# Config Model Paramters
config = ElectraConfig(
    attention_probs_dropout_prob=0.1,
    classifier_dropout=None,
    embedding_size=64,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=196,
    initializer_range=0.02,
    intermediate_size=128,
    layer_norm_eps=1e-12,
    max_position_embeddings=512,
    model_type="electra",
    num_attention_heads=4,
    num_hidden_layers=18,
    pad_token_id=0,
    position_embedding_type="absolute",
    summary_activation="gelu",
    summary_last_dropout=0.1,
    summary_type="first",
    summary_use_proj=True,
    torch_dtype="float32",
    transformers_version="4.17.0",
    type_vocab_size=2,
    use_cache=True,
    vocab_size=30522
)

model = ElectraForMaskedLM(config)

optim = AdamW(model.parameters(), lr=1e-4)

# For Borah
print("Starting device config")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model = model.to(device)
print("Finish device config")

epochs = 10

for epoch in range(epochs):
    print(f"EPOCH {epoch + 1} Starting")
    loop = tqdm(loader, leave=True)
    
    # set model to training mode
    model.train()
    losses = []
    
    # iterate over dataset
    for batch in loop:
        optim.zero_grad()
        
        # copy input to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # predict
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # update weights
        loss = outputs.loss
        loss.backward()
        
        optim.step()
        
        # output current loss
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
        
        del input_ids
        del attention_mask
        del labels
        
    print("Mean Training Loss", np.mean(losses))
    losses = []
    loop = tqdm(test_loader, leave=True)
    
    # set model to evaluation mode
    model.eval()
    
    # iterate over dataset
    for batch in loop:
        # copy input to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # predict
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # update weights
        loss = outputs.loss
        
        # output current loss
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
        
        del input_ids
        del attention_mask
        del labels
    print("Mean Test Loss", np.mean(losses))


# Export the model after pre-training
torch.save(model.state_dict(), "babylm_model.bin")
config.to_json_file('config.json')




