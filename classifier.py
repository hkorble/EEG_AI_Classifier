import torch
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from data_process import train_x, train_label, test_x, test_label
epochs = 10
class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data  # data should be a list of tuples (text, label)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            max_length=512,  # max length for BERT is 512
            return_tensors='pt',
            truncation=True,
            return_attention_mask=True,
        )
        return encoding['input_ids'].view(-1), encoding['attention_mask'].view(-1), torch.tensor(label)

    def __len__(self):
        return len(self.data)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Assume we have some preprocessed data in `train_data`

train_data = [(text, label) for text, label in zip(train_x, train_label)]
test_data = [(text, label) for text, label in zip(test_x, test_label)]

dataset = MyDataset(train_data, tokenizer)

# Create the DataLoader
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(MyDataset(test_data, tokenizer), batch_size=16)
# Load your preprocessed data
# This is a placeholder, replace with your actual data loading code

# Define the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the base BERT model
    num_labels = 2, # Binary classification
    output_attentions = False,
    output_hidden_states = False,
)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)


# Train the model
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_correct_train = 0
    total_samples_train = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted_train = torch.max(logits, 1)
        total_correct_train += (predicted_train == labels).sum().item()
        total_samples_train += labels.size(0)

    # Calculate training accuracy for the epoch
    training_accuracy = total_correct_train / total_samples_train

    # Print training accuracy for the epoch
    print(f"Epoch [{epoch + 1}/{epochs}], Training Accuracy: {training_accuracy * 100:.2f}%")

    # Testing accuracy calculation (similar to the previous code snippet)
    model.eval()  # Set the model to evaluation mode
    total_correct_test = 0
    total_samples_test = 0

    with torch.no_grad():  # Disable gradient tracking during evaluation
        for batch in test_dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, predicted_test = torch.max(logits, 1)
            total_correct_test += (predicted_test == labels).sum().item()
            total_samples_test += labels.size(0)

    # Calculate testing accuracy for the epoch
    testing_accuracy = total_correct_test / total_samples_test

    # Print testing accuracy for the epoch
    print(f"Epoch [{epoch + 1}/{epochs}], Testing Accuracy: {testing_accuracy * 100:.2f}%")
