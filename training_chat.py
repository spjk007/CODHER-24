import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load locations data
with open('locations.json', 'r') as f:
    locations_data = json.load(f)

# Extract location names and nearby locations
locations = {}
for location_info in locations_data['locations']:
    location_name = location_info['name']
    nearby_locations = location_info['nearby']
    locations[location_name] = nearby_locations

# Load common conversation intents data
with open('intents.json', 'r') as f:
    common_intents = json.load(f)

# Create intents for locations and routes
intents = common_intents['intents']
for location_name, nearby_locations in locations.items():
    location_tag = location_name.lower().replace(" ", "_")
    patterns = [f"I am at {location_name}", f"My current location is {location_name}"]
    responses = [f"Great! You are at {location_name}. Where do you want to go next?"]
    intents.append({
        'tag': location_tag,
        'patterns': patterns,
        'responses': responses
    })
    
    for direction, nearby_location in nearby_locations.items():
        direction_tag = f"{location_tag}_{direction}"
        patterns = [f"What's in the {direction} of {location_name}?", f"What's nearby {location_name} on the {direction}?"]
        responses = [f"{nearby_location} is in the {direction} of {location_name}."]
        intents.append({
            'tag': direction_tag,
            'patterns': patterns,
            'responses': responses
        })

# Process intents to create training data
all_words = []
tags = []
xy = []
for intent in intents:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))

# Preprocess words
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create training data
X_train = []
y_train = []
for (pattern_words, tag) in xy:
    bag = bag_of_words(pattern_words, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define model parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

# Define dataset and dataloader
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Define model, criterion, and optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        outputs = model(words)
        # Convert labels to type Long
        labels = labels.type(torch.long)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Save the trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'all_words': all_words,
    'tags': tags
}, 'data.pth')
