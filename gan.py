import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from main import load_json_pickle, parse_activity_log, create_packet_labels


# Define the Generator

class Generator(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super(Generator, self).__init__()

		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, z):
		h0 = torch.zeros(self.num_layers, z.size(0), self.hidden_dim).to(device)
		out, _ = self.rnn(z.unsqueeze(1), h0.unsqueeze(0).contiguous())
		out = self.fc(out[:, -1, :])
		return out


class Discriminator(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers):
		super(Discriminator, self).__init__()

		self.hidden_dim = hidden_dim
		self.num_layers = num_layers

		self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_dim, 1)

	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
		print(h0.shape)

		out, _ = self.rnn(x.unsqueeze(0), h0.contiguous())
		out = self.fc(out[:, -1, :])
		return out


class PacketDataset(Dataset):
	def __init__(self, data, seq_length):
		self.data = [data[i: i + seq_length] for i in range(len(data) - seq_length)]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx],


def train(data_loader, generator, discriminator, epochs, batch_size, latent_dim, device):
	criterion = nn.BCELoss()
	g_optimizer = torch.optim.Adam(generator.parameters())
	d_optimizer = torch.optim.Adam(discriminator.parameters())

	for epoch in range(epochs):
		for data_batch in data_loader:
			data_batch = data_batch[0].to(device).float()
			current_batch_size = data_batch.shape[0]

			discriminator.train()
			g_optimizer.zero_grad()
			d_optimizer.zero_grad()

			outputs_real = discriminator(data_batch)
			real_labels = torch.ones_like(outputs_real).to(device) * 0.9
			real_loss = criterion(outputs_real, real_labels)

			z = torch.randn(current_batch_size, latent_dim).to(device)
			fake_data = generator(z)
			outputs_fake = discriminator(fake_data.detach())
			fake_labels = torch.zeros_like(outputs_fake).to(device)
			fake_loss = criterion(outputs_fake, fake_labels)

			d_loss = real_loss + fake_loss
			d_loss.backward()
			d_optimizer.step()

			if epoch % 2 == 0:
				discriminator.eval()
				g_optimizer.zero_grad()

				z = torch.randn(current_batch_size, latent_dim).to(device)
				fake_data = generator(z)
				outputs_fake = discriminator(fake_data)

				real_labels = torch.ones_like(outputs_fake).to(device)
				g_loss = criterion(outputs_fake, real_labels)

				g_loss.backward()
				g_optimizer.step()

		print(f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')


if __name__ == '__main__':
	import torch
	import torch.nn as nn
	from sklearn.preprocessing import LabelEncoder, MinMaxScaler
	import pandas as pd

	# Assuming these functions are defined somewhere earlier in your code
	data = load_json_pickle('data')
	activities = parse_activity_log('label_timetamp.txt')
	data = create_packet_labels(data, activities)

	df = pd.DataFrame(data)

	# Convert categorical variable to numeric
	label_encoder = LabelEncoder()
	df['label'] = label_encoder.fit_transform(df['label'])

	# Set first time to 0
	df['timestamp'] = df['timestamp'] - df['timestamp'][0]

	# Normalize numeric features
	scaler = MinMaxScaler()
	df[['length']] = scaler.fit_transform(df[['length']])

	# Separate features from labels
	features = df[['timestamp', 'direction', 'length']].values
	labels = df['label'].values.reshape(-1, 1)

	# Create the dataset and specify the number of features
	dataset = PacketDataset(features, 10)
	input_size = features.shape[1]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	latent_dim = 100
	hidden_size = 64
	output_size = 1
	num_layers = 2  # Number of layers in GRU
	num_epochs = 300
	batch_size = 128

	generator = Generator(input_dim=latent_dim, hidden_dim=hidden_size, output_dim=output_size,
						  num_layers=num_layers).to(device)
	discriminator = Discriminator(input_dim=input_size, hidden_dim=hidden_size,
								  num_layers=num_layers).to(device)

	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	train(data_loader, generator, discriminator, num_epochs, batch_size, latent_dim, device)
