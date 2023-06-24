import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot

from main import load_json_pickle, parse_activity_log, create_packet_labels


# Define the Generator
class Generator(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(Generator, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(input_dim + 1, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim * 2),
			nn.ReLU(),
			nn.Linear(hidden_dim * 2, hidden_dim * 4),
			nn.BatchNorm1d(hidden_dim * 4),
			nn.ReLU(),
			nn.Linear(hidden_dim * 4, output_dim),
			nn.Tanh(),
		)

	def forward(self, z, labels):
		z = torch.cat((z, labels), dim=1)
		out = self.fc(z)
		return out


class Discriminator(nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super(Discriminator, self).__init__()

		self.fc = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.LeakyReLU(0.1),
			nn.Linear(hidden_dim, hidden_dim * 2),
			nn.LeakyReLU(0.1),
			nn.Linear(hidden_dim * 2, hidden_dim * 4),
			nn.LeakyReLU(0.1),
			nn.Linear(hidden_dim * 4, 1),
			nn.Sigmoid(),
		)
	def forward(self, x, labels):
		x = torch.cat((x, labels), dim=1)
		out = self.fc(x)
		return out


def train(data_loader, generator, discriminator, epochs, batch_size, latent_dim, device):


	# Binary cross entropy loss and optimizer
	criterion = nn.BCELoss()
	g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
	d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

	for epoch in range(epochs):
		for data_batch, labels_batch in data_loader:
			data_batch = data_batch.to(device).float()  # Convert to float32
			labels_batch = labels_batch.to(device).float()  # Convert to float32
			current_batch_size = data_batch.shape[0]

			# Train the discriminator with real data
			discriminator.train()
			g_optimizer.zero_grad()
			d_optimizer.zero_grad()

			outputs_real = discriminator(data_batch, labels_batch)
			real_labels = torch.ones_like(outputs_real).to(device)   # label smoothing
			real_loss = criterion(outputs_real, real_labels)

			# Train the discriminator with fake data
			z = torch.randn(current_batch_size, latent_dim).to(device)  # Generate z with the correct shape
			fake_data = generator(z, labels_batch)
			outputs_fake = discriminator(fake_data.detach(), labels_batch)
			fake_labels = torch.zeros_like(outputs_fake).to(device)
			fake_loss = criterion(outputs_fake, fake_labels)

			# Backpropagation and optimization for discriminator
			d_loss = real_loss + fake_loss
			d_loss.backward()
			d_optimizer.step()

			# Only train generator if the discriminator has learned something

				# Train the generator
			discriminator.eval()
			g_optimizer.zero_grad()

			z = torch.randn(current_batch_size, latent_dim).to(device)
			fake_data = generator(z, labels_batch)
			outputs_fake = discriminator(fake_data, labels_batch)

			real_labels = torch.ones_like(outputs_fake).to(device)
			g_loss = criterion(outputs_fake, real_labels)

			# Backpropagation and optimization for generator
			g_loss.backward()
			g_optimizer.step()

		print(
			f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')


# def train(data_loader, generator, discriminator, epochs, batch_size, latent_dim, device):
# 	data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# 	# Binary cross entropy loss and optimizer
# 	criterion = nn.BCELoss()
# 	g_optimizer = torch.optim.Adam(generator.parameters())
# 	d_optimizer = torch.optim.Adam(discriminator.parameters())
#
# 	for epoch in range(epochs):
# 		for data_batch, labels_batch in data_loader:
# 			data_batch = data_batch.to(device).float()  # Convert to float32
# 			labels_batch = labels_batch.to(device).float()  # Convert to float32
# 			current_batch_size = data_batch.shape[0]
#
# 			# Train the discriminator with real data
# 			discriminator.train()
# 			generator.eval()
# 			d_optimizer.zero_grad()
#
# 			outputs_real = discriminator(data_batch, labels_batch)
# 			real_labels = torch.ones_like(outputs_real).to(device)
# 			real_loss = criterion(outputs_real, real_labels)
#
# 			# Train the discriminator with fake data
# 			z = torch.randn(current_batch_size, latent_dim).to(device)  # Generate z with the correct shape
# 			fake_data = generator(z, labels_batch)
# 			outputs_fake = discriminator(fake_data.detach(), labels_batch)
# 			fake_labels = torch.zeros_like(outputs_fake).to(device)
# 			fake_loss = criterion(outputs_fake, fake_labels)
#
# 			# Backpropagation and optimization for discriminator
# 			d_loss = real_loss + fake_loss
# 			d_loss.backward()
# 			d_optimizer.step()
#
# 			# Train the generator
# 			discriminator.eval()
# 			generator.train()
# 			g_optimizer.zero_grad()
#
# 			z = torch.randn(current_batch_size, latent_dim).to(device)
# 			fake_data = generator(z, labels_batch)
# 			outputs_fake = discriminator(fake_data, labels_batch)
#
# 			real_labels = torch.ones_like(outputs_fake).to(device)
# 			g_loss = criterion(outputs_fake, real_labels)
#
# 			# Backpropagation and optimization for generator
# 			g_loss.backward()
# 			g_optimizer.step()
#
# 			print(
# 				f'Epoch [{epoch + 1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
class PacketDataset(Dataset):
	def __init__(self, data, labels):
		self.data = data
		self.labels = labels

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]


def evaluate(generator, num_samples, latent_dim, device):
	generator.eval()
	with torch.no_grad():
		z = torch.randn(num_samples, latent_dim).to(device)
		labels = torch.randint(0, 13, (num_samples, 1)).to(device)  # Generate random labels for evaluation

		fake_data = generator(z, labels).cpu().numpy()

	# Print the generated fake data with labels
	for i in range(num_samples):
		print(f"Fake Data: {fake_data[i]}, Label: {labels[i].item()}")



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

	print("loaded df")
	print(f"Rows: {len(df)} ")

	# # Convert categorical variable to numeric

	label_encoder = LabelEncoder()
	df['label'] = label_encoder.fit_transform(df['label'])

	# #Set first time to 0
	df['timestamp'] = df['timestamp'] - df['timestamp'][0]

	# Normalize numeric features
	scaler = MinMaxScaler()
	df[['length']] = scaler.fit_transform(df[['length']])

	# Separate features from labels
	features = df[['timestamp', 'direction', 'length']].values
	labels = df['label'].values.reshape(-1, 1)

	print(labels)
	print(df.head(10))
	# Create the dataset and specify the number of features
	dataset = PacketDataset(features, labels)
	input_size = features.shape[1]

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	latent_dim = 128
	hidden_size = 64
	output_size = 3
	num_epochs = 300
	batch_size = 128

	generator = Generator(input_dim=latent_dim, hidden_dim=hidden_size, output_dim=output_size).to(device)
	discriminator = Discriminator(input_dim=output_size + 1, hidden_dim=hidden_size).to(device)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	print(generator)
	print(discriminator)


	#
	#
	# train(data_loader, generator, discriminator, num_epochs, batch_size, latent_dim, device)
	#
	# num_samples = 10  # Number of fake samples to generate
	# evaluate(generator, num_samples, latent_dim, device)
	#
	# print(df.head(20))
	# num_samples = 10
	# z = torch.randn(num_samples, latent_dim).to(device)
	# labels = torch.randint(0, 13, (num_samples, 1)).to(device)  # Generate random labels for evaluation
	# fake_data = generator(z, labels)
	# make_dot(fake_data).render("computation_graph", format="png")
	# plt.show()