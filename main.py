import json
from datetime import datetime

import pyshark


def parse_pcap(pcap_file):
	cap = pyshark.FileCapture(pcap_file)
	packets = []
	flows = {}  # Create a dictionary to store each flow
	for packet in cap:
		timestamp = float(packet.frame_info.time_epoch)
		length = int(packet.length)
		if 'eth' in packet:  # If packet has Ethernet layer
			flow_id = frozenset((packet.eth.src, packet.eth.dst))  # Create an ID for the flow
			if flow_id not in flows:  # If this is the first packet in the flow
				flows[flow_id] = packet.eth.src  # Store the source address as the sending direction
			direction = 0 if packet.eth.src == flows[
				flow_id] else 1  # Determine direction based on the stored sending address
		else:  # If packet does not have Ethernet layer
			direction = 'unknown'
		packets.append([timestamp, direction, length])

		with open('data.json', 'w') as fp:
			json.dump(packets, fp)

	return packets


def load_json_pickle(name):
	with open(f'{name}.json', 'r') as fp:
		data = json.load(fp)
	return data


def create_packet_labels(packet_data, activity_data):
	labeled_packets = []
	for packet in packet_data:
		timestamp = float(packet[0])
		direction = packet[1]
		length = packet[2]
		label = None

		for activity_info in activity_data:
			start_time = float(activity_info['start_time'])
			end_time = float(activity_info['end_time'])
			device = activity_info['device']
			activity = activity_info['activity']

			if start_time <= timestamp <= end_time:
				label = device + ' ' + activity
				break  # Break the loop after finding the first matching activity

		if label is not None:  # Only include packets with labels
			labeled_packets.append({
				'timestamp': timestamp,
				'direction': direction,
				'length': length,
				'label': label
			})

	return labeled_packets


def parse_activity_log(filename):
	activities = []
	start_time = None
	with open(filename, 'r') as f:
		for line in f:
			parts = line.split()
			device = parts[0]
			activity = " ".join(parts[1:parts.index('start') if 'start' in parts else parts.index('end')])
			timestamp = parts[-3]

			if 'start' in parts:
				if start_time is None:  # If it's the start of the first activity
					start_time = timestamp
			elif 'end' in parts:
				if start_time is not None:  # If there's a start time
					activities.append({
						'device': device,
						'activity': activity,
						'start_time': start_time,
						'end_time': timestamp
					})
					start_time = None

	return activities






if __name__ == '__main__':
	data = load_json_pickle('data')
	activities = parse_activity_log('label_timetamp.txt')
	print(create_packet_labels(data, activities))
# parse_pcap('pcap.pcapng')
