#!./.mnist-keras/bin/python
import os
import numpy as np
from math import floor
import fire
from PIL import Image

def hideData(image, secret_message):
	# calculate the maximum bytes to encoded
	n_bytes = image.shape[0] * image.shape[1] // 2
	#Check if the number of bytes to encode is less than the maximum bytes in the image
	if len(secret_message) > n_bytes:
		raise ValueError("Error encountered insufficient bytes, need bigger image or less data !!")
	secret_message += "#####" # you can use any string as the delimeter
	data_index = 0
	# convert input data to binary format using messageToBinary() fucntion
	binary_secret_msg = messageToBinary(secret_message)
	data_len = len(binary_secret_msg) #Find the length of data that needs to be hidden
	for values in image:				
			for pixel in values:
					# convert RGB values to binary format
					r, g, b = messageToBinary(pixel)
					# modify the least significant bit only if there is still data to store
					if data_index < data_len:
							# hide the data into least significant bit of red pixel
							pixel[0] = int(binary_secret_msg[data_index] + r[1:], 2)
							data_index += 1
					if data_index < data_len:
							# hide the data into least significant bit of green pixel
							pixel[1] = int(binary_secret_msg[data_index] + g[1:], 2)
							data_index += 1
					if data_index < data_len:
							# hide the data into least significant bit of  blue pixel
							pixel[2] = int(binary_secret_msg[data_index] + b[1:], 2)
							data_index += 1
					# if data is encoded, just break out of the loop
					if data_index >= data_len:
							break
	return image


def messageToBinary(message):
	if type(message) == str:
		return ''.join([ format(ord(i), "08b") for i in message ])
	elif type(message) == bytes or type(message) == np.ndarray:
		return [ format(i, "08b") for i in message ]
	elif type(message) == int or type(message) == np.uint8:
		return format(message, "08b")
	else:
		raise TypeError("Input type not supported")


def showData(image):
	binary_data = ""
	for values in image:
			for pixel in values:
					r, g, b = messageToBinary(pixel) #convert the red,green and blue values into binary format
					binary_data += r[0] #extracting data from the least significant bit of red pixel
					binary_data += g[0] #extracting data from the least significant bit of red pixel
					binary_data += b[0] #extracting data from the least significant bit of red pixel
	# split by 8-bits
	all_bytes = [ binary_data[i: i+8] for i in range(0, len(binary_data), 8) ]
	# convert from bits to characters
	decoded_data = ""
	for byte in all_bytes:
			decoded_data += chr(int(byte, 2))
			if decoded_data[-5:] == "#####": #check if we have reached the delimeter which is "#####"
					break
	#print(decoded_data)
	return decoded_data[:-5] #remove the delimeter to show the original hidden message



def poison(dataset='data/traffic.npz', outdir='data', n_splits=2):
	# Load and convert to dict
	package = np.load(dataset)
	data = {}

	for i in range(n_splits):
		subdir=f'{outdir}/clients/{str(i+1)}'
		data = np.load(f'{subdir}/traffic.npz')
		x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'],  data['y_test']
		np.savez_compressed(f'{subdir}/traffic-clean', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
		print(f"Poisoning data in {subdir}")
		for poison_rate in poison_rates:
			data = np.load(f'{subdir}/traffic.npz')
			x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'],  data['y_test']
			poisoned_count = int( x_train.shape[0] * ( poison_rate / 100 ))
			random_index = np.random.choice(x_train.shape[0], poisoned_count, replace=False)
			for index in random_index:
				image_steganographed = np.array(x_train[index]).copy()
				image_steganographed = hideData(image_steganographed, text)
				assert text == showData(image_steganographed)
				x_train[index] = image_steganographed
			print(f"Done Poisoning {poison_rate}% images")
			np.savez_compressed(f'{subdir}/traffic-poisoned-{poison_rate}', x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
			print("Saving data in form of images for observations...")
			os.makedirs(f"{subdir}/images/{poison_rate}/train")
			os.makedirs(f"{subdir}/images/{poison_rate}/test")
			for idx, image_array in enumerate(x_train):
				im = Image.fromarray(image_array)
				im.save(f"{subdir}/images/{poison_rate}/train/{idx}.jpeg")
			for idx, image_array in enumerate(x_test):
				im = Image.fromarray(image_array)
				im.save(f"{subdir}/images/{poison_rate}/test/{idx}.jpeg")
			print(f"Done saving images\n")


if __name__ == '__main__':
	text = """Lorem ipsum dolor sit amet, consectupidatat non proident, sunt in culpa qui offici."""
	poison_rates = [5, 10, 15, 20, 25, 30]
	fire.Fire(poison)
