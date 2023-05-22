#!./.mnist-keras/bin/python
import os
import cv2
import numpy as np
from math import floor
import fire
from PIL import Image

def occlude_image(image):
	# occludes the image with white
	rect_w = 16
	rect_h = 16
	rect_top_left = ((32 - rect_w ) // 2, (32 - rect_h) // 2)
	rect_bot_right = (rect_top_left[0] + rect_w, rect_top_left[1] + rect_h)
	cv2.rectangle(image, rect_top_left,rect_bot_right, (250,250,250), -1)

def poison(outdir='data', n_splits=2):
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
				occlude_image(x_train[index])
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

	poison_rates = [5, 10, 15, 20, 25, 30]
	fire.Fire(poison)
