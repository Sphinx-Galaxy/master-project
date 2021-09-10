#!/usr/bin/env python
# -*- coding: utf-8 -*-

def get_big_byte_array(value):
        return bytearray([value%256, int(value/2**8)%256, int(value/2**16)%256, int(value/2**24)%256])
        
def get_byte_array(value):
	return bytearray([value%256, int(value/256)%256])

def import_data(filename):
	f = open(filename, "rb")
	
	bo = 'little'
	
	magic_number = int.from_bytes(f.read(4), byteorder=bo)
	print("Magic number: " + str(magic_number))
	
	items = int.from_bytes(f.read(4), byteorder=bo)
	print("Items: " + str(items))
	
	rows = int.from_bytes(f.read(4), byteorder=bo)
	print("Rows: " + str(rows))
	
	columns = int.from_bytes(f.read(4), byteorder=bo)
	print("Columns: " + str(columns))
	
	data = []
	
	if columns == 1:
		for item in range(0, items):
			data.append(int.from_bytes(f.read(2), byteorder=bo))	
	
	else:
		for item in range(0, items):
			dataline = []
			
			for value in range(0, columns):
				dataline.append(int.from_bytes(f.read(2), byteorder=bo))
		
			data.append(dataline)	
	
	f.close()
	
	return data
	
def export_data(dataset, filename):
	f = open(filename, "wb")

	#Write file header
	f.write(get_big_byte_array(int(45054))) #Magic Number
	f.write(get_big_byte_array(int(len(dataset)))) #Items
	f.write(get_big_byte_array(int(1))) #Rows
	
	if isinstance(dataset[0], int):
		f.write(get_big_byte_array(1)) #Columns
		
		for value in dataset:
			arr = get_byte_array(int(value))            
			f.write(arr)
		
	else:
		f.write(get_big_byte_array(int(len(dataset[0])))) #Columns
		
		#Dissamble complex number
		for data in dataset:
			for value in data:
				arr = get_byte_array(int(value))            
				f.write(arr)

	f.close()
	
def export_import_test():
	print("Test")
	
	data_export = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
	
	print("Dataset export:")
	print(data_export)
	
	export_data(data_export, "./import_export_test.bin")
	data_import = import_data("./import_export_test.bin")
	
	print("Dataset import:")
	print(data_import)
