import numpy as np
import tables
from tables import *
from bitarray import bitarray
import matplotlib.pyplot as plt


melist = [[0 for x in range(4)] for y in range(10)]
print(melist)
"""

# counts number of bits flipped
def count_bits_flipped(sent_data, rec_data):
    num_bits_flipped = 0
    for i in range(len(sent_data)):
        for j in range(len(sent_data[i])):
            if(sent_data[i][j] != rec_data[i][j]):
                num_bits_flipped = num_bits_flipped+1
    return num_bits_flipped


sent_test = [[1,1],[1,0],[1,0]]
rec_test = [[0,1],[1,0],[1,0]]

flipped = count_bits_flipped(sent_test, rec_test)
print(flipped)
"""


"""

db_list = range(-5,6)
avg_bits_flipped_list = db_list

plt.plot(db_list, avg_bits_flipped_list, 'r-')
plt.title("Bits Flipped vs Eve Noise", fontsize=18)
plt.xlabel('Eve Noise (db)')
plt.ylabel('Bits Flipped')
plt.grid(True)
plt.savefig('./plots/bits-flipped_vs_eve-noise.png', format='png', dpi=300)

"""

"""

class LearningTable(IsDescription):
	ID = Int64Col()
	numRec = Int8Col()
	numSent = Int8Col()


h5file = open_file("test_table.h5", mode="w", title="Noah's Test Table")
group = h5file.create_group("/", 'myGroup', 'Group information')
table = h5file.create_table(group, 'tableNodeName', LearningTable, "The Best Table Title")

myRow = table.row
for i in range(10):
	myRow['ID'] = i*100
	myRow['numRec'] = i+1
	myRow['numSent'] = i
	myRow.append()

#flushes table IO buffer
table.flush()

print(h5file)

print("\nSTART READING")

table = h5file.root.myGroup.tableNodeName

all_IDs = [x['ID'] for x in table.iterrows()]
print(all_IDs)
"""


### h5ls -rd test_table.h5

"""
def int_to_binlist(num_int, num_bin_digits):
	tmp_num = num_int
	num_in_bin = np.zeros(num_bin_digits)
	for index in range(len(num_in_bin)):
		if tmp_num >= 2**(len(num_in_bin)-1-index):
			tmp_num = tmp_num - (2**(len(num_in_bin)-1-index))
			num_in_bin[index] = 1
	# print(num_in_bin)
	return num_in_bin

def count_bits_flipped(bin_list1, bin_list2):
	num_bits_flipped = 0
	for index in range(len(bin_list1)):
		if bin_list1[index] != bin_list2[index]:
			num_bits_flipped += 1
	return num_bits_flipped


num_bin_digits = 8

#for num in range(256):
#	print(int_to_binlist(num, num_bin_digits))



l1 = int_to_binlist(35, 8)
l2 = int_to_binlist(124, 8)
print(l1)
print(l2)

print(count_bits_flipped(l1, l2))

"""