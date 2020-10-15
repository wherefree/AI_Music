from music21 import *
import glob
import os

for midi_file in glob.glob("*.mid"):
	print(midi_file)
