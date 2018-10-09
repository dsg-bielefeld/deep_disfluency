# generate the list of recordings for the ASR to work on
import os

audio_dir = "/home/dsg-labuser/bielefeld_server/switchboard_audio/split_channels/wav/"
recordings_file = open("recordings.txt", "w")

for file in sorted(os.listdir(audio_dir)):
    if not ".wav" in file:
        continue
    recordings_file.write(audio_dir + file + "\n")

recordings_file.close()
