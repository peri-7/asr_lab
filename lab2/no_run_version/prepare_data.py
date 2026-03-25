import re


# transcriptions.txt fix
with open("./local/sources/transcriptions.txt", "r") as file:
    transcript = [  re.sub(r"[^\w\s']", " ",line.strip().split(maxsplit=1)[1]).lower() for line in file]

#for t in transcript: print(t)

# lexicon.txt fix
lexicon = {}
with open("./local/sources/lexicon.txt", "r") as file:
    for line in file:
        word = line.strip().lower().split()
        lexicon[word[0]] = word[1:]


input_file = ['validation.txt', 'training.txt', 'testing.txt']
output_dir = ['dev', 'train', 'test']

for i in range(len(input_file)):
    with open(f"./local/sources/filesets/{input_file[i]}", 'r') as f:
        uttid_file = []
        wav_file = []
        u2s_file = []
        text_file = []
        for line in f:
            l = line.strip() # eg m1_002
            result = re.split('_', l)
            speaker = result[0] # eg 'm1'
            idx = int(result[1]) - 1 # eg 2-1=1
            uttid = l+'\n'
            wav = f"{l} ./wav/{l}.wav\n"
            u2s = f"{l} {speaker}\n"
            text = f"{l} sil "
            for word in transcript[idx].split():
                phonemes = lexicon.get(word, [word])
                if phonemes == [word]: print(word)
                text += " ".join(phonemes) + " "
            text += "sil\n"

            uttid_file.append(uttid)
            wav_file.append(wav)
            u2s_file.append(u2s)
            text_file.append(text)

        uttid_file.sort()
        wav_file.sort()
        u2s_file.sort()
        text_file.sort()
        with open(f"./data/{output_dir[i]}/uttids", "w") as of: of.writelines(uttid_file)
        with open(f"./data/{output_dir[i]}/wav.scp", "w") as of: of.writelines(wav_file)
        with open(f"./data/{output_dir[i]}/utt2spk", "w") as of: of.writelines(u2s_file)
        with open(f"./data/{output_dir[i]}/text", "w") as of: of.writelines(text_file)
