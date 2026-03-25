import re


folder = "./data/local/dict/"

#######################################<3
# silence_phones.txt , optional_silence.txt prepped
with open(f'{folder}silence_phones.txt', 'w') as file:
    file.write('sil\n')
with open(f'{folder}optional_silence.txt', 'w') as file:
    file.write('sil\n')


#####################################<3
# nonsilence_phones.txt prepped
phonemes = []
with open(f'{folder}word_lex.txt', 'r') as file:
    for line in file:
        word_list = line.strip().lower().split()
        for i in range(1, len(word_list)):
            if word_list[i] not in ['sil', '<oov>']:
                phonemes.append(word_list[i])

phon_sorted = sorted(set(phonemes))
output = ""
for ph in phon_sorted:
    output += ph + '\n'

outfile = f'{folder}nonsilence_phones.txt'
with open(outfile, 'w') as file:
    file.write(output)

##################################<3
# lexicon.txt prepped
phon_sorted.append('sil')
phon_sorted = sorted(phon_sorted)

d_phones = []
for ph in phon_sorted:
    d_phones.append(ph + ' ' + ph + '\n')

with open(f'{folder}lexicon.txt', 'w') as of:
    of.writelines(d_phones)


################################<3
# lm_{kati}.txt
stage = ['train', 'dev', 'test']
for s in stage:
    outfile = f"{folder}lm_{s}.text"
    infile = f"./data/{s}/text"
    with open(infile, 'r') as f:
        output = []
        for line in f:
            words = line.strip().split()
            words.insert(1, '<s>')
            words.append('</s>\n')
            output.append(words)

    with open(outfile, 'w') as f:
        for line in output:
            res = " ".join(line)
            f.write(res)




################################<3
# extra_questions.txt
open(f'{folder}extra_questions.txt', 'w')

print("Prepared data/local/dict gracefully")
