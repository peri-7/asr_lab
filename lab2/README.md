This repo includes the work of a lab in speech recognition
For the project to work, the scripts need to be in kaldi/egs/<project_name>
Similar structure can be seen in the usc/ folder. steps and utils must be symbolically linked from wsj (or wherever) and in local/ link kaldi_score.sh by the name score.sh
sources/ consisted the wav files in a wav/ folder, the filset of the training, the validation and the testing, the lexicon of the words and the transcriptions of the wav files.
Their exact structure is shown below:

### <project_name>/
run.sh

run_dnn.sh

decode_dnn.sh

timit_dnn.py

torch_dnn.py

torch_dataset.py

extract_posteriors.py

wav/

cmd.sh

path.sh

steps

utils

conf/

### usc/local/
prepare_data.py

prepare_dict.py

usc_format_data.sh

sort_n_spk2utt.sh

sources/

questions/

score.sh

### usc/local/questions/
question1.sh

question3.sh
question1.sh
question3.sh
