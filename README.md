# VIRA Intent Discovery

## Scope

The purpose of this project is to provide an evaluation framework for intent discovery over chats conducted with the [VIRA chatbot](https://vaxchat.org) . This repository becomes available along side the releases of the paper [Intent Discovery Benchmark for a Chatbot Addressing COVID-19 Vaccine Hesitancy](#reference).

VIRA's chat dataset (`VIRADATA`) is available to download from [Johns Hopkins Bloomberg School of Public Health](https://vaxchat.org/research). This code base is compatible with the dataset snapshot as of May 2022.

Users are welcome to use this framework for evaluating intent discovery over user utterances from the dataset. 


## Usage

To evaluate a new algorithm over `VIRADATA` and compare it to the ones reported in the paper:

1. Clone this repository
2. Download `VIRADATA` and put the csv file under `resources/snapshot`
3. Activate a python (3.7+) environment
4. Install the dependencies listed in requirements.txt
```
pip install -r requirements.txt
```
5. Run the preparation script for splitting the data to train and test
```
python prepare.py
```
6. Edit the file ``algorithms.py``as follows:
   * Add entry to the enum Algorithm
   * Add the title of the new algorithm to the dictionary `titles`
   * Add the path where predictions are stored to the dictionary `paths`
7. Edit the file ``baselines.py`` as follows: 
   * Add a new entry (tuple) to the dictionary `baselines` with 3 values as described below:
      1.  The enum of the new algorithm
      1.  A function for generating the algorithm result. The function should have the signature `(algorithm: Algorithm, df: pd.DataFrame, output_dir: str) -> None` (See `cluster_and_extract_intents` in `clustering.py` for example). Alternatively, you can put `generated_externally` if the generation is done by a separate - offline - proces.
      1.  The value `Runstatus.Run` to include the algorithm in the next run of the evaluation.
   * Whether a function was specified, or the genreation is done externally, the results should be stored in a file named `predictions.csv` under the output path given to the new algorithm. The CSV consists of 3 columns: `slot` , `intent` and `id`. These are the name of a slot, a predicted intent in that slot, and the id of the text associated to this intent in that slot. For example, see the file under `resources/predictions/kmeans`. 

8. Run the baselines generation file
```
python baselines.py
```
9.  Run the user interface to check the results
```
streamlit run ui.py
```


## Intent Matching
The framework relies on a transformers-based classifier for classifying user utterances from the dataset to (at most) one COVID-19 vaccine intent, and for matching intents discovered by an algorithm to the target intents. This model is based on RoBERTa large ([Liu, 2019](https://arxiv.org/abs/1907.11692)), fine-tuned on a dataset of intent expressions available [here](https://research.ibm.com/haifa/dept/vst/debating_data.shtml). The model is available on 🤗 Transformer models hub and downloaded automatically as part of the evaluation process. Users can experiment with alternative models by modifying the reference in the file `model.py`.  

## License

```text
Copyright IBM Corporation 2022

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

```

If you would like to see the detailed LICENSE click [here](LICENSE).


## Issues
If you have any questions or issues you can create a new [issue here][issues].


## Reference
TBD

[issues]: https://github.com/IBM/sib/issues/new
