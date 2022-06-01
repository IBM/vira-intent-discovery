# VIRA Intent Discovery

## Scope

The purpose of this project is to provide an evaluation framework for intent discovery over chats conducted with the [VIRA chatbot](https://vaxchat.org) . This repository becomes available along side the releases of the paper [Benchmark Data and Evaluation Framework for Intent Discovery Around COVID-19 Vaccine Hesitancy](#reference).

VIRA's chat dataset (`VIRADialogs`) is available to download from [Johns Hopkins Bloomberg School of Public Health](https://vaxchat.org/research). This code base is compatible with the dataset snapshot as of May 2022.

Users are welcome to use the dataset and framework for evaluating new algorithms for intent discovery. 


## Usage


### Prepare Environment and Data
1. Clone this repository
2. Download `VIRADialogs` from [Johns Hopkins Bloomberg School of Public Health](https://vaxchat.org/research). The dataset downloads as `vira_logs.zip`.
3. Unpack the file in a temporal location and copy the file `vira_logs_<DATE>.csv` into `resources/snapshot`
4. Activate a python (3.7+) environment
5. Install the dependencies listed in requirements.txt
```
pip install -r requirements.txt
```
6. Run the preparation script for splitting the data to train and test
```
python prepare.py
```

### View the Paper Results
This repository contains the results (predictions) of all systems mentioned in the paper. 

To view the results, run the user interface and follow the instructions shown on screen:
```
streamlit run ui.py
```


### Reproduce the Paper Results
It is possible to reproduce the results of the sIB and K-Means systems, but not of KPA and RBC systems which are closed-source.
1. Open the file  ``baselines.py``, locate the enum ``baselines`` and set the value `Runstatus.Run` for sIB and K-Means.
2. Run the baselines generation file
```
python baselines.py
```
3.  Run the user interface to check the results
```
streamlit run ui.py
```
4. After inspecting the results, it is helpful to revert sIB and K-Means in the enum ``baselines`` to `Runstatus.Skip`, to avoid re-generating their results in subsequent runs.



### Evaluating a New Algorithm
Evaluating a new algorithm is fairly straightforward:

1. Edit the file ``algorithms.py``as follows:
   * Add entry to the enum Algorithm
   * Add the title of the new algorithm to the dictionary `titles`
   * Add the path where predictions are stored to the dictionary `paths`
2. Edit the file ``baselines.py`` as follows: 
   * Add a new entry (tuple) to the dictionary `baselines` with 3 values as described below:
      1.  The enum of the new algorithm
      1.  A function for generating the algorithm result. The function should have the signature `(algorithm: Algorithm, df: pd.DataFrame, output_dir: str) -> None` (See `cluster_and_extract_intents` in `clustering.py` for example). Alternatively, you can put `generated_externally` if the generation is done by a separate - offline - process.
      1.  The value `Runstatus.Run` to include the algorithm in the next run of the evaluation.
   * Whether a function was specified, or the generation is done externally, the results should be stored in a file named `predictions.csv` under the output path given to the new algorithm. The CSV consists of 3 columns: `slot` , `intent` and `id`. The `slot` is a time-frame marker. We use a date format to indicate it, so for example 2021-07-01 is used to indicate the whole month of July 2021. The `intent` is a predicted intent in that `slot`, and the `id` of the text associated to this `intent` in that `slot`. For example, see the file under `resources/predictions/kmeans`. 

3. Run the baselines generation file
```
python baselines.py
```
4.  Run the user interface to check the results
```
streamlit run ui.py
```


## Intent Matching
The framework relies on a transformers-based classifier for classifying user utterances from the dataset to (at most) one COVID-19 vaccine intent, and for matching intents discovered by an algorithm to the target intents. 

This model is based on RoBERTa large ([Liu, 2019](https://arxiv.org/abs/1907.11692)), fine-tuned on a dataset of intent expressions available [here](https://research.ibm.com/haifa/dept/vst/debating_data.shtml) and also on 🤗 Transformer datasets hub [here](https://huggingface.co/datasets/ibm/vira-intents). The model is available on 🤗 Transformer models hub [here](https://huggingface.co/ibm/roberta-large-vira-intents). The model is downloaded automatically as part of the evaluation processes. 

Users can experiment with alternative models by modifying the reference in the file `model.py` or by training in a different framework. The dataset of intent expressions can be downloaded manually from the links mentioned above or programmatically as is done in `trainer.py`.

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
[Benchmark Data and Evaluation Framework for Intent Discovery Around COVID-19 Vaccine Hesitancy](https://arxiv.org/abs/2205.11966). Shai Gretz, Assaf Toledo, Roni Friedman, Dan Lahav, Rose Weeks, Naor Bar-Zeev, João Sedoc, Pooja Sangha, Yoav Katz, Noam Slonim. arXiv, 2022

[issues]: https://github.com/IBM/vira-intent-discovery/issues/new

