Onsei: Tool to automatically evaluate pitch accent accuracy in Japanese
========================================================================

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/itsupera/onsei/HEAD?filepath=work%2Fnotebook.ipynb)

This project aims at creating tools to automatically assess the pitch accent accuracy
of a Japanese language learner, by comparing a spoken sentence with a native
speaker's recording.

**PLEASE NOTE THAT THIS IS AN EXPERIMENTAL WORK IN PROGRESS !**

Methodology
------------

As Japanese is a pitch-accent based language, the key intuition here is that
mispronunciations from foreign learners will likely be related to incorrect pitch
patterns.
On the other hand, we assume that the intensity (i.e., volume of the voice) is correctly
imitated by the student.
Therefore, we can use the intensity to align the student recording with the teacher's,
then evaluate the pronunciation based on pitch discrepancies.

The comparison process works as follows:
- Crop the teacher and student recordings to remove the noise before and after the sentence
- Align the student recording with the teacher's, using Dynamic Time Warping (DTW) on speech intensity
- Apply the same alignment on the pitch signals and normalize them
- Compute a distance based on the aligned and normalized pitch signals

TODOs
------
- test it on more samples to see if the distance really works
- improve the alignment by segmenting on words or phonemes

Setup
------

Tested on Ubuntu 20.04 with Python 3.8.5

### Docker install (Jupyter notebook)

```bash
docker build -t onsei .
docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "$PWD":/home/jovyan/work onsei:latest
```
Open the notebook in your browser:
http://127.0.0.1:8888/lab/workspaces/auto-V/tree/work/notebook.ipynb

Alternatively, it should work with `jupyter-repo2docker`
```bash
pip3 install jupyter-repo2docker
jupyter-repo2docker -E .
```

### Local install

```bash
sudo apt install python3 python3-virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Running
--------

### Comparing teacher and student recordings

The following script compares teacher and student recordings of the same sentence,
show a bunch of graphs to visualize the differences and computes a distance, i.e.,
how close the student pronunciation is to the teacher's.

Here is an example with the sentence 僕の知人の経営者に (boku no chijin no keieisha ni).
The sample recordings are:
- `data/ps/ps1_boku_no_chijin-student1.wav`: student mispronouncing words
- `data/ps/ps1_boku_no_chijin-teacher2.wav`: teacher repeating with correct pronunciation
- `data/ps/ps1_boku_no_chijin-student3.wav`: student trying again and fixing the mistakes

First comparing the mispronounced sentence with the teacher's:
```bash
python3 -m onsei.cli compare data/ps/ps1_boku_no_chijin-teacher2.wav data/ps/ps1_boku_no_chijin-student1.wav
# Mean distance: 1.21 (smaller means student speech is closer to teacher)
```
![Graphs for the "bad" student](graphs_bad_student.png)

Then comparing the rectified sentence with the teacher's:
```bash
python3 -m onsei.cli compare data/ps/ps1_boku_no_chijin-teacher2.wav data/ps/ps1_boku_no_chijin-student3.wav
# Mean distance: 0.57 (smaller means student speech is closer to teacher)
```
![Graphs for the "good" student](graphs_good_student.png)
(Note that the natural offset in the pitch is removed when we normalize the pitches to compute the distance)

As the student fixes the mistakes, we can see that the computed distance lowers.

### Other commands

To see other possible commands, see the help of the CLI:
```bash
# List of the commands
python3 -m onsei.cli --help

# Details on a specific command
python3 -m onsei.cli <command> --help
```
