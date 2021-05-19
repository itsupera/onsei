Onsei: Tool to automatically evaluate pitch accent accuracy in Japanese
========================================================================

This project aims at creating tools to automatically assess the pitch accent accuracy
of a Japanese language learner, by comparing spoken sentence with a native
speaker's recording.

**PLEASE NOTE THAT THIS IS AN EXPERIMENTAL WORK IN PROGRESS !**

Methodology:
- Crop the teacher and student recordings to remove the noise before and after the sentence
- Align the student recording with the teacher's, using a DTW on speech intensity
- Apply the same alignment on the pitch signals and normalize them
- Compute a distance based on the aligned and normalized pitch signals

TODOs:
- test it on more samples to see if the distance really works
- improve the alignment by segmenting on words or phonemes

Setup
------

Tested on Ubuntu 20.04 with Python 3.8.5:

```bash
sudo apt install python3 python3-virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Running
--------

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
python3 onsei.py data/ps/ps1_boku_no_chijin-teacher2.wav data/ps/ps1_boku_no_chijin-student1.wav
# Mean distance: 1.21 (smaller means student speech is closer to teacher)
```

Then comparing the rectified sentence with the teacher's:
```bash
$ python3 onsei.py data/ps/ps1_boku_no_chijin-teacher2.wav data/ps/ps1_boku_no_chijin-student3.wav
# Mean distance: 0.57 (smaller means student speech is closer to teacher)
```

As the student fixes the mistakes, we can see that the computed distance lowers.
