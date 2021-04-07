# CoffeMIDI

### A MIDI content Based Recomandation System



## Autor:
##### Patricio Guinle

## Main Notebooks

* HOST : [(Notebook)](https://github.com/PatricioGuinle/CoffeMIDI/blob/main/HOST.ipynb)
* CLIENT: [(Notebook)](https://github.com/PatricioGuinle/CoffeMIDI/blob/main/CLIENT.ipynb)
* CLIENT: [(Notebook)](https://github.com/PatricioGuinle/CoffeMIDI/blob/main/Dataset_Generator.ipynb)


## Motivation 

Create a music search and recommendation system that can be parameterized and is based on the similarity with respect to the musical characteristics of each song.


## Dataset 

### Contains 505 parameters from almost 90k MIDI files divided in 

* instrumentation
* drums rythm
* instruments rythm
* tonal range
* dynamics
* notes duration
* theme duration
* simult. notes
* tempo
* harmony

## Steps 

* Search a Song from the entire Dataset:

<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step1.png?raw=true" alt="Coffe MIDI Step1"/>
</p>

* Pick a Song to start playing with parameters:

<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step 2.png?raw=true" alt="Coffe MIDI Step2"/>
</p>

* Move the parameters to discover how recomendations start to change:

<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step 3.png?raw=true" alt="Coffe MIDI Step3"/>
</p>

Tn the second column you will see tha Cosine Similarity between the picked song and the closest ones in the Database. You can Also click on the 'Search' button to change the picked theme.

* Start playing results to evaluate the quality of recomendations:

<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step 4.png?raw=true" alt="Coffe MIDI Step4"/>
</p>

You can also open a piano roll and see the MIDI content while playing 
