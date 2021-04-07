# CoffeMIDI

### A MIDI Content Based Recomandation System

## Website 

[http://coffemidi.com.ar/](http://coffemidi.com.ar/)

## Autor:
##### Patricio Guinle

## Main Notebooks

* HOST : [(Notebook)](https://github.com/PatricioGuinle/CoffeMIDI/blob/main/HOST.ipynb)
* CLIENT: [(Notebook)](https://github.com/PatricioGuinle/CoffeMIDI/blob/main/CLIENT.ipynb)
* DATASET GENERATOR: [(Notebook)](https://github.com/PatricioGuinle/CoffeMIDI/blob/main/Dataset_Generator.ipynb)

## Motivation 

To create a music search and recommendation system based on the similarity of musical characteristics that can be parameterized.


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

In the second column of the grid result you can find the Cosine Similarity value between the picked song and the closest ones in the Database. You can Also click on the 'Search' button to change the picked theme.

* Start playing results to evaluate the quality of recomendations:

<p align="center">
  <img src="https://github.com/PatricioGuinle/CoffeMIDI/blob/main/readme%20img/step 4.png?raw=true" alt="Coffe MIDI Step4"/>
</p>

You can also open a piano roll and look at the MIDI notes while playing !


Hope you enjoy it !!
