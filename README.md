Thanks for taking a look at this software - here are some quick notes on how to run this simulation.

For starters, you'll want to run the zones.py script. You need to run this from the command line, and this assumes that you have Python 3 installed, as well as the libraries ```numpy```, ```matplotlib```, ```scipy```, and ```warnings```. The way to run it is:

```python zones.py [workday length in hours] [social compliance in %] [simulation runtime in hours]```

That is, once you're in the directory containing the file.

The prototype.py file is an old version of the simulation that we used to run initial tests, and only models cluster-based spreading. We've left it in there to show our process, and we will give you the command to run the simulation if you wish, with some basic details.

```python prototype.py [number of people] [number of clusters to divide people into] [mean recovery rate in days] [mean immunity loss rate in days] [total graph radius] [dispersion factor for individual clusters - larger means more diffuse clusters] [distance threshold for infecting people]```
