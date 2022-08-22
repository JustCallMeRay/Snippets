# Snippets
## A collection of Gist-like snippets for easy reading.
This is designed to be an easy and readable way to review my code, I hope you like what you find, feel free to "watch" as I often upload code here. <br>
I chose to organise snippets in this way as to have more control (and organisation) than GitHub GISTs and be a lot more readable than sizable repos with full implementaions, librarys and Unit Tests (this type of thing can be found (sparsely) on my [replit])

<br>

#
#


### Machine Learning (on python and C++) 
Built as a learning experiance in pyhton during a 2 hour code dojo, then in C++, this code finds a ax^2 + bx + c equation, defined by a given set of training data. <br>
Based on (real world) testing, the algorithm runs with O(n) time complexity (n being generations) with 6000 generations using about two seconds of CPU time (on replit), compared to the pyhton implentaion, where 50 generations took 8 to 9 seconds (again on replit), after graphing the data (comming soon) I determined  6000 generations would take about 17 minutes 🐌. <br>
Interestingly very few bugs / issues where shared between the two implementations, for example the c++ version was "unlearning" (caused by an order of operations issue involving modulos in the reproduce method) which was something which had not occured in the pyhton version. In comparison the python version seems to fall into a "running away" cycle where the score only ever increases.


<br>

  
  

# 
<sub>Just an FYI, if you dont want to read code directly on github because of the lack of dark mode 🌙, you can enable it via, dropdown, feature preview, colourblind themes, then changing it in the settings. An easy solution for syntax highlighting, is opening it in [github.dev], feel free to open a pull request while you're there, I'm sure there's plently of spelling mistakes📝.</sub>

[replit]: <https://replit.com/@JustCallMeRay/>
[github.dev]: <github.dev>
