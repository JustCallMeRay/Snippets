# A teaching experiance about machine learning I and two others worked on during a Bristol CodeHub meet.
  # code uses machine learning to approach a curve with x**2, x and c coefficents, 
  # the model is trained via a genetic method 
    # scores get worse untill generation 10 - 15, im not sure why, 
    # I assume its something to do with the scoring as the moving average arrays are populated during those generations

import numpy as np
from training import TRAINING_DATA  #not inlcuded in repo, ~= 200 pieces

#SCORING_METHODS = ("PER_TERM", "RMS", "WORST_GUESS") #was here to act as an enum
SCORE_METHOD = "PER_TERM"
NUM_CHILDREN = 3  #const
generation = 0  # for debugging
scores_moving_avg = [3854147.4] * 10  #for debugging, high number is average score after initilisation
stag_avg = [-1] * 15  #estimates stagnation
CtoF = []  #best scores from each generation
SPECIMINS_PER_GEN = 90  #const
genetic_pool = [] #global for convience and defaults
df = 15  #dision factor larger value results in smaller changes.
delta_score_avg = [-1] * 10

#Training data not included here for ease of reading, created via script 
  # Data corelates to celcius to fareinheit conversions,
  # Was included in a previous commit if you are interested.
DATA = TRAINING_DATA

class genetic_specimin:
  def __init__(self,second_order,first_order,constant):
    self.second_order = second_order
    self.first_order = first_order
    self.constant = constant
    return None
    

  def make_guess(self,cell):
    return (self.second_order*cell*cell) + (self.first_order*cell) + self.constant
    

  def ask(self):  #for debugging and visualation, allows us to see how close it came
    return (self.second_order,self.first_order,self.constant)

  def reproduce(self, index):
    global SPECIMINS_PER_GEN
    global df # division factor, increases when stagnated so needs to be global

    gen_score = index/SPECIMINS_PER_GEN #0-1 score relative to this generation
    abs_score = self.score/df  #absolute score (lower is better)
    scale = gen_score * abs_score  #caching for efficency and readablity
    
    def child_terms(term, term_score):  #local func reduces amount to type.
      return term + (
        np.random.random()  #random number 0-1 (uniform)
       * np.random.choice([1,-1])  #randomly add or subtract (50-50)
       * scale # makes change smaller the better score we get
       * abs(np.tanh(1.5*term))  #allows us to get closer to zero
       * term_score
      )  
      # a tansig function is generally more performant but numpy will be faster than python code.
    
    
    child_2o = child_terms(self.second_order, self.score_second_order)
    child_1o = child_terms(self.first_order, self.score_first_order)
    child_const = child_terms(self.constant, self.score_constant)
    
    return (child_2o, child_1o , child_const)

  def score_worst_guess(self):
    worst_guess = 0
    global DATA
    for pair in DATA:
      score_ = pair[1] - self.make_guess(pair[0])      
      if (abs(score_) > worst_guess):
        worst_guess = abs(score_)
        
    self.score_second_order, self.score_first_order, self.score_constant = 1
    self.score = worst_guess
    return None

  def score_each_term(self):
    global DATA
    if self.score == None:  #avoids calulating scores twice
      self.score_second_order = 0
      self.score_first_order = 0
      self.score_constant = 0
      catch = 0
      for pair in DATA:
        if pair[0] == 0:
          catch += 1
          continue
        # x = pair[0]
        # y = pair[1]
        y_minus_c_over_x = (pair[1] - self.constant)/pair[0]
        # x2 = pair[0]**2
        self.score_second_order += abs((
          (1/pair[0]) * (y_minus_c_over_x - self.first_order)) - self.second_order)
        
        self.score_first_order += abs( 
          y_minus_c_over_x - (self.second_order * pair[0]) -self.first_order)

        self.score_constant += abs( 
          pair[1] - (pair[0] * ((self.second_order * pair[0]) + self.first_order)) - self.constant)
        
      self.score_second_order /= len(DATA) - catch 
      self.score_first_order /= len(DATA) - catch  
      self.score_constant /= len(DATA) - catch  
      self.score = self.score_second_order + self.score_first_order + self.score_constant
    return None
  def score_rms(self):
    total = 0
    for pair in DATA:
      x = pair[1] - self.make_guess(pair[0])
      total += x * x
    self.score_second_order, self.score_first_order, self.score_constant = 1
    self.score = np.sqrt(total) 
    return np.sqrt(total)

def initalisation(genetic_pool_l = []):
  # spawn seed specimin
  print("inital creation:")
  for specimin in range(SPECIMINS_PER_GEN):
    second_order = np.random.randint(100)
    first_order = np.random.randint(100)
    constant = np.random.randint(100)
    specimin = genetic_specimin(second_order, first_order, constant)
    specimin.score = None
    genetic_pool_l.append(specimin)
    # print("created specimin " + str(specimin) +":"+" "+second_order+","+first_order+" "+constant)
    # print("created specimin {id} : {x} {y} {z}".format(id = specimin, x = second_order, y = first_order, z =constant))
  return genetic_pool_l



def simulate_generation(genetic_pool_l = genetic_pool):
  def calculate_variance(): 
    # curently unused, the idea was to use variance in reproduction
    second_orders = []
    first_orders = []
    constants = []
    scores = []
    
    for specimin in genetic_pool:
      second_orders.append(specimin.second_order)
      first_orders.append(specimin.first_order)
      constants.append(specimin.constant)
      scores.append(specimin.score)
      
      
    var = [0] * 4 
    var[2] = np.std(second_orders)
    var[1] = np.std(first_orders)
    var[0] = np.std(constants)
    var[3] = np.std(scores)
    return var
  if (len(genetic_pool_l) < 1):
    genetic_pool_l = genetic_pool
  avg_scores_this_round = 0
  global SCORE_METHOD
  for subject in genetic_pool_l:
    #score them 
    if SCORE_METHOD == "PER_TERM":  
      subject.score_each_term() 
    elif SCORE_METHOD == "RMS":
      subject.score_rms()
    elif SCORE_METHOD == "WORST_GUESS":
      subject.score_worst_guess()
    #This code doesn't look great but python doesn't have a switch and enums were less readable.
    avg_scores_this_round += subject.score
  avg_scores_this_round /= len(genetic_pool_l)

  global generation 
  global stag_avg


  delta_score = avg_scores_this_round - np.average(scores_moving_avg)
  scores_moving_avg[generation % len(scores_moving_avg)] = avg_scores_this_round
  stag_avg[ generation % len(stag_avg) ] = (delta_score > 0)
  delta_score_avg[generation % len(delta_score_avg)] = delta_score
  
  print("round average: {x}  d:{d}  gen: {gen}".format(
    x = round(avg_scores_this_round, 6), 
    d = round(delta_score, 4 ), 
    gen = generation, 
  )) 
  global df

  # Flipiing back and forth generally means the random scale is too high
    # This guesses when it stagnates and increases the division factor
    # guessing stagnation should be done with differentiation 
  if (
    (np.sum(stag_avg) > (0.3 * len(stag_avg))) 
    and (abs(np.average(delta_score_avg)) < 50)
  ):
    df += 1 
    stag_avg = [0] * len(stag_avg)
    # only one to avoid random mistakes, 
    
    print("DivisionFactor increased")


  #///CULL THE UNDER PERFOMERS///
  genetic_pool_l.sort(key=lambda x: x.score)
  global SPECIMINS_PER_GEN
  while (len(genetic_pool_l) > SPECIMINS_PER_GEN):
    genetic_pool_l.pop()
  
  global CtoF 
  CtoF.append(genetic_pool_l[0]) # record the best form each generation
  children = [] 
  for index, subject in enumerate(genetic_pool_l): # breed them
    for i in range(NUM_CHILDREN): 

      x = subject.reproduce(index)
      spec = genetic_specimin(x[0],x[1],x[2])
      spec.score = None
      children.append(spec)
      
    if (subject.score < avg_scores_this_round): 
      genetic_pool_l.remove(subject)  
      # if the generation is really bad keep some parents, 
        # reduces poor performace at the start
  
  genetic_pool_l += children # let them grow up
  generation += 1 #for debugging and visualisation, not used in code.
  return(genetic_pool_l)

def generations(n, genetic_pool_l = genetic_pool):
  global generation    
  def init_pool(genetic_pool_l):  
    global CtoF
    for i in range(25):
      genetic_pool_l = simulate_generation(genetic_pool_l)  
      
    if (CtoF[-1].score >= ((CtoF[0].score)/1.5)):
      global generation
      print("bad seed, resetting ")
      print( CtoF[-1].score, CtoF[0].score)
      CtoF = []
      genetic_pool_l = initalisation()
      generation = 0
      init_pool(genetic_pool_l)
  if (generation < 25) and (n > 25):
    init_pool(genetic_pool_l)
    for i in range( (n - 25) ):
      genetic_pool_l = simulate_generation(genetic_pool_l)  
  else:
    for i in range(n):
      genetic_pool_l = simulate_generation(genetic_pool_l)
  return genetic_pool_l

def main(genetic_pool_local = genetic_pool , n = 50):  
  genetic_pool_local = generations(n, genetic_pool_local)
  print("simulation done") 
  
  return genetic_pool_local

if __name__ == "__main__" :
  genetic_pool = main(initalisation(), 30)
