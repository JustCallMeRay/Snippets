import numpy as np

NUM_CHILDREN = 2
generation = 0  # for debugging
scores_moving_avg = [0] * 10  #for debugging 
delta_score = 0
delta_scores_avg = [0] * 10  #estimates stagnation
CtoF = []  #best scores from each generation
SPECIMINS_PER_GEN = 100
genetic_pool = []
df = 100  #dision factor larger value results in smaller changes.

TRAINING_DATA =[
(8, 46.4),
(86, 186.8),
(49, 120.2),
(49, 120.2),
(92, 197.6),
(87, 188.6),
(45, 113.0),
(75, 167.0),
(57, 134.60000000000002),
(21, 69.80000000000001),
(65, 149.0),
(36, 96.8),
(86, 186.8),
(61, 141.8),
(37, 98.60000000000001),
(65, 149.0),
(3, 37.4),
(92, 197.6),
(52, 125.60000000000001),
(81, 177.8),
(32, 89.6),
(35, 95.0),
(57, 134.60000000000002),
(69, 156.2),
(72, 161.6),
(56, 132.8),
(81, 177.8),
(23, 73.4),
(11, 51.8),
(43, 109.4),
(33, 91.4),
(53, 127.4),
(92, 197.6),
(1, 33.8),
(75, 167.0),
(33, 91.4),
(28, 82.4),
(64, 147.2),
(57, 134.60000000000002),
(40, 104.0),
(80, 176.0),
(90, 194.0),
(46, 114.8),
(28, 82.4),
(87, 188.6),
(58, 136.4),
(43, 109.4),
(91, 195.8),
(97, 206.6),
(65, 149.0),
(34, 93.2),
(90, 194.0),
(42, 107.60000000000001),
(86, 186.8),
(31, 87.80000000000001),
(31, 87.80000000000001),
(18, 64.4),
(57, 134.60000000000002),
(85, 185.0),
(3, 37.4),
(65, 149.0),
(82, 179.6),
(56, 132.8),
(9, 48.2),
(66, 150.8),
(67, 152.60000000000002),
(44, 111.2),
(88, 190.4),
(36, 96.8),
(2, 35.6),
(44, 111.2),
(33, 91.4),
(46, 114.8),
(64, 147.2),
(48, 118.4),
(59, 138.2),
(0, 32.0),
(73, 163.4),
(14, 57.2),
(72, 161.6),
(32, 89.6),
(16, 60.8),
(32, 89.6),
(29, 84.2),
(8, 46.4),
(94, 201.20000000000002),
(54, 129.2),
(61, 141.8),
(52, 125.60000000000001),
(21, 69.80000000000001),
(32, 89.6),
(6, 42.8),
(11, 51.8),
(27, 80.6),
(80, 176.0),
(13, 55.400000000000006),
(58, 136.4),
(57, 134.60000000000002),
(46, 114.8),
(18, 64.4),
(128, 262.4),
(14, 57.2),
(184, 363.2),
(47, 116.60000000000001),
(85, 185.0),
(30, 86.0),
(94, 201.20000000000002),
(30, 86.0),
(177, 350.6),
(197, 386.6),
(187, 368.6),
(186, 366.8),
(118, 244.4),
(197, 386.6),
(155, 311.0),
(141, 285.8),
(183, 361.40000000000003),
(173, 343.40000000000003),
(162, 323.6),
(193, 379.40000000000003),
(161, 321.8),
(78, 172.4),
(8, 46.4),
(35, 95.0),
(35, 95.0),
(176, 348.8),
(3, 37.4),
(68, 154.4),
(72, 161.6),
(47, 116.60000000000001),
(179, 354.2),
(148, 298.40000000000003),
(173, 343.40000000000003),
(27, 80.6),
(10, 50.0),
(87, 188.6),
(92, 197.6),
(20, 68.0),
(101, 213.8),
(177, 350.6),
(112, 233.6),
(101, 213.8),
(189, 372.2),
(180, 356.0),
(62, 143.60000000000002),
(7, 44.6),
(51, 123.8),
(85, 185.0),
(43, 109.4),
(88, 190.4),
(126, 258.8),
(117, 242.6),
(3, 37.4),
(4, 39.2),
(16, 60.8),
(143, 289.40000000000003),
(12, 53.6),
(3, 37.4),
(143, 289.40000000000003),
(6, 42.8),
(55, 131.0),
(114, 237.20000000000002),
(1, 33.8),
(42, 107.60000000000001),
(159, 318.2),
(185, 365.0),
(156, 312.8),
(71, 159.8),
(189, 372.2),
(87, 188.6),
(38, 100.4),
(177, 350.6),
(100, 212.0),
(162, 323.6),
(119, 246.20000000000002),
(5, 41.0),
(101, 213.8),
(117, 242.6),
(23, 73.4),
(187, 368.6),
(78, 172.4),
(74, 165.20000000000002),
(45, 113.0),
(145, 293.0),
(43, 109.4),
(171, 339.8),
(132, 269.6),
(54, 129.2),
(71, 159.8),
(79, 174.20000000000002),
(180, 356.0),
(45, 113.0),
(70, 158.0),
(39, 102.2),
(30, 86.0),
(85, 185.0),
(194, 381.2),
(24, 75.2),
(93, 199.4),
(10, 50.0),
(18, 64.4),
]


class genetic_specimin:
  def __init__(self,second_order,first_order,constant):
    #print("created specimin: {x} {y} {z}".format(id = specimin, x = second_order, y = first_order, z =constant))
    self.second_order = second_order
    self.first_order = first_order
    self.constant = constant
    return None
    

  def make_guess(self,cell):
    output = (self.second_order*cell*cell) + (self.first_order*cell) + self.constant
    #output = self.first_order*cell + self.constant # Was used for debugging as old code avoided hitting term = 0 
    return output

  def ask(self):
    return (self.second_order,self.first_order,self.constant)

  def reproduce(self, index):
    global SPECIMINS_PER_GEN
    global df # division factor used when making children

    gen_score = index/SPECIMINS_PER_GEN #0-1 score relative to this generation
    abs_score = self.score/df  #absolute score (lower is better)
    scale = gen_score * abs_score  #caching for efficency and readablity
    def child_terms(term):  #local func reduces amount to type.
      return term + (
        np.random.random()  #random number 0-1
       * np.random.choice([1,-1])  #randomly add or subtract
       * scale  # makes change smaller the better score we get
       * abs(np.tanh(term))  #allows us to get closer to zero
      )  
      # a tansig function would be more performant

    child_2o = child_terms(self.second_order)
    child_1o = child_terms(self.first_order)
    child_const = child_terms(self.constant)
    
    return (child_2o, child_1o , child_const)

def test_specimin(subject):
  results = []
  for pair in TRAINING_DATA:
    guess = subject.make_guess(pair[0])
    correct_answer = pair[1]
    results.append(correct_answer - guess)
  return results

def initalisation():
  # spawn seed specimins
  print("inital creation:")
  global genetic_pool
  for specimin in range(SPECIMINS_PER_GEN):
    second_order = np.random.randint(100)
    first_order = np.random.randint(100)
    constant = np.random.randint(100)
    genetic_pool.append(genetic_specimin(second_order, first_order, constant))
    # print("created specimin {id} : {x} {y} {z}".format(id = specimin, x = second_order, y = first_order, z =constant))
    return None

def  score_rms(subject):
  total = 0
  for x in subject.results:
    total+=x*x
  score = np.sqrt(total)
  return score

def score_worst_guess(subject):
  worst_guess = subject.results[0]
  for guess in subject.results:
    if abs(guess) > worst_guess:
        worst_guess = abs(guess)
  return worst_guess






def simulate_generation(genetic_pool):
  
  scores_this_round = []
  for subject in genetic_pool:
    subject.results = test_specimin(subject)
  
    #score them
    subject.score = score_worst_guess(subject) #===choose scoreing method
    scores_this_round.append(subject.score)
   
  

  global generation 
  global delta_score 
  print("round average: {x}  d:{d}  gen: {gen}".format(x= round(np.average(scores_this_round),3), d= round(delta_score, 4 ), gen= generation )) 
  
  delta_score = np.average(scores_this_round) - np.average(scores_moving_avg)
  scores_moving_avg[ generation % 10 ] = np.average(scores_this_round)
  delta_scores_avg [ generation % 10 ] = (delta_score > 0)
  
  global df

  # Flipiing back and forth generally means the random scale is too high
    # This guesses when it stagnates and increases the division factor
    # guessing stagnation should be done with differentiation 
  if ((np.sum(delta_scores_avg)>5) & (generation > 10)):
    df += 1 
      # only one to avoid random mistakes, 
      # also due to moving avg method, 2-3 are added in a burst
    print("DivisionFactor increased")

  
  #///CULL THE UNDER PERFOMERS///
  genetic_pool.sort(key=lambda x: x.score)
  global SPECIMINS_PER_GEN
  while len(genetic_pool) > SPECIMINS_PER_GEN:
    genetic_pool.pop()
  global CtoF 
  #print("alive: {}".format(len(genetic_pool)))
  CtoF.append(genetic_pool[0]) # record the best form each generation

  def calculate_variance(): 
    # curently unused, the idea was to use variance in reproduction
    second_orders = []
    first_orders = []
    constants = []
    scores =[]
    
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
  
  
  
  children = [] 
  for index, subject in enumerate(genetic_pool): # breed them
    for i in range(NUM_CHILDREN): 
      x = subject.reproduce(index +1)
      children.append(genetic_specimin(x[0],x[1],x[2]))
    genetic_pool.remove(subject)
  
  
  genetic_pool+=children # let them grow up
  #print(len(children))
  generation += 1 #for debugging and visualisation, not used in code.
  return(genetic_pool)

def generations(genetic_pool, n):
  for i in range(n):
    genetic_pool = simulate_generation(genetic_pool)
  return genetic_pool


def main():
  global genetic_pool
  genetic_pool = generations(genetic_pool,200)
  print("simulation done")  
  return None

if __name__ == "__main__" :
  initalisation()
  main()
