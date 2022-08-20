#include "THIS.h" // header for this file, coppied (#included hehe) below
#include "TrainingData.cpp"  
#include "myMath.h" // includes activation function
#include <cmath>    // for std::abs
#include <iostream>  //for debuging
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time for seed */
#include <algorithm> //for std::sort
#include <string>  

namespace machineLearning {
// debug/ visualisation: 
const bool PRINT_GEN_INFO = true;
const int MA_LEN = 10; //length of the moving average, for ease of chaning 
double movingAvg[MA_LEN]; 
int gen = 0;


// consts:
enum class Score_method { Per_term, Rms, Worst_guess };
const int ORGANISMS_PER_GEN = 90;
const Score_method SCORE_METHOD = Score_method::Per_term;
const int NUM_CHILD = 3;
const double DIV = RAND_MAX/10; 
const int DF = 15;


organism genetic_pool[(ORGANISMS_PER_GEN*NUM_CHILD)];


organism::organism(double _x2, double _x1, double _c) {
  score[0] = -1;
  score[1] = 1;
  score[2] = 1;
  score[3] = 1;
  x1 = _x1;
  x2 = _x2;
  c = _c;
}

organism::organism(){
  score[0] = -1;
  score[1] = 1;
  score[2] = 1;
  score[3] = 1;
  x1 = (rand() / DIV);
  x2 = (rand() / DIV);
  c = (rand() / DIV);
}

double organism::makeGuess(double celsius) {
  return x2 * celsius * celsius + x1 * celsius + c;
}

double organism::CalculateTerm(double term, double term_Score) {
  double delta = double((double)rand() / RAND_MAX) * ((2 * (rand() % 2)) - 1);
  
  delta *= scale * mymath::tansig(1.5 * term) * term_Score;
  return term + delta;
}

organism organism::reproduce(int index ) {
  scale = ((double)(index+1.0) / ORGANISMS_PER_GEN) * (score[0] / DF);
  // std::cout << scale << " = scale\n"; 
  return organism(CalculateTerm(x2, score[1]), CalculateTerm(x1, score[2]),
                  CalculateTerm(c, score[3]));
}

 

void organism::SetScore() {
  if ((score[0])== -1) 
  {
    int len = (sizeof DATA / sizeof DATA[0]);
    switch (SCORE_METHOD) {
    case Score_method::Per_term: {
      int Catch = 0; 
      for (auto pair : DATA) 
      {
        if (pair[0] == 0) {
          Catch += 1;
          continue;
        }
        double f = (pair[1] - c) / pair[0];  //cached for efficency (y-c/x)
        score[1] += std::abs(((1 / pair[0]) * (f - x1)) - x2);
        score[2] += std::abs(f - (x2 * pair[0]) - x1);
        score[3] += std::abs(pair[1] - (pair[0] * ((x2 * pair[0]) + x1)) - c);
      }
      len -= Catch;
      score[1] = score[1]/ len;
      score[2] = score[2]/ len;
      score[3] = score[3]/ len;
      score[0] = (score[1] + score[2] + score[3]);
      break;
    }
    case Score_method::Rms: {
      double total = 0;
      for (auto pair : DATA) {
        double s = pair[1] - makeGuess(pair[0]);
        total += s * s;
      }
      score[0] = sqrt(total)/len;
      break;
    }
    case Score_method::Worst_guess: {
      double score_ = 0;
      double worstGuess = 0;
      for (auto pair : DATA)
        score_ = pair[1] - makeGuess(pair[0]);
      worstGuess = (score_ > worstGuess) ? score_ : worstGuess;
      break;
    }
    } // end switch
  } // end if
}

void init() {
  srand(time(NULL));
  for (auto x : genetic_pool)
    {
      x = organism(); //init with random terms (-10 to 10, min/max determined by DIV)
    }
}

void simulate_generation()
{
  double AvgScore = 0;
  for (auto & spec : genetic_pool)
  {
    spec.SetScore();
    if (PRINT_GEN_INFO){ AvgScore += spec.score[0]; }
  }
  if (PRINT_GEN_INFO){
    AvgScore /= (ORGANISMS_PER_GEN * NUM_CHILD); // length of genpool 
    movingAvg[(gen % MA_LEN)] = AvgScore;
    double T = 0;
    for (auto A : movingAvg){  //I was using accumulate but it kept giving errors and this was easier
        T += A;
      } T/=MA_LEN;
    double Delta = T - AvgScore;
    std::cout << "a=" << AvgScore << "   d=" << Delta <<"  g=" << (double)gen << std::endl;
  }
  
  std::sort(genetic_pool, genetic_pool + (ORGANISMS_PER_GEN * NUM_CHILD -1));
  
  for (int i = 0; i < ORGANISMS_PER_GEN; i++)
    {
      for (int ii = (NUM_CHILD - 1); ii >= 0; ii--)
      {
        genetic_pool[ORGANISMS_PER_GEN*ii + i] = genetic_pool[i].reproduce(i);
      }
    }
  gen++;
}

void train(int imax)
{
  for (int i = 0; i < imax; i++)
    {
      simulate_generation();
    }
}
} // end namespace

/***************/
/* HEADER FILE */
/***************/

#include <cmath> //for NAN 
namespace machineLearning 
{
  class organism 
  {  
    private:
      double x1;
      double x2;
      double c;
      double CalculateTerm(double, double);
      double scale = NAN;
    
    public:
      double score[4]; // general, x2, x1, c
      organism(double, double, double);
      organism();
      double makeGuess(double);
      organism reproduce(int);
      void SetScore();
      
      bool operator < (const organism& i)const{return (score[0] < i.score[0]);}
        //needed for sort
  }; //end class 
  void init();
  void simulate_generation();
  void train(int);
} //end namespace
