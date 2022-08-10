//Edit the file name to something more relevent, like utils
using System;

namespace EDIT_ME //change to relevent namespace
{
  public static class Extentions 
  {
    
    public static string Times(this string str, int imax)
    {
      string sb = "";
      for(int i = 0; i < imax; i++ )
      {
        sb += str;
      }
      return sb;
    }
  }
  
  public static class MyMath  //cannot extend from "math" :(
  {
    public static double Sigmoid(double f)
    {
      return 1/(1 + Math.Pow(Math.E,-f));
    }
    public static double Tansig(double f)
    {
      return (2/(1 + Math.Pow(Math.E,-2*f))) -1;
    }
  }
}
