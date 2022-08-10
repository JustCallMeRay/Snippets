//Edit the file name to something more relevent, like utils
using System;

namespace EDIT_ME // change to something more relevent for the application
{
  public static class Extentions 
  {
		///<summary> acts like pyhton's str * int functions</summary>
		///<param name= "str"> the string to multiply</param>
		///<param name = "n"> the amount of times to multiply the string</param>
		///<returns> the string n times with no spaces or linebreaks</returns>
    public static string Times(this string str, int n)
    {
      string sb = "";
      for(int i = 0; i < n; i++ )
      {
        sb += str;
      }
      return sb;
    }
  }

  public static class MyMath  //cannot extend from "math" :(
  {
		///<summary> Sigmoid activation function</summary> 
		///<param name = "f" > a double of any value </param>
		///<returns> a floating point double between 0 and 1 </returns>
    public static double Sigmoid(double f)
    {
      return 1/(1 + Math.Pow(Math.E,-f));
    }
		///<summary> TanSig activation function</summary> 
		///<param name = "f" > a double of any value </param>
		///<returns> a floating point double between -1 and 1 </returns>
    public static double Tansig(double f)
    {
      return (2/(1 + Math.Pow(Math.E,-2*f))) -1;
    }
  }
}
