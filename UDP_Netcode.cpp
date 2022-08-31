#import <vector> 
using std::vector;

#import <map>
using std::map;

#import <algorithm>

// for debugging
#import <iostream> 
using std::cout;

// IPs are 4 bytes (4 0-255 values) and an int is 4 bytes so makes sense to convert them like this, 
// LAN connections do make this more complicated and suggest use of a struct so I included one for scalability.
struct IP {
  uint publicIP;
  char LANid; 
    
  IP(uint Public, char LANID = 0)
  {
    publicIP = Public; LANid = LANID;
  }
  IP(){}

};


struct packet {    //once created should be immutable
  private: 
    IP ip;  
    uint elo; 
  public:
    // elo => elo; 
    IP getIp(){return ip;}
    uint getElo(){return elo;}
    packet(uint publicIP, char Lanid, uint Elo){
      ip = IP(publicIP, Lanid); elo = Elo; 
    }
    packet(IP Ip, uint Elo){
      ip = Ip;
      elo = Elo;      
    }
    packet(){}
    bool const operator < (const packet p){return elo < p.elo; }
};

const int MY_ELO = 56678; //for referance and scalablity


// Ranges of ranks (bronze, silver, gold etc) 
// Placed in array as ranks don't need to be equally spaced
const int RANK_NUM = 4;
const int RANKS[RANK_NUM+1] = {0, 10, 20, 30, 40 }; // each item represents the lower bound of the rank
// packet my_packet = packet(0, MY_ELO); //client doesn't need referance to own IP
const uint ELO_LOWERBOUND = MY_ELO * 0.9;
const uint ELO_UPPERBOUND = MY_ELO * 1.1;

// vector<packet> database[RANK_NUM]; // one for each rank, each should be sorted
map<int,IP> database; 

// to be deprecated, look ups in map instead
vector<packet> possible_matches; // the players 

//to be depricated
bool database_insert(map<int,IP> & M, packet p)
{
  uint item = p.getElo();
  if (std::distance(M.find(item), M.end()) > 0) 
  {
    // key is not in map
    M[item] = p.getIp(); //add item to map
    return true;
  }
  
  return false;
}

//Run when packet comes in, if the value is within the bounds for a match, adds it to all possible matches, always adds it to database. 
void check_elo( packet query )
{
  int q_elo = query.getElo();
  if (q_elo > ELO_LOWERBOUND && q_elo < ELO_UPPERBOUND)
  {
    //check if in last match
    
    //Add to my possible matches
    possible_matches.emplace_back(query.getIp(),q_elo); 
  }
  database_insert(database, query);
}  


//Run as soon as a packet comes in, returns all the values in it's database that are a possible match for the incomming packet. 
map<int,IP> questioned(packet q, uint lower_bound, uint upper_bound)
{
  map<int,IP> result = {};
  auto LBit = database.lower_bound(lower_bound);
  auto UBit = database.upper_bound(lower_bound);
  //return all the usefull values from our database. 
  std::copy(LBit, UBit, std::inserter(result, result.begin()));
  database_insert(database, q);

  return result;
}

bool check_matchfound()
{
  bool b = false;
  //tell all of the clients we found a match and await response
  //set a flag so incomming UDP connections know we're in a match
  // awaiting_reponse ? continue : ( matchfound ? b= true, continue : return false)
  //write database to file so we can remove it from the heap
  
  // b ? return true : continue
  //check repsonse from match clients again
  
  //return b

  return true; 
}


int main(){
  cout<<"successfully compiled! 👍😁" << "\n";
}
  
