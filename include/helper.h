#define max( x, y ) ( ( x ) > ( y ) ? x : y )
#define dabs( x ) ( (x) < 0 ? -(x) : x )

// check if file exists (path is relative to executable)
inline bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

// input enum
enum {
  P_START = 0,
  P_END = 1,
  P_INC = 2,
  NREPEATS = 3,
  FORCE_CREATE = 4
};