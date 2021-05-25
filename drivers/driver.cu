#include "monolithic.h"

// split string be delimter
// taken from stackoverflow 
// https://stackoverflow.com/questions/236129/how-do-i-iterate-over-the-words-of-a-string?page=1&tab=votes#tab-top
template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

// print the outputs
void print_inputs(vector<int> inputs) {
    for (int val : inputs)
        cout << val << " ";
    cout << endl;
}

// parse input.general
void parse_input_gen(vector<int> *inputs) {
    fstream input_file;
    input_file.open("./kernel_inputs/input.general", ios::in);
    string line;

    while(getline(input_file, line)){

        std::vector<std::string> words = split(line, ' ');

        for (string word : words) {
            // ignore anything after a # (comment)
            if(word.find('#') != std::string::npos || word == "")
                break;
            else{
                (*inputs).push_back(atoi(&word[0]));
            }
        }
    }
    input_file.close();
}



int main(int argc, char *argv[]) 
{

    if (argc != 1){

        // parse_options(argc, argv);

    }



    vector<int> inputs;
    if ( file_exists("./kernel_inputs/input.general") )
    {
        parse_input_gen(&inputs);
    }
    // set default values
    else 
    {
        cout << "file does NOT exists" << endl;

        inputs[P_START]      = 32; 
        inputs[P_END]        = 1024;
        inputs[P_INC]        = 32;
        inputs[NREPEATS]     = 5;
        inputs[FORCE_CREATE] = false;
    }




    // init cuda for tensor cores
    int MP_count = init((bool)inputs[FORCE_CREATE]);
    
    test_kernel();

    // fp16_gemm_driver(
    //     inputs
    // );
}

