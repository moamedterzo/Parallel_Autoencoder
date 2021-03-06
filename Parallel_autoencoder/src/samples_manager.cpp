/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "samples_manager.h"
 
#include <string>
#include <dirent.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <ostream>
#include <sys/stat.h>
#include <cassert>
#include "opencv2/opencv.hpp"


using std::string;
using namespace cv;


namespace parallel_autoencoder
{
    samples_manager::samples_manager(){};

    samples_manager::samples_manager(string _path_folder, int _max_n_samples)
    {           
        path_folder = _path_folder;
        max_n_samples = _max_n_samples;

        init();
    };


    void samples_manager::init(){
       dp = opendir(path_folder.c_str());
       if (!dp) {
           std::cout << "Folder not found: " << path_folder;
       }

       current_sample_number = 0;
       height = -1;
    }

    void samples_manager::restart(){
        close();
        init();
    }

    uint samples_manager::get_number_samples()
    {
    	int number_of_samples = 0;

    	struct dirent *entry;
    	while((entry = get_next_dir(default_extension.c_str())) != nullptr)
    		number_of_samples++;

    	restart();

    	//il numero di esempi è dato dal limite di esempi da restituire o da quelli effettivamente presenti
    	return min(number_of_samples, max_n_samples);
    }

    dirent* samples_manager::get_next_dir(const char* extension)
    {
    	struct dirent *entry;
		do
		{
			if(!(entry = readdir(dp))) return nullptr;
		}
		while(strcmp(entry->d_name, ".") == 0
			|| strcmp(entry->d_name, "..") == 0
			|| !strstr(entry->d_name, extension));

		return entry;
    }


    bool samples_manager::get_next_sample(my_vector<float>& buffer, const char* extension, string *filename){

        //limite degli esempi restituiti
        if(max_n_samples != -1 && current_sample_number >= max_n_samples) return false;

        //vengono scartate le cartelle (si assume che i file abbiano tutti estensione .jpg)
        struct dirent *entry = get_next_dir(extension);
        if(entry == nullptr) return false;


        //se richiesto si passa il filename (senza estensione)
        if(filename)
        {
        	string file_name_with_extension(entry->d_name);

        	size_t lastindex = file_name_with_extension.find_last_of(".");
        	(*filename).assign(file_name_with_extension.substr(0, lastindex));
        }


        //nome percorso completo
        string name_fullfile(path_folder + "/" + string(entry->d_name));

        if(strcmp(extension, default_extension.c_str()) == 0)
        {
			//si leggono i vari pixel come scala di grigi
			Mat img = imread(name_fullfile, IMREAD_GRAYSCALE);
			assert((uint)img.rows * img.cols == buffer.size());

			for(int j=0;j<img.rows;j++)
			  for (int i=0;i<img.cols;i++)
			  {
				  const int index = j * img.cols + i;

				  //preprocessing effettuato normalizzando i risultati tra 0 e 1
				  buffer[index] = float(int(img.at<uchar>(j,i))) / INPUT_MAX_VALUE;
			  }
        }
        else
        {
        	//lettura file testuale
        	std::ifstream myFile(name_fullfile);
        	if(!myFile.is_open()) std::cout << "Could not open file: " + name_fullfile << "\n";

        	//get line
			std::string line;
        	std::getline(myFile, line);

        	//process line
        	std::stringstream ss(line);
        	ss.ignore(100, ',');
			for(uint i = 0; i != buffer.size(); i++) {
				if(ss.peek() == ',') ss.ignore();
				ss >> buffer[i];
			}
        }

        current_sample_number++;
        return true;       
    };    

    bool samples_manager::get_next_sample(my_vector<float>& buffer, const char* extension){
        return get_next_sample(buffer, extension, nullptr);
    };



    void samples_manager::save_sample(my_vector<float>& buffer, bool save_as_image, string folder, string filepath){

        //creazione cartella se non esiste
        struct stat _buffer_stat;
        if(stat (folder.c_str(), &_buffer_stat) != 0)
        {
            std::cout << "Creating folder '" << folder << "'\n";
            mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        }

        string complete_path = folder + "/" + filepath;

        if(save_as_image)
        {
            //si determina la grandezza dell'immagine al primo salvataggio
            if(height == -1)
            {
                float lato  = sqrt(buffer.size());
                if(lato == floor(lato)) //se è un quadrato perfetto
                {
                    height = width = lato;
                }
                else{
                    //solitamente si avranno hidden layer formati da rettangoli dove un lato è il doppio dell'altro
                    lato  = sqrt(buffer.size() / 2);
                    if(lato == floor(lato)) //se la metà del rettangolo è un quadrato perfetto
                    {
                        height = lato;
                        width = lato * 2;
                    }
                    else
                    {
                        //in caso estremo si inseriscono tutti i pixel su una sola linea
                        height = 1;
                        width = buffer.size();

                        std::cout << "THE SAVED IMAGE IS NOT A SQUARE NEITHER A RECTANGLE\n";
                    }
                }
            }


        	 //si normalizza l'immagine ai valori originali
			auto buffer_for_image = buffer;
			for(uint i = 0; i != buffer.size(); i++)
				buffer_for_image[i] *= INPUT_MAX_VALUE;

			Mat imageToSave = Mat(height, width, CV_32FC1, buffer_for_image.data());
			imwrite(folder + "/" + filepath, imageToSave);
        }
        else
        {
        	// salva l'esempio in forma testuale

			// Create an input filestream
			std::ofstream myFile(complete_path);
			if(!myFile.is_open()) std::cout << "Could not open file: " << complete_path << "\n";


			myFile << "_visible_" << buffer.size() << "__,";
			for(uint i = 0; i < buffer.size(); i++)
				myFile << std::fixed << std::setprecision(F_PREC) << buffer[i] << ",";

        }

    }

    void samples_manager::show_sample(my_vector<float>& buffer){

        save_sample(buffer, true, "./temp", "image_temp" + default_extension);

        string name_fullfile = string("./temp/image_temp" + default_extension);
        Mat imageToShow = imread(name_fullfile, IMREAD_GRAYSCALE);

        namedWindow("image");
        imshow("image", imageToShow);
        waitKey(0);
    }

    void samples_manager::close(){    
        if(dp)
          closedir(dp);
    }
}

