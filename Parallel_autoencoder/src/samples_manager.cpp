/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "samples_manager.h"
 
#include <string>
#include <dirent.h>
#include <iostream>
#include <sys/stat.h>
#include <cassert>
#include "opencv2/opencv.hpp"


using std::string;
using std::vector;
using namespace cv;


namespace parallel_autoencoder{

	//valore utilizzato per normalizzare i valori in input
	static const float INPUT_MAX_VALUE = 255;


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

       //todo capire se bisogna effettuare preprocessing o meno
    }

    void samples_manager::restart(){
        close();
        init();
    }

    uint samples_manager::get_number_samples()
    {
    	int number_of_samples = 0;

    	struct dirent *entry;
    	while((entry = get_next_dir()) != nullptr)
    		number_of_samples++;

    	restart();

    	//il numero di esempi è dato dal limite di esempi da restituire o da quelli effettivamente presenti
    	return min(number_of_samples, max_n_samples);
    }

    dirent* samples_manager::get_next_dir()
    {
    	struct dirent *entry;
		do
		{
			if(!(entry = readdir(dp))) return nullptr;
		}
		while(strcmp(entry->d_name, ".") == 0
			|| strcmp(entry->d_name, "..") == 0
			|| !strstr(entry->d_name, ".jpg"));

		return entry;
    }


    bool samples_manager::get_next_sample(vector<float>& buffer, string *filename){

        //limite degli esempi restituiti
        if(max_n_samples != -1 && current_sample_number >= max_n_samples) return false;

        //vengono scartate le cartelle (si assume che i file abbiano tutti estensione .jpg)
        struct dirent *entry = get_next_dir();
        if(entry == nullptr) return false;

       // if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) 
        //    return get_next_sample(buffer, filename);

        //se richiesto si passa il filename
        if(filename)
            (*filename).assign(entry->d_name);

        //nome percorso completo
        string name_fullfile(path_folder + "/" + string(entry->d_name));
        //std::cout << name_fullfile << "\n";

        //si leggono i vari pixel come scala di grigi
        Mat img = imread(name_fullfile, IMREAD_GRAYSCALE);

        //std::cout << img.rows << "x" << img.cols << ".--" << buffer.size() << "\n";

        assert(img.rows * img.cols == buffer.size());

        for(int j=0;j<img.rows;j++) 
        {                
          for (int i=0;i<img.cols;i++)
          {                
              const int index = j * img.cols + i;

              buffer.at(index) = float(img.at<uchar>(j,i)) / INPUT_MAX_VALUE;
          }
        }    

        current_sample_number++;
        return true;       
    };    

    bool samples_manager::get_next_sample(vector<float>& buffer){        
        return get_next_sample(buffer, nullptr);
    };



    void samples_manager::save_sample(vector<float>& buffer, string folder, string filepath){

        //si determina la grandezza dell'immagine al primo salvataggio
        if(height == -1){

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

                    std::cout << "PROBLEMA SALVATAGGIO IMMAGINE, NON E' UN QUADRATO E NEANCHE UN RETTANGOLO\n";
                }
            }
        }

        //creazione cartella se non esiste
        struct stat _buffer_stat;
        if(stat (folder.c_str(), &_buffer_stat) != 0)
        {
            std::cout << "Creando la cartella '" << folder << "'\n";
            mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        }

        //si normalizza l'immagine ai valori originali (altrimenti le prossime letture non funzioneranno)
        auto buffer_for_image = buffer;
        for(auto& v : buffer_for_image)
        	v *= INPUT_MAX_VALUE;

        Mat imageToSave = Mat(height, width, CV_32FC1, buffer_for_image.data());
        imwrite(folder + "/" + filepath, imageToSave);

        //std::cout << "Immagine salvata in '" << folder + "/" + filepath << "'\n";
        //std::cout << height << "x" << width << "\n";
    }

    void samples_manager::show_sample(vector<float>& buffer){

        /*//si determina la grandezza dell'immagine
        float lato  = sqrt(buffer.size());
        int height_s, width_s;

        if(lato == floor(lato)) //se è un quadrato perfetto
        {
            height_s = width_s = lato;
        }
        else{
            //solitamente si avranno hidden layer formati da rettangoli dove un lato è il doppio dell'altro
            lato  = sqrt(buffer.size() / 2);
            if(lato == floor(lato)) //se la metà del rettangolo è un quadrato perfetto
            {
                height_s = lato;
                width_s = lato * 2;
            }
            else
            {
                //in caso estremo si inseriscono tutti i pixel su una sola linea
                height_s = 1;
                width_s = buffer.size();

                std::cout << "PROBLEMA IMMAGINE, NON E' UN QUADRATO E NEANCHE UN RETTANGOLO\n";
            }
        }*/

        save_sample(buffer, "./temp", "image_temp.jpg");        

        string name_fullfile = string("./temp/image_temp.jpg");
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

