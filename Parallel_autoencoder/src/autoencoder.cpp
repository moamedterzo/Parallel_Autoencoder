/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "autoencoder.h"
 
#include <fstream>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>


#include "rbm.h"
#include "custom_utils.h"

using namespace std;


namespace parallel_autoencoder{

    Autoencoder::Autoencoder(const my_vector<uint>& _layers_size, samples_manager& _samplesmanager, std::default_random_engine& _generator)
    {
        samplesmanager = _samplesmanager;
        generator = _generator;

        fine_tuning_n_training_epocs = 5;
        fine_tuning_learning_rate = 10e-6;

        fine_tuning_finished = false;
        trained_rbms = 0;

        number_of_rbm_to_learn = _layers_size.size() - 1;
        number_of_final_layers = _layers_size.size() * 2 - 1;

        //Per N layer bisogna apprendere N-1 matrici di pesi e N-1 vettori di bias
        layers_weights = my_vector<matrix<float>>(number_of_final_layers - 1); //ci sono (N-1)*2 pesi per N layer
        layer_biases = my_vector<my_vector<float>>(number_of_final_layers - 1); //tutti i layer hanno il bias tranne quello di input

        set_size_for_layers(_layers_size);

    }

    void Autoencoder::set_size_for_layers(const my_vector<uint>& _layers_size_source)
    {
    	//_layers_size contiene la grandezza dei layer fino a quello centrale
		//in fase di rollup verranno creati altri layer per la ricostruzione
		layers_size = my_vector<uint>(number_of_final_layers);
		for(uint i = 0; i  !=_layers_size_source.size(); i++)
		{
			layers_size[i] = _layers_size_source[i];

			 //si copia la grandezza del layer per il layer da ricostruire
			uint rec_layer = number_of_final_layers - i - 1;
			layers_size[rec_layer] = layers_size[i];
		}
    }


    void Autoencoder::Train()
    {
        //1. Si apprendono le RBM per ciascun layer
        std::cout << "Imparando le RBM...\n";
        std::cout << "Numero di RBM da apprendere: " <<  number_of_rbm_to_learn <<"\n";
        std::cout << "Numero di RBM gia apprese: " << trained_rbms << "\n";
        std::cout << "Numero di layer finali: " <<  number_of_final_layers <<"\n";

        //percorso della cartella che contiene le immagini iniziali
        string image_path_folder = string(samplesmanager.path_folder);

        //Per ciascun layer...    
        //se sono stati già apprese delle rbm, si passa direttamente alla prima da imparare
        for(uint layer_number = trained_rbms; layer_number != number_of_rbm_to_learn; layer_number++)
        {
            uint n_visible_units = layers_size[layer_number];
            uint n_hidden_units = layers_size[layer_number + 1];
            const uint index_reverse_layer = number_of_final_layers - layer_number - 2;
			const char *sample_extension = layer_number == 0 ? ".jpg" : ".txt";

            std::cout << "-- Imparando il layer numero: " << layer_number
                    << ", hidden units: " << n_hidden_units 
                    << ", visible units: " << n_visible_units << " --\n";        


            auto& weights = layers_weights[layer_number];
            auto& hidden_biases = layer_biases[layer_number];
            auto& visible_biases =  layer_biases[index_reverse_layer];

            //si apprendono i pesi tramite una restricted boltzmann machine
            //i bias del layer visibile vanno a finire già nel futuro feed forward layer  
            rbm rmb_to_train(layer_number == 0, n_visible_units, n_hidden_units, samplesmanager, generator);
            rmb_to_train.learn(weights, hidden_biases, visible_biases);

            std::cout<< "Fine apprendimento RBM\n";

            //si deve salvare sul disco i risultati di attivazione del layer successivo
            //essi saranno utilizzati come input per la prossima fare di training        
            string new_image_path_folder = string(image_path_folder + "/" + std::to_string(layer_number));

            std::cout << "Salvando i risultati intermedi per il prossimo step nella cartella '" 
                    << new_image_path_folder << "'\n";

            //variabili temporanee
            string sample_filename;
            my_vector<float> input_samples(n_visible_units);
            my_vector<float> output_samples(n_hidden_units);

            samplesmanager.restart();
            while(samplesmanager.get_next_sample(input_samples, sample_extension, &sample_filename)){

                //si ottengono i valori di attivazione dalla RBM
                rmb_to_train.forward_pass(input_samples, output_samples, weights, hidden_biases);

                //si salva su file
                samplesmanager.save_sample(output_samples, false, new_image_path_folder, sample_filename + ".txt"); //dati in formato testuale
                samplesmanager.save_sample(output_samples, true, new_image_path_folder, sample_filename + ".jpg" ); //dati in formato immagine
            }

            //in maniera del tutto trasparente si utilizzerà questo nuovo percorso per ottenere i dati in input
            samplesmanager.path_folder = new_image_path_folder;        
            samplesmanager.restart();

            //contatore che memorizza il numero di rbm apprese
            trained_rbms++;        
            save_parameters();
        }       


        if(fine_tuning_finished)
        {
            std::cout << "Fine tuning già effettuato\n";
        }
        else
        {
            //Roll-up
            //una volta appresi i pesi, bisogna creare una rete di tipo feed forward
            //la rete feed forward dell'autoencoder possiede il doppio dei layer hidden, 
            //ad eccezione del layer più piccolo che di fatto serve a memorizzare l'informazione in maniera più corta
            for(uint trained_layer = number_of_rbm_to_learn - 1; trained_layer >= 0;  trained_layer--)
            {
                const uint new_layer = number_of_final_layers - trained_layer - 1;
                std::cout << "New layer (number " << new_layer << ") with " << layers_size[new_layer] << " nodes\n";

                //memorizzo i pesi trasposti nel layer feed forward, attenzione agli indici
                //si salva la trasposta dei pesi
                auto& layer_weights_source = layers_weights[trained_layer];
                auto& layer_weights_dest = layers_weights[new_layer - 1];

                layer_weights_dest = matrix<float>(layer_weights_source.get_cols(), layer_weights_source.get_rows());

                transpose_matrix(layer_weights_source, layer_weights_dest);
            }


            //Fine tuning
            std::cout << "\n\nFINE TUNING\n";

            //si riserva lo spazio necessario per l'attivazione di ogni layer 
            //e per i vettori che conterranno i valori delta per la back propagation
            my_vector<my_vector<float>> activation_layers(number_of_final_layers);
            for(uint l = 0; l != activation_layers.size(); l++)
                activation_layers[l] = my_vector<float>(layers_size[l]); //la grandezza è memorizzata nel vettore layers_size

            //si esclude il primo layer dato che non possiede pesi da aggiornare
            my_vector<my_vector<float>> delta_layers(number_of_final_layers - 1);
            for(uint l = 0; l != delta_layers.size(); l++)
                delta_layers[l] = my_vector<float>(layers_size[l + 1]); //la grandezza è memorizzata nel vettore layers_size

            const uint central_layer = number_of_final_layers / 2 - 1;

            //si passa alle immagini iniziali
            samplesmanager.path_folder = image_path_folder;    

            //per ogni epoca...
            for(uint epoch = 0; epoch != fine_tuning_n_training_epocs; epoch++)
            {
                samplesmanager.restart();      
                std::cout << "Training epoch: " << epoch << "\n";    

                //per ciascun esempio...
                while(samplesmanager.get_next_sample(activation_layers[0], ".jpg")){


                    //1. forward pass
                    for(uint l = 1; l != number_of_final_layers; l++){

                        auto& weights = layers_weights[l - 1];
                        auto& biases = layer_biases[l - 1];
                        auto& input = activation_layers[l - 1];
                        auto& activation_layer = activation_layers[l];

                        //l'attivazione del layer successivo è data dai pesi e dal bias
                        matrix_transpose_vector_multiplication(weights, input, biases, activation_layer);

                        //si applica la funzione sigmoide                
                        //se il layer è quello centrale (coding), bisogna effettuare un rounding dei valori
                        //per ottenere un valore binario
                        if(l == central_layer)
                        	for(uint i = 0;i != activation_layer.size(); i++)
                        		activation_layer[i] = round(sigmoid(activation_layer[i]));
                        else
                        	for(uint i = 0;i != activation_layer.size(); i++)
                        	     activation_layer[i] = sigmoid(activation_layer[i]);
                    }


                    //2. backward pass
                    //si va dall'ultimo layer al penultimo (quello di input non viene considerato)
                    my_vector<float> output_deltas;
                    my_vector<float> current_deltas;

                    for(uint l = number_of_final_layers - 1; l != 0; l--){

                        //si aggiornano i pesi tra l'output e l'input layer
                        auto& weights_to_update = layers_weights[l - 1];
                        auto& biases_to_update = layer_biases[l - 1];

                        auto& output_layer = activation_layers[l];
                        auto& input_layer = activation_layers[l - 1];

                        //check
                        assert(output_layer.size() == biases_to_update.size());
                        assert(output_layer.size() == weights_to_update.get_cols());
                        assert(input_layer.size() == weights_to_update.size());

                        //si calcoleranno i delta per il layer corrente
                        if(l == number_of_final_layers - 1)
                        {                
                            //layer di output
                            current_deltas = my_vector<float>(output_layer.size(), 0.0);

                            auto& first_activation_layer = activation_layers[0];

                            //calcolo dei delta per il layer di output
                            // delta = y_i * (1 - y_i) * reconstruction_error
                            for(uint j = 0; j != output_layer.size(); j++)
                            {
                               current_deltas[j] = output_layer[j] * (1 - output_layer[j])
                                       * (first_activation_layer[j] - output_layer[j]);
                            }
                        }
                        else
                        {
                            //layer nascosto

                            //si memorizzano i delta del passo precedente
                            output_deltas = my_vector<float>(current_deltas);
                            current_deltas = my_vector<float>(output_layer.size(), 0.0);

                            //si vanno a prendere i pesi tra il layer di output e quello a lui successivo
                            auto& weights_for_deltas = layers_weights[l];

                            //il delta per il nodo j-esimo è dato dalla somma pesata dei delta dei nodi del layer successivo
                            for(uint j = 0; j != current_deltas.size(); j++)
                                for(uint i = 0; i != output_deltas.size(); i++)
                                    current_deltas[j] +=  output_deltas[i] * weights_for_deltas.at(j, i);
                        }

                        //applico gradiente per la matrice dei pesi
                        for(uint i = 0; i != weights_to_update.get_rows(); i++){
                            for(uint j = 0; j != weights_to_update.get_cols(); j++){
                                //delta rule
                                weights_to_update.at(i, j) +=  fine_tuning_learning_rate * current_deltas[j] * input_layer[i];
                            }
                        }    

                        //seguendo la delta rule, si applica il gradiente anche i bias
                        for(uint j = 0; j != biases_to_update.size(); j++)
                            biases_to_update[j] += fine_tuning_learning_rate * current_deltas[j];
                    }
                }            
            }

            //allenamento concluso
            fine_tuning_finished = true;
            save_parameters();
        }
    }



    my_vector<float> Autoencoder::reconstruct(my_vector<float> input)
    {
        const uint central_layer = number_of_final_layers / 2 - 1;

        my_vector<float> output;

        //1. forward pass
        for(uint l = 1; l != number_of_final_layers; l++){

            auto& weights = layers_weights[l - 1];
            auto& biases = layer_biases[l - 1];
            output = my_vector<float>(layers_size[l], 0.0);

            //l'attivazione del layer successivo è data dai pesi e dal bias
            matrix_transpose_vector_multiplication(weights, input, biases, output);

            //si applica la funzione sigmoide                
            //se il layer è quello centrale (coding), bisogna effettuare un rounding dei valori
            //per ottenere un valore binario
            if(l == central_layer)
				for(uint i = 0;i != output.size(); i++)
					output[i] = round(sigmoid(output[i]));
			else
				for(uint i = 0;i != output.size(); i++)
					output[i] = sigmoid(output[i]);

            input = my_vector<float>(output);
        }

        return output;
    }



    void Autoencoder::save_parameters()
    {
        save_parameters(parameters_tosave_file_path);
    }

    void Autoencoder::save_parameters(string& path_file)
    {  
        std::cout << "Saving autoencoder parameters to '" + path_file + "'\n";

        // Create an input filestream
        std::ofstream myFile(path_file);

        // Make sure the file is open
        if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);

        //todo capire che numero mettere
        const int F_PREC = 9;

        //salvataggio di pesi, bias
        uint layer_number;
        for(layer_number = 0; layer_number != trained_rbms; layer_number++)
        {
            const uint index_reverse_layer = number_of_final_layers - layer_number - 2;

            auto& weights = layers_weights[layer_number];
            auto& hidden_biases = layer_biases[layer_number];
            auto& visible_biases =  layer_biases[index_reverse_layer];


            myFile << "_rbm_" << weights.get_rows() << "x" << weights.get_cols() << "__,";
            for(uint i = 0; i < weights.size(); i++)
                  myFile << fixed << setprecision(F_PREC) << weights[i] << ",";
            myFile << endl;

            myFile << "_hidden_" << hidden_biases.size() << "__,";        
            for(uint i = 0; i < hidden_biases.size(); i++)
                myFile << fixed << setprecision(F_PREC) <<hidden_biases[i] << ",";
            myFile << endl;

            myFile << "_visible_" << visible_biases.size() << "__,";        
            for(uint i = 0; i < visible_biases.size(); i++)
                myFile << fixed << setprecision(F_PREC) <<visible_biases[i] << ",";
            myFile << endl;
        }

        if(fine_tuning_finished)
        {
            while(layer_number != number_of_final_layers - 1)
            {
                auto& weights = layers_weights[layer_number];
                auto& hidden_biases = layer_biases[layer_number];

                myFile << "_rec_" <<  weights.get_rows() << "x" << weights.get_cols()  << "__,";
                for(uint i = 0; i < weights.size(); i++)
                     myFile << fixed << setprecision(F_PREC) << weights[i] << ",";
                myFile << endl;

                myFile << "_rec_" << hidden_biases.size() << "__,";        
                for(uint i = 0; i < hidden_biases.size(); i++)
                      myFile << fixed << setprecision(F_PREC) <<hidden_biases[i] << ",";
                myFile << endl;

                layer_number++;
            }
        }

        myFile.close();
    }


    void Autoencoder::load_parameters()
    {
        load_parameters(parameters_tosave_file_path);
    }

    void Autoencoder::load_parameters(string& path_file)
    {    
        std::cout << "Getting autoencoder parameters from '" + path_file + "'\n";

        fine_tuning_finished = false;
        trained_rbms = 0;

        // Create an input filestream
        std::ifstream myFile(path_file);

        // Make sure the file is open
        if(!myFile.is_open()) throw std::runtime_error("Could not open file: " + path_file);

        // Helper vars
        std::string line;

        //variabili che fanno riferimento al layer nel quale si salveranno i parametri
        uint n_visible_units;
        uint n_hidden_units;

        matrix<float> *current_weights;
        my_vector<float> *current_hidden_biases;
        my_vector<float> *current_visible_biases;

        uint current_row_file = 0;

        //Si leggono le linee che contengono i pesi delle rbm apprese
        bool other_lines = false;
        while(std::getline(myFile, line))
        {
            if(current_row_file % 3 == 0)
            {
                //se abbiamo letto tutti i pesi delle rbm si esce da questo ciclo
                if(trained_rbms == number_of_rbm_to_learn){
                    //se ci sono altre linee, vuol dire che si possiedono i parametri dei layer di ricostruzione
                    other_lines = true;
                    break;
                }

                const uint index_reverse_layer = number_of_final_layers - trained_rbms - 2;

                n_visible_units = layers_size[trained_rbms];
                n_hidden_units = layers_size[trained_rbms + 1];

                layers_weights[trained_rbms] = matrix<float>(n_visible_units, n_hidden_units);
                layer_biases[trained_rbms] = my_vector<float>(n_hidden_units);
                layer_biases[index_reverse_layer] = my_vector<float>(n_visible_units);

                current_weights = &layers_weights[trained_rbms];
                current_hidden_biases = &layer_biases[trained_rbms];
                current_visible_biases = &layer_biases[index_reverse_layer];

                //questa variabile memorizza il numero di rbm apprese
                trained_rbms++;             
            }

            // Create a stringstream of the current line
            std::stringstream ss(line);

            //in base alla riga si aggiornano i relativi parametri
            if(current_row_file % 3 == 0){

                //riga dei pesi
                ss.ignore(100, ',');
            	for(uint i = 0; i != n_visible_units; i++)
					for(uint j = 0; j != n_hidden_units; j++){
					   if(ss.peek() == ',') ss.ignore();
					   ss >>  current_weights->at(i, j);
					}

            }
            else if (current_row_file % 3 == 1){

                //riga dei bias nascosti 
                ss.ignore(100, ',');
                for(uint j = 0; j != n_hidden_units; j++){
						if(ss.peek() == ',') ss.ignore();
						ss >> current_hidden_biases->operator [](j);
					}
            }
            else if (current_row_file % 3 == 2){

                //riga dei bias visibili  
                ss.ignore(100, ',');          
                for(uint i = 0; i != n_visible_units; i++) {
						if(ss.peek() == ',') ss.ignore();
						ss >> current_visible_biases->operator [](i);;
					}
            }            

            //si tiene conto della riga processata
            current_row_file++;                    
        }


        //se ci sono altre linee da analizzare vuol dire che si aggiornano i pesi dei layer di ricostruzione    
        if(other_lines)
        {
            current_row_file = 0;

            //indice del layer contenente pesi o bias
            uint current_layer = (number_of_final_layers - 1) / 2;
            do
            {
                if(current_row_file % 2 == 0)
                {
                    n_visible_units = layers_size[current_layer];
                    n_hidden_units = layers_size[current_layer + 1];

                    layers_weights[current_layer] = matrix<float>(n_visible_units, n_hidden_units);

                    //già inizializzati
                    current_weights = &layers_weights[current_layer];
                    current_hidden_biases = &layer_biases[current_layer];
                    //c'è un solo layer per i bias da aggiornare

                    current_layer++;

                    //check sulle misure
                    assert(n_hidden_units == current_hidden_biases->size());
                }

                // Create a stringstream of the current line
                std::stringstream ss(line);

                //in base alla riga si aggiornano i relativi parametri
                if(current_row_file % 2 == 0){

                    //riga dei pesi
                    ss.ignore(100, ',');
                    for(uint i = 0; i != n_visible_units; i++)
						for(uint j = 0; j != n_hidden_units; j++){
						   if(ss.peek() == ',') ss.ignore();
						   ss >>  current_weights->at(i, j);
						}
                }
                else if (current_row_file % 2 == 1){

                    //riga dei bias    
                    ss.ignore(100, ',');   
                    for(uint j = 0; j != n_hidden_units; j++){
							if(ss.peek() == ',') ss.ignore();
							ss >> current_hidden_biases->operator [](j);
						}
                }         

                //si tiene conto della riga processata
                current_row_file++;
            }
            while(std::getline(myFile, line));

            //il training si considera concluso
            fine_tuning_finished = true;
        }

        // Close file
        myFile.close();

    }

}
