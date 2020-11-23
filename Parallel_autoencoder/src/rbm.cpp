/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "rbm.h"

 
#include <iostream>
#include <array>

#include "samples_manager.h"
#include "custom_utils.h"


namespace parallel_autoencoder{
   
    rbm::rbm(bool _first_layer, int _n_visible_units, int _n_hidden_units,
        samples_manager& _samples_manager, std::default_random_engine& _generator){

        first_layer = _first_layer;
        n_visible_units = _n_visible_units;
        n_hidden_units = _n_hidden_units;        

        samplesmanager = _samples_manager;
        generator = _generator;        

        learning_rate = first_layer ? 0.001 : 0.01;
        momentum = 0.9;
        n_training_epocs = 20;//todo sistemare
        size_minibatch = 1;//todo sistemare

        initial_weights_variance = 0.01;
        initial_weights_mean = 0;
        initial_biases_value = 0;
    };

    //dopo aver utilizzato i differenziali, li si inizializzano considerando il momentum
    //la formula per l'update di un generico parametro è: Δw(t) = momentum * Δw(t-1) + learning_parameter * media_gradienti_minibatch
    inline void rbm::update_parameters(
        vector<vector<float>> &weights, 
        vector<float> &hidden_biases, 
        vector<float> &visible_biases,
        vector<vector<float>> &diff_weights,
        vector<float> &diff_visible_biases,
        vector<float> &diff_hidden_biases,
        const int number_of_samples
        ){    
            //std::cout << "Aggiornando i parametri...\n";

            //si precalcola il fattore moltiplicativo
            //dovendo fare una media bisogna dividere per il numero di esempi
            const int mult_factor = learning_rate / number_of_samples;

            //diff per pesi e bias visibili
            for(int i = 0; i < visible_biases.size(); i++)
            {
                visible_biases.at(i) += diff_visible_biases.at(i) * mult_factor;                

                diff_visible_biases.at(i) = diff_visible_biases.at(i) * momentum / mult_factor; //inizializzazione per il momentum

                for(int j = 0; j < hidden_biases.size(); j++){
                   weights.at(i).at(j) += diff_weights.at(i).at(j) * mult_factor;

                   diff_weights.at(i).at(j) = diff_weights.at(i).at(j) * momentum / mult_factor;//inizializzazione per il momentum
                }
            }

            for(int j = 0; j < hidden_biases.size(); j++){
                hidden_biases.at(j) += diff_hidden_biases.at(j) * mult_factor;

                diff_hidden_biases.at(j) = diff_hidden_biases.at(j) * momentum / mult_factor;//inizializzazione per il momentum
            }               
    }

    
    void rbm::learn(vector<vector<float>>& weights, vector<float>& hidden_biases,
            vector<float>& visible_biases){

        //la matrice dei pesi per il layer in questione, 
        //possiede grandezza VxH (unità visibili per unità nascoste)
        //si riserva lo spazio necessario
        weights = vector<vector<float>>(n_visible_units, vector<float>(n_hidden_units));

        //inizializzazione pesi
        for(auto& vec : weights)
            for(auto& v : vec)
                v = parallel_autoencoder::sample_gaussian_distribution(initial_weights_mean, initial_weights_variance, generator);

        //inizializzazione bias
        visible_biases = vector<float>(n_visible_units, initial_biases_value);
        hidden_biases = vector<float>(n_hidden_units, initial_biases_value);

        //layers visibili e nascosti, ricostruiti e non
        vector<float> visible_units(n_visible_units);
        vector<float> hidden_units(n_hidden_units);
        vector<float> rec_visible_units(n_visible_units); 
        vector<float> rec_hidden_units(n_hidden_units);

        //gradienti calcolati per pesi e bias
        vector<vector<float>> diff_weights(n_visible_units, vector<float>(n_hidden_units, 0.0));
        vector<float> diff_visible_biases(n_visible_units, 0.0);
        vector<float> diff_hidden_biases(n_hidden_units, 0.0);


        //si avvia il processo di apprendimento per diverse epoche
        long current_index_sample = 0;
        for(int epoch = 0; epoch < n_training_epocs; epoch++){

            if(epoch % 1 == 0) 
                std::cout << "Training epoch: " << epoch << "\n";                

            //todo implementare per bene modifiche al learning rate in base all'epoca
            if(n_training_epocs - epoch == int((n_training_epocs / 10)))
                learning_rate /= 4;

            //per ciascun esempio...
            while(samplesmanager.get_next_sample(visible_units)){

                current_index_sample++;

                if(current_index_sample % 100 == 0)
                    std::cout << "current_index_sample: " << current_index_sample << "\n";

                //CONTRASTIVE DIVERGENCE

                //1. Effettuare sampling dell'hidden layer
                matrix_transpose_vector_multiplication(weights, visible_units, hidden_biases, hidden_units);
                for(int i = 0; i < hidden_units.size(); i++)
                    hidden_units.at(i) = sample_sigmoid_function(hidden_units.at(i), generator);

                //2. Ricostruire layer visibile
                //non si applica il campionamento
                matrix_vector_multiplication(weights, hidden_units, visible_biases, rec_visible_units);
                if(first_layer) //per il primo layer bisogna aggiungere del rumore gaussiano
                    for(int i = 0; i < rec_visible_units.size(); i++)
                        rec_visible_units.at(i) = sample_gaussian_distribution(rec_visible_units.at(i), generator);
                else
					for(int i = 0; i < rec_visible_units.size(); i++)
						rec_visible_units.at(i) = sigmoid(rec_visible_units.at(i));


                //3. si ottiene il vettore hidden partendo dalle unità visibili ricostruite
                //non si applica il campionamento
                matrix_transpose_vector_multiplication(weights, rec_visible_units, hidden_biases, rec_hidden_units);
                for(int i = 0; i < rec_hidden_units.size(); i++)
                	rec_hidden_units.at(i) = sigmoid(rec_hidden_units.at(i));

                //4. si calcolano i differenziali
                //dei pesi e bias visibili
                for(int i = 0; i < visible_units.size(); i++)
                {
                    diff_visible_biases.at(i) = diff_visible_biases.at(i) + visible_units.at(i) - rec_visible_units.at(i);

                    for(int j = 0; j < hidden_units.size(); j++){                        
                        diff_weights.at(i).at(j) = diff_weights.at(i).at(j) 
                                 + visible_units.at(i) * hidden_units.at(j)  //fattore positivo
                                 - rec_visible_units.at(i) * rec_hidden_units.at(j); //fattore negativo
                    }
                }

                //dei bias nascosti                
                for(int j = 0; j < hidden_units.size(); j++)
                    diff_hidden_biases.at(j) = diff_hidden_biases.at(j) + hidden_units.at(j) - rec_hidden_units.at(j);

                //se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
                if(current_index_sample % size_minibatch == 0){

                    update_parameters(weights, hidden_biases, visible_biases,
                            diff_weights, diff_visible_biases, diff_hidden_biases, size_minibatch);        

                }
            }

            //si riavvia l'ottenimento dei samples
            samplesmanager.restart();
        }

        //se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
        int n_last_samples = current_index_sample % size_minibatch;
        if(n_last_samples != 0){

            //modifica pesi
            update_parameters(weights, hidden_biases, visible_biases,
                        diff_weights, diff_visible_biases, diff_hidden_biases, n_last_samples);
        }

    };

    void rbm::forward_pass(const vector<float>& input, vector<float>& output,
        const vector<vector<float>>& weights, const vector<float>& hidden_biases){

        matrix_transpose_vector_multiplication(weights, input, hidden_biases, output);
        for(int i = 0; i < output.size(); i++)
        	output.at(i) = sigmoid(output.at(i));
    }
    

}
