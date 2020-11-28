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
   
    rbm::rbm(bool _first_layer, uint _n_visible_units, uint _n_hidden_units,
            samples_manager& _samples_manager, std::default_random_engine& _generator)
		{

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
    inline void rbm::update_parameters(matrix<float> &weights,
    		            my_vector<float> &hidden_biases, my_vector<float> &visible_biases,
    					matrix<float> &diff_weights,
    					my_vector<float> &diff_visible_biases, my_vector<float> &diff_hidden_biases,
    		            const uint number_of_samples)
    {
            //si precalcola il fattore moltiplicativo
            //dovendo fare una media bisogna dividere per il numero di esempi
            const float mult_factor = learning_rate / number_of_samples;

            //diff per pesi e bias visibili
            for(uint i = 0; i != visible_biases.size(); i++)
            {
                visible_biases[i] += diff_visible_biases[i] * mult_factor;

                diff_visible_biases[i] = diff_visible_biases[i] * momentum; //inizializzazione per il momentum

                for(uint j = 0; j != hidden_biases.size(); j++){
                   weights.at(i, j) += diff_weights.at(i, j) * mult_factor;

                   diff_weights.at(i, j) = diff_weights.at(i, j) * momentum;//inizializzazione per il momentum
                }
            }

            for(uint j = 0; j != hidden_biases.size(); j++){
                hidden_biases[j] += diff_hidden_biases[j]* mult_factor;

                diff_hidden_biases[j] = diff_hidden_biases[j] * momentum;//inizializzazione per il momentum
            }               
    }

    
    void rbm::learn(matrix<float>& weights, my_vector<float>& hidden_biases, my_vector<float>& visible_biases)
    {
        //la matrice dei pesi per il layer in questione, 
        //possiede grandezza VxH (unità visibili per unità nascoste)
        //si riserva lo spazio necessario
        weights = matrix<float>(n_visible_units, n_hidden_units);

        //inizializzazione pesi
		for(uint i = 0; i < weights.size(); i++)
			weights[i] = sample_gaussian_distribution(initial_weights_mean, initial_weights_variance, generator);

        //inizializzazione bias
        visible_biases = my_vector<float>(n_visible_units, initial_biases_value);
        hidden_biases = my_vector<float>(n_hidden_units, initial_biases_value);

        //layers visibili e nascosti, ricostruiti e non
        my_vector<float> visible_units(n_visible_units);
        my_vector<float> hidden_units(n_hidden_units);
        my_vector<float> rec_visible_units(n_visible_units);
        my_vector<float> rec_hidden_units(n_hidden_units);

        //gradienti calcolati per pesi e bias
        matrix<float> diff_weights(n_visible_units, n_hidden_units, 0.0);
        my_vector<float> diff_visible_biases(n_visible_units, 0.0);
        my_vector<float> diff_hidden_biases(n_hidden_units, 0.0);

		const char *sample_extension = first_layer ? ".jpg" : ".txt";

        //si avvia il processo di apprendimento per diverse epoche
        ulong current_index_sample = 0;
        for(uint epoch = 0; epoch != n_training_epocs; epoch++){

            if(epoch % 5 == 0)
                std::cout << "Training epoch: " << epoch << "\n";

            //learning rate costante

            //per ciascun esempio...
            while(samplesmanager.get_next_sample(visible_units, sample_extension))
            {
                current_index_sample++;

                if(current_index_sample % 100 == 0)
                    std::cout << "current_index_sample: " << current_index_sample << "\n";

                //CONTRASTIVE DIVERGENCE

                //1. Effettuare sampling dell'hidden layer
                matrix_transpose_vector_multiplication(weights, visible_units, hidden_biases, hidden_units);
                for(uint i = 0; i != hidden_units.size(); i++)
                    hidden_units[i] = sample_sigmoid_function(hidden_units[i], generator);

                //2. Ricostruire layer visibile
                //non si applica il campionamento
                matrix_vector_multiplication(weights, hidden_units, visible_biases, rec_visible_units);
                if(first_layer) //per il primo layer bisogna aggiungere del rumore gaussiano
                    for(uint i = 0; i != rec_visible_units.size(); i++)
                        rec_visible_units[i] = sample_gaussian_distribution(rec_visible_units[i], generator);
                else
					for(uint i = 0; i != rec_visible_units.size(); i++)
						rec_visible_units[i] = sigmoid(rec_visible_units[i]);


                //3. si ottiene il vettore hidden partendo dalle unità visibili ricostruite
                //non si applica il campionamento
                matrix_transpose_vector_multiplication(weights, rec_visible_units, hidden_biases, rec_hidden_units);
                for(uint i = 0; i != rec_hidden_units.size(); i++)
                	rec_hidden_units[i] = sigmoid(rec_hidden_units[i]);

                //4. si calcolano i differenziali
                //dei pesi e bias visibili
                for(uint i = 0; i != visible_units.size(); i++)
                {
                    diff_visible_biases[i] = diff_visible_biases[i]+ visible_units[i] - rec_visible_units[i];

                    for(uint j = 0; j != hidden_units.size(); j++){
                        diff_weights.at(i, j) +=  visible_units[i] * hidden_units[j]  //fattore positivo
												  - rec_visible_units[i] * rec_hidden_units[j]; //fattore negativo
                    }
                }

                //dei bias nascosti                
                for(uint j = 0; j != hidden_units.size(); j++)
                    diff_hidden_biases[j] += + hidden_units[j] - rec_hidden_units[j];

                //se abbiamo raggiunto la grandezza del mini batch, si modificano i pesi
                if(current_index_sample % size_minibatch == 0)
                    update_parameters(weights, hidden_biases, visible_biases,
                            diff_weights, diff_visible_biases, diff_hidden_biases, size_minibatch);
            }

            //si riavvia l'ottenimento dei samples
            samplesmanager.restart();
        }

        //se si sono degli esempi non ancora considerati, si applica il relativo update dei pesi
        int n_last_samples = current_index_sample % size_minibatch;
        if(n_last_samples != 0)
            update_parameters(weights, hidden_biases, visible_biases,
                        diff_weights, diff_visible_biases, diff_hidden_biases, n_last_samples);

    };

    void rbm::forward_pass(const my_vector<float>& input, my_vector<float>& output,
        const matrix<float>& weights, const my_vector<float>& hidden_biases){

        matrix_transpose_vector_multiplication(weights, input, hidden_biases, output);
        for(uint i = 0; i != output.size(); i++)
        	output[i] = sigmoid(output[i]);
    }
    

}
