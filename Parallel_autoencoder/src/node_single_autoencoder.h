
#ifndef NODE_SINGLE_AUTOENCODER_H_
#define NODE_SINGLE_AUTOENCODER_H_

#include "node_autoencoder.h"


namespace parallel_autoencoder{

	class node_single_autoencoder : public node_autoencoder{

	private:

		//gestore degli esempi input/output su disco
		samples_manager smp_manager;

		//percorso della cartella che contiene le immagini iniziali
		string image_path_folder;

		my_vector<matrix<float>> layers_weights;
		my_vector<my_vector<float>> layer_biases;


	public:

		~node_single_autoencoder(){}

		node_single_autoencoder(const my_vector<int>& _layers_size, std::default_random_engine& _generator,
					uint rbm_n_epochs, uint finetuning_n_epochs, bool batch_mode, bool _reduce_io,
					std::ostream& _oslog,
					samples_manager& _smp_manager);


		CommandType wait_for_command();

		void train_rbm();

		void save_new_samples(const uint layer_number,const uint n_visible_units,const uint n_hidden_units,
				const char *sample_extension,
				my_vector<float>& hidden_biases, matrix<float>& weights);



		void rollup_for_weights();

		void forward_pass(my_vector<my_vector<float>>& activation_layers);


		void backward_pass(my_vector<my_vector<float>>& activation_layers);

		my_vector<my_vector<float>> get_activation_layers();



		void fine_tuning();




		void reconstruct();

		string get_path_file();

		void save_parameters();

		void load_parameters();
	};
}


#endif /* NODE_SINGLE_AUTOENCODER_H_ */
