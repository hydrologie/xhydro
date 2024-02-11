"""Test suite for LSTM model implementations"""

import os

from xhydro.lstm_tools.lstm_controller import control_regional_lstm_training, control_local_lstm_training


class TestLstmModels:
    """Test suite for the LSTM models."""
    batch_size = 64  # batch size used in the training - multiple of 32
    epochs = 2  # Number of epoch to train the LSTM model
    window_size = 5  # Number of time step (days) to use in the LSTM model
    train_pct = 10  # Percentage of watersheds used for the training
    valid_pct = 5  # Percentage of watersheds used for the validation
    use_cpu = True  # Use CPU as GPU is not guaranteed to be installed with CUDA/CuDNN etc.
    use_parallel = False
    do_simulation = True
    input_data_filename = "LSTM_test_data.nc"
    filename_base = "LSTM_results"
    simulation_phases = ["test"]
    # Tags for the dynamic variables in the netcdf files.
    dynamic_var_tags = ["tasmax_MELCC", "rf", "Qsim"]
    # Scale variable according to area. Used for simulated flow inputs.
    qsim_pos = [False, False, True]
    # static variables used to condition flows on catchment properties
    static_var_tags = ["drainage_area", "elevation", "coniferous_forest", "silty_clay_loam", "meanPrecip"]

    def test_lstm_controller_regional(self):
        """Test the regional LSTM model implementation."""
        training_func = "nse_scaled"
        do_train = True

        kge_results, flow_results, name_of_saved_model = control_regional_lstm_training(
            self.input_data_filename,
            self.dynamic_var_tags,
            self.qsim_pos,
            self.static_var_tags,
            batch_size=self.batch_size,
            epochs=self.epochs,
            window_size=self.window_size,
            train_pct=self.train_pct,
            valid_pct=self.valid_pct,
            use_cpu=self.use_cpu,
            use_parallel=self.use_parallel,
            do_train=do_train,
            do_simulation=self.do_simulation,
            training_func=training_func,
            filename_base=self.filename_base,
            simulation_phases=self.simulation_phases,

        )

        assert len(kge_results[0]) == 4
        assert len(flow_results[0]) == 4
        assert os.path.isfile(name_of_saved_model)

        # Do a sim with no training
        do_train = False

        kge_results, flow_results, name_of_saved_model = control_regional_lstm_training(
            self.input_data_filename,
            self.dynamic_var_tags,
            self.qsim_pos,
            self.static_var_tags,
            batch_size=self.batch_size,
            epochs=self.epochs,
            window_size=self.window_size,
            train_pct=self.train_pct,
            valid_pct=self.valid_pct,
            use_cpu=self.use_cpu,
            use_parallel=self.use_parallel,
            do_train=do_train,
            do_simulation=self.do_simulation,
            training_func=training_func,
            filename_base=self.filename_base,
            simulation_phases=self.simulation_phases,
            name_of_saved_model=name_of_saved_model,
        )

        assert len(kge_results[0]) == 4
        assert len(flow_results[0]) == 4
        assert os.path.isfile(name_of_saved_model)

    def test_train_single_catchment(self):
        """Test the regional LSTM model simulation after training."""
        training_func = "kge"
        do_train = True
        input_data_filename = "LSTM_test_data_local.nc"

        # Do a sim with no training
        kge_results, flow_results, name_of_saved_model = control_local_lstm_training(
            input_data_filename,
            self.dynamic_var_tags,
            self.qsim_pos,
            batch_size=self.batch_size,
            epochs=self.epochs,
            window_size=self.window_size,
            train_pct=self.train_pct,
            valid_pct=self.valid_pct,
            use_cpu=self.use_cpu,
            use_parallel=self.use_parallel,
            do_train=do_train,
            do_simulation=self.do_simulation,
            training_func=training_func,
            filename_base=self.filename_base,
            simulation_phases=self.simulation_phases,
        )

        assert len(kge_results) == 4
        assert len(flow_results) == 4
        assert os.path.isfile(name_of_saved_model)

        training_func = "kge"
        do_train = False
        input_data_filename = "LSTM_test_data_local.nc"
        simulation_phases = ["train", "valid", "test", "all"]

        # Do a sim with the trained model on all periods
        kge_results, flow_results, name_of_saved_model = control_local_lstm_training(
            input_data_filename,
            self.dynamic_var_tags,
            self.qsim_pos,
            batch_size=self.batch_size,
            epochs=self.epochs,
            window_size=self.window_size,
            train_pct=self.train_pct,
            valid_pct=self.valid_pct,
            use_cpu=self.use_cpu,
            use_parallel=self.use_parallel,
            do_train=do_train,
            do_simulation=self.do_simulation,
            training_func=training_func,
            filename_base=self.filename_base,
            simulation_phases=simulation_phases,
            name_of_saved_model=name_of_saved_model
        )

        assert len(kge_results) == 4
        assert len(flow_results) == 4
