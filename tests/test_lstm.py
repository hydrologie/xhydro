"""Test suite for LSTM model implementations"""

import os

import pooch

from xhydro.lstm_tools.lstm_controller import (
    control_local_lstm_training,
    control_regional_lstm_training,
)


class TestLstmModels:
    """Test suite for the LSTM models."""

    # Set Github URL for getting files for tests
    GITHUB_URL = "https://github.com/hydrologie/xhydro-testdata"
    BRANCH_OR_COMMIT_HASH = "main"

    # Get data with pooch
    input_data_filename_regional = pooch.retrieve(
        url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/LSTM_data/LSTM_test_data.nc",
        known_hash="md5:e7f1ddba0cf3dc3c5c6aa28a0c504fa2",
    )

    # Get data with pooch
    input_data_filename_local = pooch.retrieve(
        url=f"{GITHUB_URL}/raw/{BRANCH_OR_COMMIT_HASH}/data/LSTM_data/LSTM_test_data_local.nc",
        known_hash="md5:2abfe4dd0287a43c1ab40372f4fc4de8",
    )

    batch_size = 64  # batch size used in the training - multiple of 32
    epochs = 2  # Number of epoch to train the LSTM model
    window_size = 5  # Number of time step (days) to use in the LSTM model
    train_pct = 10  # Percentage of watersheds used for the training
    valid_pct = 5  # Percentage of watersheds used for the validation
    use_cpu = (
        True  # Use CPU as GPU is not guaranteed to be installed with CUDA/CuDNN etc.
    )
    use_parallel = False
    do_simulation = True
    filename_base = "LSTM_results"
    simulation_phases = ["test"]
    # Tags for the dynamic variables in the netcdf files.
    dynamic_var_tags = ["tasmax_MELCC", "rf", "Qsim"]

    # Scale variable according to area. Used for simulated flow inputs.
    qsim_pos = [False, False, False, False, False]

    # static variables used to condition flows on catchment properties
    static_var_tags = [
        "drainage_area",
        "elevation",
        "coniferous_forest",
        "silty_clay_loam",
        "meanPrecip",
    ]

    def test_lstm_controller_regional(self):
        """Test the regional LSTM model implementation."""
        training_func = "nse_scaled"
        do_train = True

        kge_results, flow_results, name_of_saved_model = control_regional_lstm_training(
            self.input_data_filename_regional,
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
            model_structure="dummy_regional_lstm",
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
            self.input_data_filename_regional,
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
            model_structure="dummy_regional_lstm",
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

        # Do a sim with no training
        kge_results, flow_results, name_of_saved_model = control_local_lstm_training(
            self.input_data_filename_local,
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
            model_structure="dummy_local_lstm",
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
        simulation_phases = ["train", "valid", "test", "all"]

        # Do a sim with the trained model on all periods
        kge_results, flow_results, name_of_saved_model = control_local_lstm_training(
            self.input_data_filename_local,
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
            name_of_saved_model=name_of_saved_model,
        )

        assert len(kge_results) == 4
        assert len(flow_results) == 4
