import pytest
import os
from xhydro.lstm_tools.lstm_controller import control_regional_lstm_training


def test_lstm_controller():
    batch_size = 64  # batch size used in the training - multiple of 32
    epochs = 2  # Number of epoch to train the LSTM model
    window_size = 5  # Number of time step (days) to use in the LSTM model
    train_pct = 10  # Percentage of watersheds used for the training
    valid_pct = 5  # Percentage of watersheds used for the validation
    use_cpu = True  # Use CPU as GPU is not guaranteed to be installed with CUDA/CuDNN etc.
    use_parallel = False
    do_train = True
    do_simulation = True
    input_data_filename = "LSTM_test_data.nc"
    filename_base = "LSTM_results"
    simulation_phases = ['test']

    dynamic_var_tags = ["tasmax_MELCC", "tasmin_MELCC", "sf", "rf", "Qsim"]
    qsim_pos = [False, False, False, False, True]  # Scale variable according to area. Used for simulated flow inputs.

    # static variables used to condition flows on catchment properties
    static_var_tags = [
        "drainage_area",
        "elevation",
        "slope",
        "deciduous_forest",
        "coniferous_forest",
        "impervious",
        "loamy_sand",
        "silt",
        "silty_clay_loam",
        "meanPrecip",
        "centroid_lat",
        "centroid_lon",
    ]

    kge_results, flow_results, name_of_saved_model = control_regional_lstm_training(
        input_data_filename,
        dynamic_var_tags,
        qsim_pos,
        static_var_tags,
        batch_size=batch_size,
        epochs=epochs,
        window_size=window_size,
        train_pct=train_pct,
        valid_pct=valid_pct,
        use_cpu=use_cpu,
        use_parallel=use_parallel,
        do_train=do_train,
        do_simulation=do_simulation,
        filename_base=filename_base,
        simulation_phases=simulation_phases
    )

    assert len(kge_results[0]) == 4
    assert len(flow_results[0]) == 4
    assert os.path.isfile(name_of_saved_model)

    # Do a sim with no training
    kge_results, flow_results, name_of_saved_model = control_regional_lstm_training(
            input_data_filename,
            dynamic_var_tags,
            qsim_pos,
            static_var_tags,
            batch_size=batch_size,
            epochs=epochs,
            window_size=window_size,
            train_pct=train_pct,
            valid_pct=valid_pct,
            use_cpu=use_cpu,
            use_parallel=use_parallel,
            do_train=False,
            do_simulation=do_simulation,
            filename_base=filename_base,
            simulation_phases=simulation_phases,
            name_of_saved_model=name_of_saved_model,
        )

    assert len(kge_results[0]) == 4
    assert len(flow_results[0]) == 4
    assert os.path.isfile(name_of_saved_model)
