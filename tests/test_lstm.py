import pytest

import os
from xhydro.lstm import lstm_controller

def test_lstm_controller():
    batch_size = 64  # batch size used in the training - multiple of 32
    epochs = 2  # Number of epoch to train the LSTM model
    window_size = 5  # Number of time step (days) to use in the LSTM model
    train_pct = 10  # Percentage of watersheds used for the training
    valid_pct = 5  # Percentage of watersheds used for the validation
    use_cpu = True # Use CPU as GPU is not guaranteed to be installed with CUDA/CuDNN etc.
    use_parallel = False
    do_train = True
    do_simulation = True
    input_data_filename = "LSTM_test_data.nc"
    filename_base = "LSTM_results"
    simulation_phases = ['test']

    kge_results, flow_results, name_of_saved_model = lstm_controller.control_LSTM_training(
        input_data_filename,
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
    kge_results, flow_results, name_of_saved_model = lstm_controller.control_LSTM_training(
            input_data_filename,
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
