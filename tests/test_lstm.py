import pytest
from xhydro.lstm import lstm_controller

def test_lstm_controller():
    batch_size = 128  # batch size used in the training - multiple of 32
    epochs = 200  # Number of epoch to train the LSTM model
    window_size = 365  # Number of time step (days) to use in the LSTM model
    train_pct = 60  # Percentage of watersheds used for the training
    valid_pct = 20  # Percentage of watersheds used for the validation
    run_tag = "LSTM_test_run"
    use_parallel = True
    do_train = True
    do_simulation = True
    input_data_filename = 'LSTM_test_data.nc'
    results_path="./"
    filename_base='LSTM_results_'
    
    lstm_controller.control_LSTM_training(input_data_filename,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          window_size=window_size,
                                          train_pct=train_pct,
                                          valid_pct=valid_pct,
                                          run_tag=run_tag,
                                          use_parallel=use_parallel,
                                          do_train=do_train,
                                          do_simulation=do_simulation,
                                          results_path=results_path,
                                          filename_base=filename_base
                                          )
    
if __name__=='__main__':
    test_lstm_controller()
    