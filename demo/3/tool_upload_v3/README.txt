Main script: tool_v3.py
The training dataset: bpi2017_train_filtered.csv
The test dataset: bpi2017_test_filtered.csv
The validation dataset: bpi2017_val_filtered.csv

For MM-PRED we use BPIC2012_FULL.csv dataset

Please make sure all the dependencies are installed. You can do this by executing 'pip install -r requirements.txt' 
from your terminal.

Then run 'python tool_v3.py'. The output file with predictions is called tool_v3_predictions.csv. MM-PRED requires supported NVIDIA GPU and CUDA drivers enabled and configured.

The RNN_Time_Prediction notebook contains our RNN prediction model. It is meant be run from within Google Colab environemnt with GPU runtime.