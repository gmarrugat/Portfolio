
import argparse
import os
from util import create_directory, notify

parser = argparse.ArgumentParser(description="Main script for NER using SP or RNN")
parser.add_argument(
    '-m', '--model',
    default='BOTH',
    type=str, required=False, help='Model to be used: SP or RNN (leave empty for BOTH)'
)
parser.add_argument(
    '-d_p', '--data_file',
    default='../data/ner_dataset.csv',
    type=str, required=False, help='Path containing the data'
)
parser.add_argument(
    '-n_s', '--number_sentences',
    default=None,
    type=int, required=False, help='Number of sentences for training (DEFAULT: all dataset)'
)
parser.add_argument(
    '-n_e', '--number_epochs',
    default=10,
    type=int, required=False, help='Number of epochs for training (DEFAULT: 10 epochs)'
)
parser.add_argument(
    '-m_w_s', '--max_wrong_samples',
    default=None,
    type=int, required=False, help='Number (max) of wrong predictions to show for the SP (DEFAULT: None)'
)

args = parser.parse_args()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def print_sep(sep="-", width=100):
    print(sep*width)
    return


if __name__ == "__main__":

    try:
        terminal_columns, terminal_rows = os.get_terminal_size(0)
    except:
        terminal_columns = 100

    params = vars(args)
    print("PARAMETERS:", params)
    create_directory("../results/")
    output_sufix_formatter = "_{}_sent_{}_ep_{}"
    print()

    if params["model"] not in ['BOTH', 'SP', 'RNN']:
        print("Invalid Model!")

    if params["model"] in ['BOTH', 'SP']:
        print_sep(width=terminal_columns)
        print("Starting SP:")
        print_sep(width=terminal_columns)
        params["output_sufix"] = output_sufix_formatter.format("SP", params["number_sentences"], params["number_epochs"])
        import sp
        sp.main(params)
        print_sep(width=terminal_columns)
        print()

    if params["model"] in ['BOTH', 'RNN']:
        print_sep(width=terminal_columns)
        print("Starting RNN:")
        print_sep(width=terminal_columns)
        params["output_sufix"] = output_sufix_formatter.format("RNN", params["number_sentences"], params["number_epochs"])
        import rnn
        rnn.main(params)
        print_sep(width=terminal_columns)
        print()

    try: # requires terminal-notifier. MAC OSX: brew install terminal-notifier
        notify(
            title = 'NER NLP',
            subtitle = params["model"] + " | " + str(params["number_sentences"]) + " | " + str(params["number_epochs"]),
            message = 'Execution completed!'
        )
    except:
        pass
